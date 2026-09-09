"""Command-line interface: ``tractorjax-spherex {retrieve,fetch-catalog,run,spectra}``.

``run`` builds a :class:`~tractorjax_spherex.config.PhotometryConfig` from an
optional ``--config`` file overridden by any explicit flags, so a full run is
reproducible from a single YAML/TOML plus the command line.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import PhotometryConfig
from .version import __version__


def _add_run_args(p):
    p.add_argument("--cutouts-dir", required=True,
                   help="directory of cutout_*.fits (+ summary.ecsv)")
    p.add_argument("--catalog", required=True, help="reference catalog path")
    p.add_argument("--ra", type=float, default=None, help="target/field-center RA (deg)")
    p.add_argument("--dec", type=float, default=None, help="target/field-center Dec (deg)")
    p.add_argument("--output", default=None, help="output parquet path")
    p.add_argument("--resume", action="store_true", help="skip cutouts already in --output")
    p.add_argument("--max-cutouts", type=int, default=None)
    p.add_argument("--config", default=None, help="PhotometryConfig YAML/TOML file")
    # config overrides (only applied when explicitly passed)
    p.add_argument("--solver", default=None,
                   choices=["linear", "eigfloor", "eigfloor_prior", "lasso"])
    p.add_argument("--eig-floor", type=float, default=None)
    p.add_argument("--lasso-alpha", default=None)
    p.add_argument("--protect-zmag-max", type=float, default=None)
    p.add_argument("--fit-zmag-max", type=float, default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--tile-halo", type=int, default=None)
    p.add_argument("--pad-bucket", type=int, default=None)
    p.add_argument("--tile-chunk", type=int, default=None)
    p.add_argument("--bkg-model", default=None,
                   choices=["photutils", "cwave+photutils", "plane", "none"])
    p.add_argument("--backend", default=None, choices=["jax", "cpu-tractor"])
    p.add_argument("--device", default=None, choices=["auto", "gpu", "cpu"])
    p.add_argument("--precision", default=None, choices=["fp32", "fp64"])
    p.add_argument("--prefetch", default=None, choices=["thread", "sync"])
    p.add_argument("--gpu-mem-fraction", type=float, default=None)
    p.add_argument("--gpu-preallocate", action="store_true", default=None)
    p.add_argument("--max-ps-cap", default=None, type=_cap,
                   help="fixed point-source batch width; 'auto' measures the "
                        "field, 0 disables the cap")
    p.add_argument("--max-gal-cap", default=None, type=_cap,
                   help="fixed galaxy batch width; 'auto' measures the field, "
                        "0 disables the cap")
    p.add_argument("--strict", action="store_true", default=None,
                   help="abort on the first failing cutout instead of skipping "
                        "it (never write a partial product)")


def _cap(value):
    """Cap flag: the literal 'auto', or an int."""
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"cap must be an integer or 'auto'; got {value!r}") from None


_CONFIG_FLAGS = ("solver", "eig_floor", "lasso_alpha", "protect_zmag_max",
                 "fit_zmag_max", "tile_size", "tile_halo", "pad_bucket",
                 "tile_chunk", "bkg_model", "backend", "device", "precision",
                 "prefetch", "gpu_mem_fraction", "gpu_preallocate",
                 "max_ps_cap", "max_gal_cap", "strict")


def _config_from_args(args) -> PhotometryConfig:
    config = (PhotometryConfig.from_file(args.config)
              if args.config else PhotometryConfig())
    for name in _CONFIG_FLAGS:
        val = getattr(args, name, None)
        if val is not None:
            setattr(config, name, val)
    config.validate()
    return config


def _cmd_run(args):
    from .pipeline import run_photometry
    config = _config_from_args(args)
    target = (args.ra, args.dec) if args.ra is not None and args.dec is not None else None
    results = run_photometry(args.cutouts_dir, args.catalog, config,
                             target=target, output=args.output,
                             resume=args.resume,
                             max_cutouts=args.max_cutouts, progress=True)
    # A partial product must not look like a successful run to a shell script.
    n_failed = int(results.meta.get("tractorjax_spherex.n_cutouts_failed", 0))
    if n_failed:
        print(f"INCOMPLETE: {n_failed} cutouts were skipped; the output is "
              f"missing them (see the log, and the complete=False flag in the "
              f"parquet metadata).", file=sys.stderr)
        return 2
    return 0


def _cmd_retrieve(args):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from spherex_retrieval import retrieve
    coord = SkyCoord(ra=args.ra * u.deg, dec=args.dec * u.deg)
    size = args.size_arcsec * u.arcsec if args.size_arcsec else args.size_pix
    retrieve(coord, size, output_dir=args.out, include_wavelength=True,
             include_sapm=True, subset_psf=True)


def _cmd_fetch_catalog(args):
    from .catalog.fetch_ls import fetch_ls_dr10
    fetch_ls_dr10(args.ra, args.dec, radius_deg=(args.radius_arcsec / 3600.0
                  if args.radius_arcsec else None),
                  cutout_pixels=args.cutout_pixels, name=args.name, out=args.out)


def _cmd_spectra(args):
    from astropy.table import Table

    from .spectra import bin_spectrum, build_spectra
    phot = Table.read(args.photometry)
    ids = None if args.all else ([args.id] if args.id is not None else None)
    spectra = build_spectra(phot, ids=ids, min_snr=args.min_snr)
    if args.out:
        for sid, spec in spectra.items():
            spec.write(f"{args.out}_{sid}.ecsv", overwrite=True)
    if args.plot:
        import matplotlib.pyplot as plt

        from .spectra import plot_spectrum
        for sid, spec in spectra.items():
            ax = plot_spectrum(spec, binned=bin_spectrum(spec, dlam=args.bin)
                               if args.bin else None, label=f"id={sid}")
            ax.figure.savefig(f"{args.plot}_{sid}.png", dpi=150)
            plt.close(ax.figure)
    print(f"Built {len(spectra)} spectra")


def build_parser():
    p = argparse.ArgumentParser(
        prog="tractorjax-spherex",
        description="Forced photometry / spectrophotometry from SPHEREx L2 images")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("retrieve", help="download L2 cutouts (spherex-retrieval)")
    pr.add_argument("--ra", type=float, required=True)
    pr.add_argument("--dec", type=float, required=True)
    pr.add_argument("--size-pix", type=int, default=100)
    pr.add_argument("--size-arcsec", type=float, default=None)
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=_cmd_retrieve)

    pf = sub.add_parser("fetch-catalog", help="fetch a Legacy Survey DR10 catalog")
    pf.add_argument("--ra", type=float, required=True)
    pf.add_argument("--dec", type=float, required=True)
    pf.add_argument("--cutout-pixels", type=int, default=100)
    pf.add_argument("--radius-arcsec", type=float, default=None)
    pf.add_argument("--name", default="field")
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=_cmd_fetch_catalog)

    prun = sub.add_parser("run", help="run forced photometry over a field")
    _add_run_args(prun)
    prun.set_defaults(func=_cmd_run)

    ps = sub.add_parser("spectra", help="assemble spectra from a photometry parquet")
    ps.add_argument("--photometry", required=True)
    ps.add_argument("--id", type=int, default=None)
    ps.add_argument("--all", action="store_true")
    ps.add_argument("--min-snr", type=float, default=None)
    ps.add_argument("--bin", type=float, default=None, help="bin width (micron)")
    ps.add_argument("--out", default=None, help="prefix for per-source ecsv output")
    ps.add_argument("--plot", default=None, help="prefix for per-source png output")
    ps.set_defaults(func=_cmd_spectra)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    main()
