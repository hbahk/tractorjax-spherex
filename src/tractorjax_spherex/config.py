"""User-facing configuration for the SPHEREx forced-photometry pipeline.

:class:`PhotometryConfig` captures every optimization option in one place, with
the SPHEREx deblending campaign's **configuration of record** as defaults:
solver ``eigfloor`` on the catalog truncated at z-band AB 21
(``fit_zmag_max=21``), tile 15 / halo 3 / pad-bucket 32 / prefetch thread /
fp32, ``cwave+photutils`` background, PSF zone interpolation and core
re-registration on. It serialises to/from YAML and TOML so a run is fully
reproducible from a single file. See the *Choosing a solver* and
*Configuration* pages in the docs for the trade-offs behind each field.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .constants import MAX_GAL_CAP, MAX_MOG_K_CAP, MAX_PS_CAP

SOLVERS = ("linear", "eigfloor", "eigfloor_prior", "lasso")
BACKENDS = ("jax", "cpu-tractor")
BKG_MODELS = ("photutils", "cwave+photutils", "plane", "none")
DEVICES = ("auto", "gpu", "cpu")
PRECISIONS = ("fp32", "fp64")
PREFETCH = ("thread", "sync")


class ConfigError(ValueError):
    """Raised for an invalid or unsupported PhotometryConfig combination."""


class CapExceededError(RuntimeError):
    """A tile needs more flux slots than the configured fixed cap allows.

    Carries the width that was actually required so the message is actionable
    (and so a caller can retry with a big-enough cap) instead of the caller
    having to parse a number out of an engine string.
    """

    def __init__(self, kind: str, needed: int, cap: int,
                 cutout_index: int | None = None):
        self.kind = kind              # "ps" | "gal" | "mog_k"
        self.needed = int(needed)
        self.cap = int(cap)
        self.cutout_index = cutout_index
        where = "" if cutout_index is None else f"cutout {cutout_index}: "
        super().__init__(
            f"{where}max_{kind}={self.needed} exceeds max_{kind}_cap={self.cap}. "
            f"This cutout would be SKIPPED, silently shrinking the product. Fix "
            f"by setting max_{kind}_cap='auto' (measures the field and sizes the "
            f"cap), max_{kind}_cap={self.needed} or larger, or pad_bucket=32 "
            f"(the default, which turns fixed caps off).")


@dataclass
class PhotometryConfig:
    """All forced-photometry options, defaulting to the configuration of record.

    Attributes are grouped as solver / catalog-depth / tiling-batching /
    background / rendering / execution. The defaults reproduce the calibrated
    blind photo-z product of the SPHEREx deblending campaign: ``eigfloor`` on
    the catalog truncated at ``fit_zmag_max=21``. Set ``fit_zmag_max=None`` to
    fit the full catalog (the ``eigfloor_prior`` trade-off arm), and change
    ``solver`` (read ``docs/solvers``) to select a different estimator.
    """

    # --- solver -----------------------------------------------------------
    solver: str = "eigfloor"
    eig_floor: float = 1e-2
    lasso_alpha: float | str = "auto"
    lasso_n_iter: int = 1000
    protect_zmag_max: float = 20.0
    # Plateau scan over 0.05-0.5 against WISE/IRAC broadband anchors (2026-08-03):
    # bright-end fidelity improves monotonically as the prior tightens and
    # saturates below ~0.15, where `prior_sigma_min_ujy` (not this fraction) sets
    # the width for the faint population; photo-z metrics are insensitive across
    # the whole range. 0.15 is the plateau entry point and the adopted default.
    prior_sigma_frac: float = 0.15
    prior_sigma_min_ujy: float = 5.0

    # --- catalog depth ----------------------------------------------------
    # Fit only sources with z-band AB mag brighter than this; None => the full
    # catalog. 21 is the configuration of record: at full Legacy Survey depth
    # a SPHEREx cutout holds ~22 catalog sources per PSF core, the flux solve is
    # under-determined, and the paired scatter on common sources is 1.3x wider
    # than at m_z<21 for the same estimator (2026-09 campaign; DR10 sources
    # fainter than 21 are below the single-visit SPHEREx noise). Set None for
    # the regularized full-catalog arm (`eigfloor_prior`). A catalog with no
    # usable `flux_z` at all skips the cut with a warning instead of emptying.
    fit_zmag_max: float | None = 21.0

    # --- tiling / batching ------------------------------------------------
    tile_size: int = 15
    tile_halo: int = 3
    pad_bucket: int = 32
    tile_chunk: int = 0
    # None => auto policy (full-depth caps only when fitting the full catalog);
    # a positive int fixes the width; 0 disables that cap; "auto" measures the
    # field's real densest-tile occupancy up front (see occupancy.py) and sizes
    # the cap from it, which is the only setting guaranteed not to drop cutouts.
    max_ps_cap: int | str | None = None
    max_gal_cap: int | str | None = None
    max_mog_k_cap: int | str | None = None
    # Headroom multiplier applied to a measured ("auto") cap.
    cap_auto_margin: float = 1.1
    # A cutout that raises during build or solve is skipped and reported. With
    # strict=True the first failure aborts the run instead, so a partial product
    # can never be mistaken for a complete one.
    strict: bool = False
    # cpu-tractor backend only: solve tile by tile on the SAME tile grid as the
    # JAX backend (one upstream optimize_forced_photometry per tile) instead of
    # one whole-cutout joint solve. Tens of fluxes per tile converge fast and
    # are well conditioned, where the whole-cutout system at full catalog depth
    # is both slow and degenerate. Set False for the whole-cutout ("global
    # geometry") path, which is kept as the cross-check of the tiled result.
    # Ignored by the JAX backend, which is always tiled.
    cpu_tiling: bool = True
    # cpu-tractor tiled path: fit one free constant per tile alongside the
    # fluxes (upstream tractor's ``sky=True``). The JAX backend's tiled solve
    # ALWAYS carries this background column, so this on is what makes the two
    # backends the same solve rather than the same geometry with different
    # nuisance parameters. On by default for that reason (measured: it cuts the
    # median CPU-vs-JAX disagreement on S/N>5 sources by 2-4x at no time cost).
    # Inert with cpu_tiling=False, which has no tiles and is kept as the
    # reproduction of pre-tiling products; the backend logs that once per run.
    cpu_tile_background: bool = True

    # --- background -------------------------------------------------------
    # CWAVE-aware by default: the LVF maps wavelength onto detector stripes, so
    # an airglow line (He I 1.083 um above all) is a stripe that a 2-D
    # background cannot see. Fitting the wavelength profile first takes the
    # He-line residual from +4.84 sigma to -0.08 sigma; the campaign carries it
    # unconditionally. `photutils` is the A/B control, not the recommendation.
    bkg_model: str = "cwave+photutils"
    bkg_box_size: int = 10
    bkg_filter_size: int = 3
    bkg_cwave_nbins: int = 48
    bkg_cwave_min_per_bin: int = 20

    # --- rendering --------------------------------------------------------
    # psf_sampling = native-pixel size per PSF-stamp pixel (0.2 => the PSF cube
    # is 5x oversampled). fixed_max_factor is the oversampled rendering factor.
    psf_sampling: float = 0.2
    # Spatially-varying PSF: blend the delivered zone kernels bilinearly at
    # each tile's core centre (SPHEREx Sky Simulator convention, clamped at the
    # lattice edge) instead of rounding the tile to one zone. Blended in the
    # Fourier domain on the JAX backend, per grid cell on the CPU backend
    # (measured +15% pipelined wall clock at 9-12 zones on the JAX backend
    # after the grouped-GEMM blend). A no-op on cutouts whose bundle carries a
    # single PSF zone -- which is what a retrieval without a zone margin gives
    # for any cutout smaller than the ~185 px zone pitch.
    psf_zone_interp: bool = True
    # Correct the delivered PSF's core registration: shift each zone kernel by
    # minus its measured core offset (calib/psf_core_offsets.ecsv, 726/726
    # detector-zone cells) plus the fixed 0.05 native px 10x->5x binning grid
    # term. JAX backend: per-basis-element Fourier phase ramps; CPU backend:
    # Lanczos-shifted stamps. ON by default: without it every source is rendered
    # ~0.05 native px (~0.3") off its catalog position, which is a real flux
    # error on a blend -- measured on one blended QSO, turning it off moves the
    # target's per-visit fluxes by up to 13-15% (p90) on both backends. The
    # SPHEREx deblending campaign runs with it on. Requires psf_zone_interp on
    # the JAX backend (the shifts ride on the zone basis).
    #
    # The table was measured on the QR2 OPTICAL PSF product and the grid term
    # belongs to its 10x->5x binning; neither applies to the R7 effective PSF
    # (QR3/DR1 bundles, PSFKIND='EPSF'), which is anchored to the R7 astrometry
    # (measured registration <= 0.02 px). "auto" (default) applies the shift to
    # optical cutouts and skips it on effective ones; True forces it and raises
    # on an effective cutout; False never shifts.
    psf_core_shift: bool | str = "auto"
    fixed_max_factor: float = 5.0

    # --- execution --------------------------------------------------------
    backend: str = "jax"
    device: str = "auto"
    precision: str = "fp32"
    prefetch: str = "thread"
    gpu_mem_fraction: float | None = None
    gpu_preallocate: bool = False

    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ConfigError` on any invalid or unsupported setting."""
        if self.solver not in SOLVERS:
            raise ConfigError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        if self.backend not in BACKENDS:
            raise ConfigError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.bkg_model not in BKG_MODELS:
            raise ConfigError(
                f"bkg_model must be one of {BKG_MODELS}, got {self.bkg_model!r}")
        if self.device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.precision not in PRECISIONS:
            raise ConfigError(
                f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.prefetch not in PREFETCH:
            raise ConfigError(f"prefetch must be one of {PREFETCH}, got {self.prefetch!r}")
        if self.tile_size <= 0:
            raise ConfigError("tile_size must be positive")
        if self.tile_halo < 0:
            raise ConfigError("tile_halo must be non-negative")
        for name in ("max_ps_cap", "max_gal_cap", "max_mog_k_cap"):
            value = getattr(self, name)
            if isinstance(value, str) and value != "auto":
                raise ConfigError(
                    f"{name} must be an int, None, or 'auto'; got {value!r}")
        if self.cap_auto_margin < 1.0:
            raise ConfigError(
                f"cap_auto_margin must be >= 1.0 (it is headroom on a measured "
                f"width); got {self.cap_auto_margin}")
        # The upstream `tractor` CPU backend has no eigfloor/lasso/prior solvers
        # (it does a single WLS solve). Fail early with an actionable message
        # instead of a confusing solver error deep in the backend.
        if self.backend == "cpu-tractor" and self.solver != "linear":
            raise ConfigError(
                f"backend='cpu-tractor' supports only solver='linear' (a single "
                f"weighted least-squares solve); got solver={self.solver!r}. Use "
                f"backend='jax' (device='cpu' works with no GPU) for eigfloor / "
                f"eigfloor_prior / lasso. See docs/cpu_backend.")
        # On the JAX backend the core shifts are applied as phase ramps on the
        # zone basis, so they need that basis to exist. This combination became
        # reachable by accident when psf_core_shift moved to True by default --
        # a user who only sets psf_zone_interp=False would otherwise hit the
        # error deep in the backend, one cutout into the run.
        if self.psf_core_shift not in (True, False, "auto"):
            raise ConfigError(
                f"psf_core_shift must be True, False or 'auto'; got {self.psf_core_shift!r}")
        if (self.backend == "jax" and self.psf_core_shift
                and not self.psf_zone_interp):
            raise ConfigError(
                "psf_core_shift=True requires psf_zone_interp=True on "
                "backend='jax' (the core shifts ride on the zone basis). Set "
                "psf_core_shift=False to keep psf_zone_interp=False, or use "
                "backend='cpu-tractor', which supports the shift standalone.")
        # cpu_tile_background is deliberately NOT cross-validated against
        # backend / cpu_tiling. It is on by default, so an error on either
        # combination would reject `backend="jax"` and the documented
        # `cpu_tiling=False` cross-check out of the box. Instead: the JAX
        # backend always fits the column (so True is simply true there), and
        # CpuTractorBackend logs once per run when cpu_tiling=False makes it
        # inert — a log line rather than the silent no-op this package has been
        # bitten by before.

    def wants_auto_caps(self) -> bool:
        """True if any cap is ``"auto"`` and the caps are actually in force."""
        if not self._caps_active():
            return False
        return any(isinstance(v, str) and v == "auto"
                   for v in (self.max_ps_cap, self.max_gal_cap,
                             self.max_mog_k_cap))

    def _caps_active(self) -> bool:
        """Whether the depth-conditional policy applies caps at all."""
        if self.pad_bucket:
            return False
        return self.fit_zmag_max is None or self.fit_zmag_max <= 0

    def resolved_caps(self, occupancy=None
                      ) -> tuple[int | None, int | None, int | None]:
        """Return the (ps, gal, mog_k) caps after the depth-conditional policy.

        Full-depth fits (``fit_zmag_max`` unset and no ``pad_bucket``) default to
        the field caps so the solver compiles once; a z-cut or an explicit
        ``pad_bucket`` turns the caps off. Explicit non-None values override; a
        value <= 0 disables that cap.

        ``"auto"`` sizes the cap from a measured
        :class:`~tractorjax_spherex.occupancy.Occupancy` (times
        ``cap_auto_margin``) instead of the built-in default, so the field's own
        crowding decides the width and no cutout is dropped. Passing ``"auto"``
        without an ``occupancy`` is a programming error: the pipeline measures
        it before the first solve.
        """
        full_depth = self._caps_active()
        measured = (None, None, None)
        if occupancy is not None:
            measured = (occupancy.max_ps, occupancy.max_gal, None)
        out: list[int | None] = []
        for value, default, meas in (
                (self.max_ps_cap, MAX_PS_CAP, measured[0]),
                (self.max_gal_cap, MAX_GAL_CAP, measured[1]),
                (self.max_mog_k_cap, MAX_MOG_K_CAP, measured[2])):
            if isinstance(value, str):
                if value != "auto":
                    raise ConfigError(
                        f"cap must be an int, None, or 'auto'; got {value!r}")
                if not full_depth:
                    out.append(None)
                elif meas is None:
                    raise ConfigError(
                        "cap='auto' needs a measured occupancy; "
                        "tractorjax_spherex.occupancy.measure_occupancy runs "
                        "this automatically inside run_photometry")
                else:
                    # Round before ceil: 100 * 1.1 is 110.00000000000001 in
                    # binary, and a cap of 111 would misreport the headroom.
                    # max(1, ...) keeps a source-free field from asking for
                    # a zero-width batch.
                    scaled = round(meas * self.cap_auto_margin, 6)
                    out.append(max(1, math.ceil(scaled)))
            elif value is None:
                out.append(default if full_depth else None)
            elif value <= 0:
                out.append(None)
            else:
                out.append(int(value))
        return tuple(out)  # type: ignore[return-value]

    def solver_spec(self) -> dict:
        """Return the engine solver spec dict (matches the production driver)."""
        if self.solver == "lasso":
            try:
                alpha: float | str = float(self.lasso_alpha)
            except (TypeError, ValueError):
                alpha = self.lasso_alpha  # "auto"
            return {"kind": "lasso", "alpha": alpha,
                    "debias_signfree": "protected", "n_iter": self.lasso_n_iter}
        if self.solver == "eigfloor":
            return {"kind": "eigfloor", "floor": self.eig_floor}
        if self.solver == "eigfloor_prior":
            return {"kind": "eigfloor_prior", "floor": self.eig_floor}
        return {"kind": "linear"}

    # --- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict) -> PhotometryConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> PhotometryConfig:
        """Load a config from a ``.yaml``/``.yml`` or ``.toml`` file."""
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text) or {}
        elif path.suffix == ".toml":
            import tomllib
            data = tomllib.loads(text)
        else:
            raise ConfigError(
                f"unsupported config extension {path.suffix!r}; use .yaml/.yml/.toml")
        return cls.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
