"""User-facing configuration for the SPHEREx forced-photometry pipeline.

:class:`PhotometryConfig` captures every optimization option in one place, with
the frozen "F3" blind-production defaults (solver ``eigfloor``, tile 15 / halo 3
/ pad-bucket 32 / prefetch thread / fp32). It serialises to/from YAML and TOML
so a run is fully reproducible from a single file. See the *Choosing a solver*
and *Configuration* pages in the docs for the trade-offs behind each field.
"""

from __future__ import annotations

import json
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


@dataclass
class PhotometryConfig:
    """All forced-photometry options with blind-production (F3) defaults.

    Attributes are grouped as solver / catalog-depth / tiling-batching /
    background / rendering / execution. The defaults reproduce the calibrated
    blind photo-z product; change ``solver`` (and read ``docs/solvers``) to
    select a different estimator.
    """

    # --- solver -----------------------------------------------------------
    solver: str = "eigfloor"
    eig_floor: float = 1e-2
    lasso_alpha: float | str = "auto"
    lasso_n_iter: int = 1000
    protect_zmag_max: float = 20.0
    prior_sigma_frac: float = 0.5
    prior_sigma_min_ujy: float = 5.0

    # --- catalog depth ----------------------------------------------------
    # None => fit the full catalog (production blind regime). A positive value
    # fits only sources with z-band AB mag brighter than it.
    fit_zmag_max: float | None = None

    # --- tiling / batching (F3) ------------------------------------------
    tile_size: int = 15
    tile_halo: int = 3
    pad_bucket: int = 32
    tile_chunk: int = 0
    # None => auto policy (full-depth caps only when fitting the full catalog);
    # a positive int fixes the width; 0 disables that cap.
    max_ps_cap: int | None = None
    max_gal_cap: int | None = None
    max_mog_k_cap: int | None = None

    # --- background -------------------------------------------------------
    bkg_model: str = "photutils"
    bkg_box_size: int = 10
    bkg_filter_size: int = 3
    bkg_cwave_nbins: int = 48
    bkg_cwave_min_per_bin: int = 20

    # --- rendering --------------------------------------------------------
    # psf_sampling = native-pixel size per PSF-stamp pixel (0.2 => the PSF cube
    # is 5x oversampled). fixed_max_factor is the oversampled rendering factor.
    psf_sampling: float = 0.2
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
        # The upstream `tractor` CPU backend has no eigfloor/lasso/prior solvers
        # (it does a single WLS solve). Fail early with an actionable message
        # instead of a confusing solver error deep in the backend.
        if self.backend == "cpu-tractor" and self.solver != "linear":
            raise ConfigError(
                f"backend='cpu-tractor' supports only solver='linear' (a single "
                f"weighted least-squares solve); got solver={self.solver!r}. Use "
                f"backend='jax' (device='cpu' works with no GPU) for eigfloor / "
                f"eigfloor_prior / lasso. See docs/cpu_backend.")
        if self.backend == "cpu-tractor" and self.pad_bucket:
            # Harmless but meaningless; tiling/batching knobs don't apply.
            pass

    def resolved_caps(self) -> tuple[int | None, int | None, int | None]:
        """Return the (ps, gal, mog_k) caps after the depth-conditional policy.

        Full-depth fits (``fit_zmag_max`` unset and no ``pad_bucket``) default to
        the field caps so the solver compiles once; a z-cut or an explicit
        ``pad_bucket`` turns the caps off. Explicit non-None values override; a
        value <= 0 disables that cap.
        """
        full_depth = self.fit_zmag_max is None or self.fit_zmag_max <= 0
        if self.pad_bucket:
            full_depth = False
        out = []
        for value, default in ((self.max_ps_cap, MAX_PS_CAP),
                               (self.max_gal_cap, MAX_GAL_CAP),
                               (self.max_mog_k_cap, MAX_MOG_K_CAP)):
            if value is None:
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
    def from_dict(cls, data: dict) -> "PhotometryConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "PhotometryConfig":
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
