# Backgrounds & systematics

Forced photometry only recovers clean fluxes if the pixels handed to the solver
are genuinely source-plus-noise. Two things stand between the raw L2 cutout and
that: a **background** the pipeline refines per cutout, and a handful of SPHEREx
**instrument systematics** you should know are present even after refinement.
Background behavior is set with `PhotometryConfig(bkg_model=...)`; see
{doc}`configuration` for the full field list.

## Background models

Every model starts from the same base: the L2 **ZODI HDU** — the pipeline's
zodiacal-light model of the cutout — is the initial background, and each model
*refines it in place* (a model returns `zodi` plus its own correction, so the
ZODI estimate is never discarded). Selection is `config.bkg_model`, one of
`photutils`, `cwave+photutils`, `plane`, `none`. The fit always excludes bad
pixels: flagged (`MASKBITS`), source-flagged (used only to mask the fit, never
the photometry), and non-finite/non-positive `VARIANCE`.

### `photutils` — 2-D residual background

Runs {class}`photutils.background.Background2D` on the **ZODI-subtracted
residual** `img - zodi` (default box `10`, filter `3`) and adds the result to the
ZODI base. This is the general-purpose choice: it removes smooth large-scale
residual structure the ZODI model missed without assuming a functional form. Use
it when you specifically want the airglow term out of the way — it is the A/B
control for `cwave+photutils`, which is the default and does this same 2-D step
after removing the wavelength profile.

```python
PhotometryConfig(bkg_model="photutils", bkg_box_size=10, bkg_filter_size=3)
```

If the cutout is smaller than `bkg_box_size` in either axis (or `Background2D`
raises), the model returns the ZODI base unchanged.

### `cwave+photutils` — airglow-line removal, then 2-D (default)

SPHEREx sits in low-Earth orbit and sees **geocoronal airglow** — most notably
the He I 1.083 µm line. Because the linear variable filter (LVF) maps a fixed
wavelength onto a fixed detector location, such a line lands on **iso-wavelength
stripes** across the detector rather than tracking sky structure. `Background2D`
alone smears it into the 2-D estimate.

This model first fits a **smooth 1-D profile `B(CWAVE)`** to the residual binned
by each pixel's central wavelength (a monotone PCHIP through per-bin medians,
`bkg_cwave_nbins=48` bins, `bkg_cwave_min_per_bin=20` pixels), subtracts it, and
*then* runs the standard `Background2D` on what remains. It falls back to plain
`photutils` when no `CWAVE` map is present or there are too few valid pixels to
bin. Keep `include_wavelength=True` at retrieval so the `CWAVE` map is in the
cutout.

This is the **default**, and it is not conditional on the field looking like it
has airglow: on the SPHEREx deblending campaign it takes the He 1.083 µm line
residual from +4.84σ to −0.08σ, and it is a wavelength-domain step, so there is
no reason to make it opt-in. Where there is no line to remove it costs little —
on one real 15 px field the two models differ by p90 0.13% in flux. The
exception is a *synthetic* field with no airglow and few background pixels per
source, where the profile fit can absorb source flux (the packaged offline demo
sets `bkg_model="photutils"` for exactly that reason).

```python
PhotometryConfig(bkg_model="cwave+photutils",
                 bkg_cwave_nbins=48, bkg_cwave_min_per_bin=20)
```

### `plane` — weighted tilted plane

A single inverse-variance-weighted least-squares plane `a + b·x + c·y` added to
the ZODI base. Three parameters, no boxes: robust when there are too few clean
background pixels for `Background2D` to bin, or on small/sparse cutouts. It
cannot follow curved residuals; on a degenerate fit it returns the ZODI base
unchanged.

```python
PhotometryConfig(bkg_model="plane")
```

### `none` — ZODI as delivered

Trusts the ZODI HDU with no refinement. Use it for debugging, or when the
residual-fit models would risk absorbing real faint-source flux and you want the
background fully decoupled from the science pixels.

```python
PhotometryConfig(bkg_model="none")
```

### Choosing

| situation | model |
|---|---|
| real SPHEREx data, anything | `cwave+photutils` *(default)* |
| A/B control, or a synthetic field with no airglow | `photutils` |
| small or sparse cutout, few clean pixels | `plane` |
| debugging / keep background decoupled | `none` |

## Known systematics

These are properties of the SPHEREx data and instrument, not of the estimator.
Some the pipeline handles for you; others are irreducible caveats to carry into
interpretation.

### Within-detector wavelength reversal — *handled*

Central wavelength is **not monotonic in detector pixel**: along a detector the
LVF wavelength can increase and then decrease, so raw per-visit points are not
ordered in wavelength. {func}`tractorjax_spherex.spectra.build_spectra` sorts
each source's points by `central_wavelength` when it assembles a spectrum, so the
reversal is transparent downstream — never order a spectrum by pixel, visit, or
cutout index.

### CWAVE varies across a cutout — *sources labelled per-pixel*

A single cutout spans a **range** of central wavelengths (the `cwave_center`
stored per cutout is only the value at its center). The pipeline therefore does
not tag every source in a cutout with one wavelength: at extraction it samples
the per-pixel `CWAVE` (and `CBAND`) map bilinearly **at each source's own pixel
position**, so `central_wavelength`/`bandwidth` are per-source. A cutout with no
`CWAVE` map still gets photometered — its sources are simply labelled NaN
wavelength and dropped when spectra are assembled.

### VARIANCE is affine in the observed rate — *weight-noise bias at the faint end*

The L2 `VARIANCE` plane is affine in the **observed** count rate (a read-noise
floor plus a photon term proportional to the measured source+background rate).
The pipeline weights the fit by `1 / VARIANCE`, so the weights are computed from
the same noisy pixels being fit. That coupling biases fluxes low at the faint
end (the classic Poisson-weight bias): where a pixel fluctuates high it is
down-weighted and vice versa. It is small relative to the per-visit noise and
largely averages out over repeat visits, but do not treat single-visit faint
fluxes as bias-free. Error calibration is a separate matter — see the note in
{doc}`solvers`.

### IRSA absolute flux offset — *estimator-independent*

There is a known absolute flux-scale caveat in the IRSA-delivered SPHEREx L2
calibration: an overall offset that shifts every source together. No choice of
solver removes it, because it is common to all sources and every estimator here
fits the *same* delivered pixels — it cancels in relative quantities (spectral
shape, colors, source-to-source ratios) but propagates into absolute fluxes and
absolute magnitudes. Weigh it when you compare SPHEREx absolute fluxes against an
external absolute scale.

### PSF-header erratum — *handled upstream*

The as-delivered L2 PSF product carried a header erratum. It is corrected in the
`spherex-retrieval` layer when the cutout MEF is written, so the `PSF` /
`PSF_ZONES` HDUs this package reads are already right and no photometry-side
handling is needed. Regenerate cutouts with a current `spherex-retrieval` rather
than patching PSFs by hand. (This is distinct from the CPU-backend
`PixelizedPSF` flux-normalization bug in {doc}`cpu_backend`, which is about
rendering, not the delivered header.)
