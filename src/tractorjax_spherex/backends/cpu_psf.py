"""A flux-correct oversampled PSF for the upstream (CPU) Tractor.

Upstream ``tractor.psf.PixelizedPSF`` handles an oversampled PSF stamp
(``sampling < 1``) by point-sampling it onto the native grid but **omits the
1/sampling**2 pixel-area factor**, so a unit-flux point-source template comes out
too faint by ``oversample**2`` and forced fluxes come out high by the same
factor (~25x at 5x oversampling). The tractor-jax engine fixed this; this class
ports that fix to the CPU Tractor.

:class:`OversampledPixelizedPSF` overrides only the two ``sampling != 1``
methods: both the point-source path and the Fourier path (used for galaxies)
integer-factor-block-integrate the oversampled PSF to native pixels (accurate
low-resolution rendering — the pixel response applied exactly once) and apply
the correct flux scale; an effective PSF (``pixel_integrated=True``) is
block-centre-sampled instead on both paths.

Usage::

    psf = OversampledPixelizedPSF(psf5x_stamp, sampling=0.2)   # 5x oversampled
    # then hand `psf` to tractor.Image(...); do NOT pass sampling=0.2 to the
    # unpatched tractor.psf.PixelizedPSF.
"""

from __future__ import annotations

import numpy as np
from tractor.patch import Patch
from tractor.psf import PixelizedPSF, lanczos_shift_image


class OversampledPixelizedPSF(PixelizedPSF):
    """PixelizedPSF for an oversampled stamp, with flux normalization corrected."""

    def __init__(self, img, sampling=1.0, Lorder=3, pixel_integrated=False):
        # The upstream (legacy) tractor.psf.PixelizedPSF knows nothing of PSF
        # kinds; the flag is handled entirely in this subclass.
        super().__init__(img, sampling=sampling, Lorder=Lorder)
        #: True for an EFFECTIVE PSF (the SPHEREx R7 ePSF): the stamp already
        #: contains the pixel response, so both rendering paths sample the
        #: block centres (times k^2) instead of block-integrating.
        self.pixel_integrated = bool(pixel_integrated)
        # The integer-factor block-integration path (the one forced photometry
        # uses) centers the binned PSF at native index (nativeW-1)/2, which only
        # equals the patch origin's implied center (nativeW//2) for ODD native
        # sizes. An even native size mis-centers the model by half a native pixel
        # (~0.5px astrometric error, ~10% flux bias), so reject it loudly rather
        # than bias silently. The standard 51x51 @ sampling=0.2 -> 11 is odd.
        # PixelizedPSF sets self.radius = hypot(H/2, W/2) in STAMP pixels, but
        # every consumer (galaxy patch halfsize above all) treats getRadius()
        # as NATIVE pixels. For an oversampled stamp that inflates the PSF
        # radius by 1/sampling (51x51 @ 5x: 36 instead of 7.2 native px), so
        # every galaxy model patch grows by ~+29 px per side and the forced
        # solve's sparse system gets ~13x the nonzeros -- measured 90+ s per
        # full-depth cutout against ~5 s once corrected.
        if sampling != 1.0:
            self.radius = float(np.hypot(*[d / 2.0 for d in img.shape])) * sampling
        if sampling != 1.0 and (self.nativeH % 2 == 0 or self.nativeW % 2 == 0):
            raise ValueError(
                f"OversampledPixelizedPSF native output size "
                f"{self.nativeH}x{self.nativeW} must be odd for correct centering "
                f"(got even from stamp {img.shape} at sampling={sampling}). Choose "
                f"a stamp size so ceil(size*sampling) is odd.")

    def __str__(self):
        return "OversampledPixelizedPSF"

    def _native_from_canvas(self, img, dx, dy, k):
        """Shift ``img`` by the sub-pixel offset at oversampled resolution and
        bring it to native pixels: the block sum for an optical PSF, the
        block-centre sample times ``k^2`` for an effective one (the ePSF value
        at the pixel centre is the fraction of the flux in that pixel).
        Returned in the block-sum normalisation, i.e. to be divided by ``k^2``
        and multiplied by ``scale`` like the block sum."""
        target_h = self.nativeH * k
        target_w = self.nativeW * k
        h, w = img.shape
        margin = int(np.ceil(max(abs(dx * k), abs(dy * k)))) + 10
        canvas_h = max(h, target_h) + 2 * margin
        canvas_w = max(w, target_w) + 2 * margin
        crop_x0 = (canvas_w - target_w) // 2
        crop_y0 = (canvas_h - target_h) // 2
        target_center_x = crop_x0 + (target_w - 1) / 2.0
        target_center_y = crop_y0 + (target_h - 1) / 2.0
        desired_x = target_center_x + dx * k
        desired_y = target_center_y + dy * k
        pw = round(desired_x - (w // 2))
        ph = round(desired_y - (h // 2))
        pw = max(0, min(canvas_w - w, pw))
        ph = max(0, min(canvas_h - h, ph))
        pad_img = np.zeros((canvas_h, canvas_w), dtype=img.dtype)
        pad_img[ph:ph + h, pw:pw + w] = img
        shift_x = desired_x - ((w // 2) + pw)
        shift_y = desired_y - ((h // 2) + ph)
        shifted = lanczos_shift_image(pad_img, shift_x, shift_y)
        crop = shifted[crop_y0:crop_y0 + target_h, crop_x0:crop_x0 + target_w]
        crop = crop.reshape(self.nativeH, k, self.nativeW, k)
        if self.pixel_integrated:
            wts = np.zeros(k, dtype=crop.dtype)
            lo, hi = (k - 1) // 2, k // 2
            wts[lo] += 0.5 if lo != hi else 1.0
            wts[hi] += 0.5 if lo != hi else 0.0
            return np.einsum("ajbk,j,k->ab", crop, wts, wts) * (k * k)
        return crop.sum(axis=(1, 3))

    def _getOversampledPointSourcePatch(self, px, py, minval=0., modelMask=None,
                                        radius=None, **kwargs):
        img = self.getImage(px, py)
        ix = round(float(px))
        iy = round(float(py))
        dx = px - ix
        dy = py - iy

        scale = 1.0 / self.sampling ** 2
        if modelMask is not None:
            mh, mw = modelMask.shape
            mx0, my0 = modelMask.x0, modelMask.y0
            xl, yl, native_img = self._sampleImage(
                img, dx, dy, xlo=mx0 - ix, ylo=my0 - iy, width=mw, height=mh)
            return Patch(xl + ix, yl + iy, native_img * scale)

        factor = 1.0 / self.sampling
        is_integer_factor = abs(factor - round(factor)) < 1e-4

        if is_integer_factor and radius is None:
            # Block-integrate the oversampled PSF to native pixels (flux-exact):
            # shift by the sub-pixel offset at oversampled resolution, then sum
            # each k x k block. This keeps the PSF pixel-integrated at native
            # scale rather than point-sampled. (Effective PSF: the block-centre
            # sample instead, see _native_from_canvas.)
            k = round(factor)
            img = self._native_from_canvas(img, dx, dy, k) / (k ** 2)   # * scale below
            xl = -(self.nativeW // 2)
            yl = -(self.nativeH // 2)
        else:
            xl, yl, img = self._sampleImage(img, dx, dy)

        img = img * scale
        x0 = ix + xl
        y0 = iy + yl

        if radius is not None:
            R = int(np.ceil(radius))
            H, W = img.shape
            cx, cy = W // 2, H // 2
            xlo = max(cx - R, 0)
            ylo = max(cy - R, 0)
            img = img[ylo:min(cy + R + 1, H - 1), xlo:min(cx + R + 1, W - 1)]
            x0 += xlo
            y0 += ylo
        return Patch(x0, y0, img)

    def _getOversampledFourierTransform(self, px, py, radius):
        sz = self.getFourierTransformSize(radius)
        key = (sz, px, py)
        if key in self.fftcache:
            return self.fftcache[key]
        dx = px - int(px)
        dy = py - int(py)
        factor = 1.0 / self.sampling
        if abs(factor - round(factor)) < 1e-4:
            # Integer oversampling: the SAME native kernel as the point-source
            # path -- the block sum of the Lanczos-shifted stamp for an optical
            # PSF (the pixel response applied once), the block-centre samples
            # for an effective one. Until 0.3.1 this path point-sampled the
            # optical stamp (_sampleImage), which dropped the pixel response
            # from every galaxy model on this backend while point sources kept
            # it: galaxies came out too peaked (the tractor_jax comment on the
            # same bug: ~5 %) and disagreed with the JAX backend.
            k = round(factor)
            img = self._native_from_canvas(self.getImage(px, py), dx, dy, k) / (k ** 2)
        else:
            _, _, img = self._sampleImage(None, dx, dy)
        img = img * (1.0 / self.sampling ** 2)   # the flux-normalization fix
        pad, cx, cy = self._padInImage(sz, sz, img=img)
        cx += dx
        cy += dy
        P = np.fft.rfft2(pad).astype(np.complex64)
        pH, pW = pad.shape
        v = np.fft.rfftfreq(pW)
        w = np.fft.fftfreq(pH)
        rtn = P, (cx, cy), (pH, pW), (v, w)
        self.fftcache[key] = rtn
        return rtn
