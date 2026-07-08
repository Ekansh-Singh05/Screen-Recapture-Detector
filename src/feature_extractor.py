"""Hand-crafted feature extraction for the screen-recapture-detector classifier.

Extracts a 193-dimensional float32 vector from a pre-processed BGR uint8
image.  Every feature is grounded in the physics of screen photography.

Feature groups and dimensions
------------------------------
1.  Sharpness            3   Laplacian variance, Brenner, Tenengrad
2.  FFT Frequency        5   Magnitude spectrum statistics
3.  Moire Detection      2   Periodic peaks + spectral flatness
4.  Gradient Stats       6   Sobel magnitude + X/Y component statistics
5.  Gradient Orientation 8   Orientation histogram (8 x 45-degree bins)
6.  Edge Density         2   Canny fraction + Sobel fraction
7.  LBP Texture         26   Uniform LBP histogram (radius=3, P=24)
8.  Entropy              1   Shannon entropy of greyscale
9.  Brightness           3   HSV-V mean / std / median
10. Contrast             2   Greyscale std + RMS contrast
11. Saturation           3   HSV-S mean / std / P90
12. HSV Statistics       9   H, S, V each: mean, std, skewness
13. LAB Statistics       9   L, A, B each: mean, std, skewness
14. Color Histogram     96   B, G, R each: 32-bin normalised histogram
15. Glare / Reflection   4   Bright-pixel fraction, blob count, max area, specular score
16. Screen Border        3   Border/centre edge ratio, symmetry, corner contrast
17. Perspective Distort  3   Harris corner count, border corner ratio, Hough line count
18. Noise Statistics     4   Noise mean, std, ratio, SNR
19. Texture Statistics   4   Energy, homogeneity proxy, coarseness, local contrast
                       ---
Total                  193

Usage::

    from src.preprocessing import preprocess
    from src.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    img = preprocess("photo.jpg")
    vec = extractor.extract(img)     # shape (193,)  dtype float32
    names = extractor.get_feature_names()  # len 193
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np
from scipy import stats as sp_stats
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

from src.config import CFG
from src.logger import get_logger

log = get_logger(__name__)

# Small epsilon to avoid division-by-zero throughout.
_EPS: float = 1e-9


class FeatureExtractor:
    """Extract a fixed-length float32 feature vector from a BGR uint8 image.

    The extractor is stateless — it holds no mutable state and can be
    shared across threads.

    Attributes:
        _cfg: Alias for ``CFG.features``.
    """

    def __init__(self) -> None:
        self._cfg = CFG.features

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract all feature groups and return one concatenated vector.

        Args:
            img: Pre-processed BGR ``uint8`` image of shape ``(H, W, 3)``.
                Must come from :func:`~src.preprocessing.preprocess`.

        Returns:
            1-D ``float32`` array of shape ``(193,)``.

        Raises:
            ValueError: If *img* is empty or not a 3-channel uint8 array.
        """
        self._validate(img)

        # Derive the colour spaces once; share across groups.
        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv:  np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # LAB in float32 [L:0-100, a/b:-127..127]
        lab:  np.ndarray = cv2.cvtColor(
            img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab
        )

        parts: List[np.ndarray] = [
            self._sharpness(gray),           # 3
            self._fft_frequency(gray),        # 5
            self._moire_detection(gray),      # 2
            self._gradient_stats(gray),       # 6
            self._gradient_orientation(gray), # 8
            self._edge_density(gray),         # 2
            self._lbp_texture(gray),          # 26
            self._entropy(gray),              # 1
            self._brightness(hsv),            # 3
            self._contrast(gray),             # 2
            self._saturation(hsv),            # 3
            self._hsv_statistics(hsv),        # 9
            self._lab_statistics(lab),        # 9
            self._color_histogram(img),       # 96
            self._glare_reflection(gray),     # 4
            self._screen_border(gray),        # 3
            self._perspective_distortion(gray), # 3
            self._noise_statistics(gray),     # 4
            self._texture_statistics(gray),   # 4
        ]

        vector = np.concatenate(parts).astype(np.float32)
        vector = self._sanitise(vector)

        log.debug("Feature vector: %d dims", vector.shape[0])
        return vector

    def get_feature_names(self) -> List[str]:
        """Return a name for every dimension of the feature vector.

        The list length always equals the vector length from :meth:`extract`.

        Returns:
            List of 193 human-readable feature name strings.
        """
        cfg = self._cfg
        n_lbp   = cfg.lbp_n_points + 2       # uniform LBP histogram bins
        n_hist  = cfg.hist_bins               # per colour channel
        n_gorient = cfg.grad_orient_bins

        names: List[str] = []

        # 1. Sharpness (3)
        names += ["sharp_laplacian_var", "sharp_brenner", "sharp_tenengrad"]

        # 2. FFT Frequency (5)
        names += [
            "fft_mag_mean", "fft_mag_std",
            "fft_high_freq_ratio", "fft_low_freq_ratio", "fft_peak_dist",
        ]

        # 3. Moiré Detection (2)
        names += ["moire_peak_count", "moire_spectral_flatness"]

        # 4. Gradient Stats (6)
        names += [
            "grad_sobel_mean", "grad_sobel_std",
            "grad_x_mean", "grad_x_std",
            "grad_y_mean", "grad_y_std",
        ]

        # 5. Gradient Orientation (n_gorient = 8)
        names += [f"grad_orient_{i}" for i in range(n_gorient)]

        # 6. Edge Density (2)
        names += ["edge_canny_density", "edge_sobel_density"]

        # 7. LBP Texture (n_lbp = 26)
        names += [f"lbp_{i}" for i in range(n_lbp)]

        # 8. Entropy (1)
        names += ["entropy_shannon"]

        # 9. Brightness (3)
        names += ["bright_mean", "bright_std", "bright_median"]

        # 10. Contrast (2)
        names += ["contrast_std", "contrast_rms"]

        # 11. Saturation (3)
        names += ["sat_mean", "sat_std", "sat_p90"]

        # 12. HSV Statistics (9)
        for ch in ("h", "s", "v"):
            names += [f"hsv_{ch}_mean", f"hsv_{ch}_std", f"hsv_{ch}_skew"]

        # 13. LAB Statistics (9)
        for ch in ("l", "a", "b"):
            names += [f"lab_{ch}_mean", f"lab_{ch}_std", f"lab_{ch}_skew"]

        # 14. Color Histogram (3 * n_hist = 96)
        for ch in ("b", "g", "r"):
            names += [f"hist_{ch}_{i}" for i in range(n_hist)]

        # 15. Glare / Reflection (4)
        names += [
            "glare_pixel_frac", "glare_blob_count",
            "glare_max_area_ratio", "glare_specular_score",
        ]

        # 16. Screen Border (3)
        names += ["border_edge_ratio", "border_lr_symmetry", "border_corner_contrast"]

        # 17. Perspective Distortion (3)
        names += ["persp_harris_count", "persp_corner_border_ratio", "persp_hough_lines"]

        # 18. Noise Statistics (4)
        names += ["noise_mean", "noise_std", "noise_ratio", "noise_snr"]

        # 19. Texture Statistics (4)
        names += ["tex_energy", "tex_homogeneity", "tex_coarseness", "tex_local_contrast"]

        return names

    # ------------------------------------------------------------------ #
    #  Feature group implementations                                       #
    # ------------------------------------------------------------------ #

    def _sharpness(self, gray: np.ndarray) -> np.ndarray:
        """Three complementary sharpness measures.

        Why three?  Each metric responds differently to the type of blur:
        * Laplacian variance: sensitive to all frequencies.
        * Brenner: efficient, strong for periodic textures.
        * Tenengrad: robust to isolated noise spikes.

        Screen re-captures suffer double-blur: the camera must focus on a
        flat plane at a fixed distance (removing depth-of-field variation)
        AND the screen's anti-aliasing filter softens pixel edges.

        Returns:
            Array [laplacian_var, brenner_focus, tenengrad].  Shape (3,).
        """
        g = gray.astype(np.float64)

        # Laplacian variance — high = sharp.
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brenner focus: sum of squared row-wise differences 2 pixels apart.
        brenner = float(np.mean((g[:-2, :] - g[2:, :]) ** 2))

        # Tenengrad: mean energy of Sobel gradient magnitude.
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx ** 2 + gy ** 2))

        return np.array([lap_var, brenner, tenengrad])

    def _fft_frequency(self, gray: np.ndarray) -> np.ndarray:
        """Frequency-domain analysis via 2-D DFT.

        Screen photographs carry elevated high-frequency energy from the
        pixel grid, subpixel colour filters (RGB stripe / PenTile), and
        LCD backlight interference patterns.

        Features:
            mag_mean: Mean of the (DC-zeroed) magnitude spectrum.
            mag_std: Std of the magnitude spectrum.
            high_freq_ratio: Fraction of energy in the outer ring
                (beyond ``fft_high_freq_frac`` of the spectrum radius).
            low_freq_ratio: Fraction of energy in the central disc.
            peak_dist: Distance of the dominant peak from DC.

        Returns:
            Array of shape (5,).
        """
        fft   = np.fft.fft2(gray.astype(np.float32))
        fft_s = np.fft.fftshift(fft)
        mag   = np.abs(fft_s).astype(np.float64)

        h, w  = mag.shape
        cy, cx = h // 2, w // 2

        # Zero out DC component so statistics are not dominated by it.
        mag_nodc = mag.copy()
        mag_nodc[cy, cx] = 0.0

        mag_mean = float(mag_nodc.mean())
        mag_std  = float(mag_nodc.std())

        # Build a radial distance map.
        yy, xx = np.ogrid[:h, :w]
        r_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = np.sqrt(cy ** 2 + cx ** 2)
        split = r_max * self._cfg.fft_high_freq_frac

        low_mask  = r_map <= split
        high_mask = ~low_mask
        # Exclude DC from both masks.
        low_mask[cy, cx] = False

        total_energy = mag_nodc.sum() + _EPS
        high_ratio = float(mag_nodc[high_mask].sum() / total_energy)
        low_ratio  = float(mag_nodc[low_mask].sum()  / total_energy)

        # Distance of the brightest non-DC peak from the centre.
        mag_nodc[cy, cx] = 0.0
        peak_idx  = np.unravel_index(np.argmax(mag_nodc), mag_nodc.shape)
        peak_dist = float(np.sqrt((peak_idx[0] - cy) ** 2 + (peak_idx[1] - cx) ** 2))

        return np.array([mag_mean, mag_std, high_ratio, low_ratio, peak_dist])

    def _moire_detection(self, gray: np.ndarray) -> np.ndarray:
        """Detect Moire-like periodic patterns in the frequency spectrum.

        A screen's regular pixel grid creates multiple satellite peaks in
        the 2-D DFT around the fundamental grid frequency.  We count peaks
        that are 3-sigma above the background and measure spectral flatness.

        Features:
            peak_count: Number of significant frequency peaks (>3 sigma).
                Normalised by image area so it is resolution-independent.
            spectral_flatness: Geometric mean / Arithmetic mean of the
                magnitude spectrum.  Close to 1 for white noise (natural
                image); close to 0 for tonal/periodic content (screen).

        Returns:
            Array of shape (2,).
        """
        fft_s = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        mag   = np.abs(fft_s).astype(np.float64)
        h, w  = mag.shape

        # Remove DC.
        mag[h // 2, w // 2] = 0.0

        # Significant peaks: pixels whose value exceeds mean + 3 * std.
        threshold = mag.mean() + 3.0 * mag.std()
        peak_count = float(np.sum(mag > threshold)) / mag.size

        # Spectral flatness (Wiener entropy).
        flat = mag.ravel() + _EPS
        geom_mean  = float(np.exp(np.mean(np.log(flat))))
        arith_mean = float(flat.mean())
        spectral_flatness = geom_mean / (arith_mean + _EPS)

        return np.array([peak_count, spectral_flatness])

    def _gradient_stats(self, gray: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude and X/Y component statistics.

        Screen content (UI text, icons) generates strong axis-aligned
        gradients (0 degrees and 90 degrees).  Natural scenes produce
        gradients at all orientations, yielding a more uniform distribution
        and lower mean-magnitude-to-std ratio.

        Returns:
            Array [sobel_mean, sobel_std, gx_mean, gx_std, gy_mean, gy_std].
            Shape (6,).
        """
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        return np.array([
            mag.mean(), mag.std(),
            gx.mean(),  gx.std(),
            gy.mean(),  gy.std(),
        ])

    def _gradient_orientation(self, gray: np.ndarray) -> np.ndarray:
        """Weighted gradient orientation histogram.

        Bins the gradient angle into ``grad_orient_bins`` uniform buckets
        over [-pi, pi], weighted by gradient magnitude.  Screen photos
        skew toward 0 and pi/2 bins (horizontal/vertical UI elements).

        Returns:
            Normalised histogram of shape ``(grad_orient_bins,)``.
        """
        n_bins = self._cfg.grad_orient_bins
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(gy, gx)
        mag   = np.sqrt(gx ** 2 + gy ** 2)

        hist, _ = np.histogram(
            angle.ravel(), bins=n_bins, range=(-np.pi, np.pi),
            weights=mag.ravel(),
        )
        return (hist / (hist.sum() + _EPS)).astype(np.float64)

    def _edge_density(self, gray: np.ndarray) -> np.ndarray:
        """Canny edge pixel fraction and Sobel edge pixel fraction.

        Screen device bezels create a sharp rectangular border with very
        high edge density that does not appear in genuine camera shots of
        natural scenes.

        Returns:
            Array [canny_density, sobel_density].  Shape (2,).
        """
        edges_canny = cv2.Canny(gray, 50, 150)
        canny_density = float(edges_canny.sum()) / (gray.size * 255.0 + _EPS)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(gx ** 2 + gy ** 2)
        # Fraction of pixels with sobel magnitude above 10% of max.
        threshold = 0.1 * sobel_mag.max() if sobel_mag.max() > 0 else 1.0
        sobel_density = float(np.sum(sobel_mag > threshold)) / gray.size

        return np.array([canny_density, sobel_density])

    def _lbp_texture(self, gray: np.ndarray) -> np.ndarray:
        """Uniform Local Binary Pattern histogram.

        LBP is a rotation-invariant texture descriptor.  Screen subpixel
        structures (RGB stripe pattern, pixel grid) produce characteristic
        LBP codes that are statistically absent in real-scene textures.

        Using ``method='uniform'`` limits the histogram to P+2 bins,
        suppressing non-uniform (noisy) patterns while retaining the
        stable, meaningful ones.

        Returns:
            Normalised histogram of shape ``(lbp_n_points + 2,)``.
            i.e. shape (26,) with default P=24.
        """
        cfg = self._cfg
        lbp = local_binary_pattern(
            gray, P=cfg.lbp_n_points, R=cfg.lbp_radius, method=cfg.lbp_method
        )
        n_bins = cfg.lbp_n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        return (hist.astype(np.float64) / (hist.sum() + _EPS))

    def _entropy(self, gray: np.ndarray) -> np.ndarray:
        """Shannon entropy of the greyscale image.

        Higher entropy = more information = more varied content.  A screen
        showing a single face has lower entropy than a genuine outdoor
        scene.  Also, JPEG compression artefacts from multiple generations
        of re-compression (common in screen captures) reduce entropy.

        Returns:
            Array [shannon_entropy].  Shape (1,).
        """
        return np.array([float(shannon_entropy(gray))])

    def _brightness(self, hsv: np.ndarray) -> np.ndarray:
        """HSV Value-channel statistics.

        Backlit screens produce a characteristically high mean brightness
        with a narrow distribution (low std) — they emit light rather than
        reflecting it, so there are no deep shadows.

        Returns:
            Array [v_mean, v_std, v_median].  Shape (3,).
            Values are normalised to [0, 1].
        """
        v = hsv[:, :, 2].astype(np.float64) / 255.0
        return np.array([v.mean(), v.std(), float(np.median(v))])

    def _contrast(self, gray: np.ndarray) -> np.ndarray:
        """Greyscale standard deviation and RMS contrast.

        Monitors typically compress dynamic range compared to real-world
        lighting.  Low contrast_std and low RMS both indicate a narrow
        tonal range, consistent with a screen shot.

        Returns:
            Array [gray_std, rms_contrast].  Shape (2,).
        """
        g = gray.astype(np.float64) / 255.0
        gray_std = float(g.std())
        rms      = float(np.sqrt(np.mean((g - g.mean()) ** 2)))
        return np.array([gray_std, rms])

    def _saturation(self, hsv: np.ndarray) -> np.ndarray:
        """HSV Saturation-channel statistics.

        Camera white-balance + monitor colour-gamut mismatch tends to
        reduce saturation in screen photos.  Additionally, screen glare
        de-saturates pixels near bright reflections.

        Returns:
            Array [s_mean, s_std, s_p90].  Shape (3,).
        """
        s = hsv[:, :, 1].astype(np.float64) / 255.0
        return np.array([s.mean(), s.std(), float(np.percentile(s, 90))])

    def _hsv_statistics(self, hsv: np.ndarray) -> np.ndarray:
        """Mean, std, and skewness for each of the H, S, V channels.

        Skewness is particularly diagnostic:
        * H skewness: LCD blue-shift creates positive H skew.
        * S skewness: Glare creates negatively skewed saturation.
        * V skewness: Even backlight creates symmetric V distribution.

        Returns:
            Array of shape (9,): [H_mean, H_std, H_skew, S_mean, ...].
        """
        out: List[float] = []
        for ch_idx in range(3):
            ch = hsv[:, :, ch_idx].astype(np.float64).ravel() / 255.0
            out += [float(ch.mean()), float(ch.std()), float(sp_stats.skew(ch))]
        return np.array(out)

    def _lab_statistics(self, lab: np.ndarray) -> np.ndarray:
        """Mean, std, and skewness for each of the L, a, b channels.

        CIE LAB separates luminance (L) from colour (a, b), making it
        better than RGB for detecting colour casts from monitor backlights.
        Skewness in the a (green-red) and b (blue-yellow) channels
        signals the warm/cool shift introduced by LCD/OLED displays.

        Args:
            lab: LAB float32 image.

        Returns:
            Array of shape (9,): [L_mean, L_std, L_skew, a_mean, ...].
        """
        out: List[float] = []
        for ch_idx in range(3):
            ch = lab[:, :, ch_idx].astype(np.float64).ravel()
            out += [float(ch.mean()), float(ch.std()), float(sp_stats.skew(ch))]
        return np.array(out)

    def _color_histogram(self, img: np.ndarray) -> np.ndarray:
        """Normalised per-channel (B, G, R) color histograms.

        Screens produce characteristic colour banding:
        * sRGB gamut clipping creates sharp histogram boundaries.
        * Multiple JPEG re-compression rounds create comb-like gaps.
        * Monitor backlights shift the blue channel distribution rightward.

        Returns:
            Concatenated [B_hist, G_hist, R_hist] of shape (3 * hist_bins,)
            = (96,) with default hist_bins=32.
        """
        n_bins = self._cfg.hist_bins
        hists: List[np.ndarray] = []
        for ch_idx in range(3):
            ch = img[:, :, ch_idx]
            h = cv2.calcHist([ch], [0], None, [n_bins], [0, 256]).ravel()
            h = h.astype(np.float64) / (ch.size + _EPS)
            hists.append(h)
        return np.concatenate(hists)

    def _glare_reflection(self, gray: np.ndarray) -> np.ndarray:
        """Detect bright specular reflections on screen glass.

        Screen surfaces introduce glare from ambient light.  Real camera
        photos may have highlights but rarely have the large, flat, nearly-
        white regions characteristic of monitor reflection.

        Features:
            glare_pixel_frac: Fraction of pixels above glare_threshold.
            glare_blob_count: Number of large connected bright blobs
                (normalised by image area).
            glare_max_area_ratio: Area of the largest bright blob as a
                fraction of the total image area.
            specular_score: Mean intensity of the top-1% brightest pixels.

        Returns:
            Array of shape (4,).
        """
        threshold = self._cfg.glare_threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        glare_pixel_frac = float(binary.sum()) / (gray.size * 255.0 + _EPS)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = max(4, gray.size // 5000)  # at least 4 px; scales with resolution
        blob_areas = [
            stats[i, cv2.CC_STAT_AREA]
            for i in range(1, n_labels)
            if stats[i, cv2.CC_STAT_AREA] >= min_area
        ]
        glare_blob_count = float(len(blob_areas)) / (gray.size + _EPS)
        max_area_ratio   = float(max(blob_areas, default=0)) / (gray.size + _EPS)

        # Top-1 % specular score.
        k = max(1, gray.size // 100)
        flat = gray.ravel().astype(np.float64)
        specular_score = float(np.partition(flat, -k)[-k:].mean()) / 255.0

        return np.array([glare_pixel_frac, glare_blob_count, max_area_ratio, specular_score])

    def _screen_border(self, gray: np.ndarray) -> np.ndarray:
        """Detect a rectangular screen bezel or device frame.

        A handheld screen photo typically shows the device frame: a
        sharp rectangular edge near the image boundary.  This creates an
        abnormally high edge density in the image margins compared to a
        genuine camera photo where important content fills the frame.

        Features:
            border_edge_ratio: Canny edge density in the margin ring vs.
                the inner content area.
            border_lr_symmetry: Absolute difference between left and
                right margin edge densities (low = symmetric bezel).
            border_corner_contrast: Mean absolute brightness difference
                between corner regions and the image centre.

        Returns:
            Array of shape (3,).
        """
        m = self._cfg.border_margin
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)

        # Border ring (outer m pixels on all sides).
        border_mask = np.zeros_like(edges, dtype=bool)
        border_mask[:m, :] = True
        border_mask[-m:, :] = True
        border_mask[:, :m] = True
        border_mask[:, -m:] = True

        centre_mask = ~border_mask

        def _density(mask: np.ndarray) -> float:
            n = mask.sum()
            return float(edges[mask].sum()) / (n * 255.0 + _EPS) if n > 0 else 0.0

        border_density = _density(border_mask)
        centre_density = _density(centre_mask)
        border_edge_ratio = border_density / (centre_density + _EPS)

        # Left vs right symmetry.
        left_density  = _density(border_mask & (np.arange(w)[None, :] < m))
        right_density = _density(border_mask & (np.arange(w)[None, :] >= w - m))
        border_lr_symmetry = float(abs(left_density - right_density))

        # Corner vs centre brightness contrast.
        corner_size = m
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:],
        ]
        corner_mean = float(np.mean([c.mean() for c in corners]))
        centre_crop = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        border_corner_contrast = float(abs(corner_mean - centre_crop.mean()) / 255.0)

        return np.array([border_edge_ratio, border_lr_symmetry, border_corner_contrast])

    def _perspective_distortion(self, gray: np.ndarray) -> np.ndarray:
        """Estimate geometric distortion from handheld screen photography.

        A camera held at an angle to a screen introduces keystoning: lines
        that are parallel on screen converge in the image.  Three proxies:

        * Harris corner count: More corners = more geometric structure
          (UI elements, screen bezel edges).
        * Corner border ratio: Ratio of Harris corners near the image
          border vs. the interior — a framed screen concentrates corners
          near the edges.
        * Hough line count: More straight lines = more screen structure.

        Returns:
            Array of shape (3,).
        """
        cfg = self._cfg

        # Harris corner detector.
        gray_f = np.float32(gray)
        harris = cv2.cornerHarris(
            gray_f, cfg.harris_block_size, cfg.harris_ksize, cfg.harris_k
        )
        # Threshold at 1% of max response.
        threshold = 0.01 * harris.max() if harris.max() > 0 else 1e-6
        corner_mask = harris > threshold
        total_corners = float(corner_mask.sum())
        harris_count = total_corners / (gray.size + _EPS)

        # Corners near the border vs interior.
        m = self._cfg.border_margin
        h, w = gray.shape
        border_ring = np.zeros_like(corner_mask, dtype=bool)
        border_ring[:m, :] = True; border_ring[-m:, :] = True
        border_ring[:, :m] = True; border_ring[:, -m:] = True
        border_corners  = float(corner_mask[border_ring].sum())
        interior_corners = float(corner_mask[~border_ring].sum())
        corner_border_ratio = border_corners / (interior_corners + _EPS)

        # Probabilistic Hough lines (straight line count).
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=30, maxLineGap=10,
        )
        hough_count = float(len(lines)) / (gray.size + _EPS) if lines is not None else 0.0

        return np.array([harris_count, corner_border_ratio, hough_count])

    def _noise_statistics(self, gray: np.ndarray) -> np.ndarray:
        """Estimate image noise characteristics.

        Camera sensor noise (photon shot noise + read noise) has a
        Gaussian distribution.  Screen re-capture noise is a combination
        of camera noise and display quantisation noise, which has a
        different spectral profile — it tends to be correlated with the
        pixel grid frequency rather than spatially white.

        Noise is estimated as the residual after Gaussian low-pass filtering.

        Features:
            noise_mean: Mean absolute residual.
            noise_std: Std of the residual.
            noise_ratio: Noise energy as a fraction of signal energy.
            noise_snr: 20 * log10(signal_RMS / noise_RMS).  In dB.

        Returns:
            Array of shape (4,).
        """
        ksize = self._cfg.noise_blur_kernel
        blurred = cv2.GaussianBlur(gray, ksize, 0)
        residual = gray.astype(np.float64) - blurred.astype(np.float64)

        noise_mean = float(np.abs(residual).mean())
        noise_std  = float(residual.std())

        signal_rms = float(np.sqrt(np.mean(gray.astype(np.float64) ** 2))) + _EPS
        noise_rms  = float(np.sqrt(np.mean(residual ** 2))) + _EPS

        noise_ratio = noise_rms / signal_rms
        noise_snr   = float(20.0 * np.log10(signal_rms / noise_rms))

        return np.array([noise_mean, noise_std, noise_ratio, noise_snr])

    def _texture_statistics(self, gray: np.ndarray) -> np.ndarray:
        """Simple texture descriptors without GLCM (fast, O(HW)).

        GLCM-based features would be more principled but take ~50 ms on a
        256x256 image.  These O(HW) proxies capture the same information:

        * Energy: Sum of squared pixel intensities (normalised).
            High energy = uniform/repetitive texture (screen grid).
        * Homogeneity: Inverse of local variance mean.
            Low variance = smooth / homogeneous regions (screen background).
        * Coarseness: Mean of local pixel range in 5x5 blocks.
            Low coarseness = fine-grained / uniform texture.
        * Local contrast: Std of local standard deviations (5x5 windows).
            Low = globally uniform contrast (screen), High = varied (scene).

        Returns:
            Array [energy, homogeneity, coarseness, local_contrast]. (4,).
        """
        g = gray.astype(np.float64) / 255.0

        # Energy.
        energy = float(np.mean(g ** 2))

        # Local variance (5x5 window).
        mu    = cv2.blur(gray.astype(np.float64), (5, 5))
        mu2   = cv2.blur((gray.astype(np.float64)) ** 2, (5, 5))
        local_var = np.maximum(mu2 - mu ** 2, 0.0)

        homogeneity   = float(1.0 / (local_var.mean() + 1.0))  # avoid /0
        local_contrast = float(np.sqrt(local_var).std())

        # Coarseness: mean of local range (max - min) in 5x5 blocks.
        ksize = (5, 5)
        local_max = cv2.dilate(gray.astype(np.float64), np.ones(ksize))
        local_min = cv2.erode(gray.astype(np.float64), np.ones(ksize))
        coarseness = float((local_max - local_min).mean() / 255.0)

        return np.array([energy, homogeneity, coarseness, local_contrast])

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(img: np.ndarray) -> None:
        """Raise ValueError if *img* is not a valid BGR uint8 image.

        Args:
            img: Candidate image array.

        Raises:
            ValueError: On any format violation.
        """
        if img is None or img.size == 0:
            raise ValueError("Empty image received.")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) uint8 image; got shape {img.shape}."
            )
        if img.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 image; got dtype {img.dtype}."
            )

    @staticmethod
    def _sanitise(vec: np.ndarray) -> np.ndarray:
        """Replace NaN and Inf values with 0.0.

        This is a safety net — individual feature methods should not
        produce NaN/Inf, but a corrupted input image could trigger edge
        cases in OpenCV or scipy.  Returning NaN to the classifier would
        silently corrupt predictions.

        Args:
            vec: Raw feature vector.

        Returns:
            Same array with NaN and Inf replaced by 0.0.
        """
        bad = ~np.isfinite(vec)
        if bad.any():
            n = int(bad.sum())
            log.warning(
                "Sanitised %d non-finite value(s) in feature vector (replaced with 0.0).", n
            )
            vec = vec.copy()
            vec[bad] = 0.0
        return vec
