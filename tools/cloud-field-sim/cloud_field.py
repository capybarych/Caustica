"""Numpy transcription of the volumetric density function in shaders/world/clouds.slang.

Run from anywhere; the shader is located relative to this file. Requires numpy only.

Written against the shader source, line by line: same integer hash (uint32 wraparound included), same
lattice periods, same smoothstep conventions, same stage order. `density()` corresponds to
cloudVolumeDensity() for CLOUD_STYLE_VOLUMETRIC, `formation_thickness()` to cloudFormationThickness(),
`formation_base()` to cloudFormationBase().

Why this exists: the shader can only be *compiled* in this environment, not observed, so the properties
the feature is judged on (does a formation's generated extent match the sliders, does the volume close
inside its own bounds, is the field actually 3D) have to be measured on something. The transcription is
only as good as its fidelity to the shader, which is why it stays deliberately literal — no shortcuts,
no re-derived maths.

Constants are read from the shader source at import time, so a constant that changes there changes here
and a measurement can never silently describe an older tuning.
"""
import re
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "shaders" / "world" / "clouds.slang"


def _constants(text):
    values = {}
    for m in re.finditer(r"public static const float\s+(\w+)\s*=\s*(-?[\d.]+)\s*;", text):
        values[m.group(1)] = float(m.group(2))
    for m in re.finditer(r"public static const int\s+(\w+)\s*=\s*(\d+)\s*;", text):
        values.setdefault(m.group(1), float(m.group(2)))
    return values


C = _constants(SRC.read_text())
CELL = C["CLOUD_CELL_BLOCKS"]
MASK = int(C["CLOUD_PERIOD_CELLS"]) - 1  # CLOUD_CELL_MASK = PERIOD_CELLS - 1
U32 = np.uint32(0xFFFFFFFF)

CELL_MASK = C["CLOUD_EDGE_SOFTNESS"]
ISO_BASE = C["CLOUD_ISO_BASE"]
ISO_COVERAGE = C["CLOUD_ISO_COVERAGE"]
ISO_TOP = C["CLOUD_ISO_TOP"]
RIM_BAND = C["CLOUD_RIM_BAND"]
BULGE = C["CLOUD_BULGE"]
SOFTNESS = C["CLOUD_SHAPE_SOFTNESS"]
WORLEY_MIX = C["CLOUD_WORLEY_MIX"]
WARP_BLOCKS = C["CLOUD_WARP_BLOCKS"]
CROWN_START = C["CLOUD_CROWN_START"]
CROWN_VARIANCE = C["CLOUD_CROWN_VARIANCE"]
EROSION = C["CLOUD_EROSION"]
BASE_JITTER = C["CLOUD_BASE_JITTER_BLOCKS"]


def _mix(h):
    """(h ^ (h >> 13)) * 1274126177 ; h ^ (h >> 16), in uint32."""
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    return h ^ (h >> np.uint32(16))


def hash2(ix, iz):
    c = (ix.astype(np.int64) & MASK).astype(np.uint32)
    d = (iz.astype(np.int64) & MASK).astype(np.uint32)
    h = c * np.uint32(374761393) + d * np.uint32(668265263)
    h = _mix(h)
    return (h & np.uint32(0x00FFFFFF)).astype(np.float64) * (1.0 / 16777216.0)


def hash3(ix, iy, iz):
    a = (ix.astype(np.int64) & MASK).astype(np.uint32)
    b = (iy.astype(np.int64) & MASK).astype(np.uint32)
    c = (iz.astype(np.int64) & MASK).astype(np.uint32)
    h = a * np.uint32(374761393) + b * np.uint32(668265263) + c * np.uint32(1442695041)
    h = _mix(h)
    return (h & np.uint32(0x00FFFFFF)).astype(np.float64) * (1.0 / 16777216.0)


def _frac(x):
    return x - np.floor(x)


def _lerp(a, b, t):
    return a + (b - a) * t


def noise2(x, z):
    ix = np.floor(x); iz = np.floor(z)
    fx = _frac(x); fz = _frac(z)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uz = fz * fz * (3.0 - 2.0 * fz)
    return _lerp(_lerp(hash2(ix, iz), hash2(ix + 1, iz), ux),
                 _lerp(hash2(ix, iz + 1), hash2(ix + 1, iz + 1), ux), uz)


def noise3(x, y, z):
    ix = np.floor(x); iy = np.floor(y); iz = np.floor(z)
    fx = _frac(x); fy = _frac(y); fz = _frac(z)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)
    c = lambda dx, dy, dz: hash3(ix + dx, iy + dy, iz + dz)  # noqa: E731
    a = _lerp(c(0, 0, 0), c(1, 0, 0), ux)
    b = _lerp(c(0, 1, 0), c(1, 1, 0), ux)
    cc = _lerp(c(0, 0, 1), c(1, 0, 1), ux)
    d = _lerp(c(0, 1, 1), c(1, 1, 1), ux)
    return _lerp(_lerp(a, b, uy), _lerp(cc, d, uy), uz)


def worley3(x, y, z):
    """cloudWorley3: F1 over the 27 surrounding cells, with the same box rejection."""
    ix = np.floor(x); iy = np.floor(y); iz = np.floor(z)
    fx = _frac(x); fy = _frac(y); fz = _frac(z)
    best = np.full(np.shape(x), 8.0)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                outside = (np.maximum(np.maximum(dx - fx, fx - (dx + 1.0)), 0.0) ** 2
                           + np.maximum(np.maximum(dy - fy, fy - (dy + 1.0)), 0.0) ** 2
                           + np.maximum(np.maximum(dz - fz, fz - (dz + 1.0)), 0.0) ** 2)
                live = outside < best
                if not live.any():
                    continue
                h = hash3(ix + dx, iy + dy, iz + dz)
                h = (h * 16777216.0).astype(np.uint32)  # recover the 24-bit word
                ox = (h & np.uint32(255)).astype(np.float64) * (1.0 / 255.0)
                oy = ((h >> np.uint32(8)) & np.uint32(255)).astype(np.float64) * (1.0 / 255.0)
                oz = ((h >> np.uint32(16)) & np.uint32(255)).astype(np.float64) * (1.0 / 255.0)
                ex = dx + ox - fx; ey = dy + oy - fy; ez = dz + oz - fz
                best = np.where(live, np.minimum(best, ex * ex + ey * ey + ez * ez), best)
    return np.sqrt(best)


def fbm3(x, y, z, octaves):
    total = 0.0; amp = 0.62; norm = 0.0
    for _ in range(octaves):
        total = total + noise3(x, y, z) * amp
        norm += amp; amp *= 0.45
        x = x * 2.0 + 13.0; y = y * 2.0 + 7.0; z = z * 2.0 + 29.0
    return total / max(norm, 1.0e-5)


def worley_fbm(x, y, z, octaves):
    total = 0.0; amp = 0.62; norm = 0.0
    for _ in range(octaves):
        total = total + (1.0 - np.minimum(worley3(x, y, z), 1.0)) * amp
        norm += amp; amp *= 0.38
        x = x * 2.0 + 5.0; y = y * 2.0 + 11.0; z = z * 2.0 + 3.0
    return total / max(norm, 1.0e-5)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def coverage_field(sx, sz):
    p = sx / (CELL * C["CLOUD_WEATHER_DIV"]); q = sz / (CELL * C["CLOUD_WEATHER_DIV"])
    return (noise2(p, q) * 0.60
            + noise2(p * 2.0 + 11.0, q * 2.0 + 11.0) * 0.28
            + noise2(p * 4.0 + 29.0, q * 4.0 + 29.0) * 0.12)


def formation_thickness(sx, sz, min_t, max_t):
    p = sx / (CELL * C["CLOUD_FORMATION_DIV"]); q = sz / (CELL * C["CLOUD_FORMATION_DIV"])
    n = noise2(p + 53.0, q + 97.0) * 0.65 + noise2(p * 2.0 + 7.0, q * 2.0 + 31.0) * 0.35
    t = np.clip((n - 0.28) * 2.5, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return min_t + (max_t - min_t) * t


def formation_base(sx, sz):
    p = sx / (CELL * C["CLOUD_FORMATION_DIV"]); q = sz / (CELL * C["CLOUD_FORMATION_DIV"])
    n = noise2(p * 2.0 + 17.0, q * 2.0 + 41.0)
    return BASE_JITTER * smoothstep(0.22, 0.78, n)


def density(sx, y, sz, cov=0.55, min_t=16.0, max_t=22.0, morph=0.0, detail=True, octaves=2,
            iso_top=None):
    """cloudVolumeDensity for the volumetric style. y is height above the deck's base plane."""
    iso_top = ISO_TOP if iso_top is None else iso_top
    sx = np.asarray(sx, float); y = np.asarray(y, float); sz = np.asarray(sz, float)
    sx, y, sz = np.broadcast_arrays(sx, y, sz)
    out = np.zeros(sx.shape)

    weather = coverage_field(sx, sz)
    threshold = 1.0 - cov
    mask = smoothstep(threshold, threshold + CELL_MASK, weather)
    if not (mask > 0).any():
        return out
    core = np.clip((weather - threshold) / max(1.0 - threshold, 1.0e-3), 0.0, 1.0)
    rim = smoothstep(threshold, threshold + RIM_BAND, weather)

    thickness = formation_thickness(sx, sz, min_t, max_t)
    base = formation_base(sx, sz)
    rel = y - base
    act = (mask > 0) & (rel > 0) & (rel < thickness)
    if not act.any():
        return out
    sx, sz, mask, core, rim = sx[act], sz[act], mask[act], core[act], rim[act]
    rel, thickness = rel[act], thickness[act]
    hf = rel / thickness

    px, py, pz = sx, rel + morph, sz
    wu = sx / (CELL * C["CLOUD_WARP_DIV"]); wv = sz / (CELL * C["CLOUD_WARP_DIV"])
    wx = noise2(wu, wv) - 0.5
    wz = noise2(wu + 41.0, wv + 17.0) - 0.5
    amp = WARP_BLOCKS * (0.45 + 1.1 * hf)
    px = px + wx * amp
    pz = pz + wz * amp

    sp = CELL * C["CLOUD_SHAPE_DIV"]
    sy = py * C["CLOUD_SHAPE_Y_STRETCH"] / sp
    perlin = fbm3(px / sp, sy, pz / sp, octaves)
    lp = CELL * C["CLOUD_LOBE_DIV"]
    ly = py * C["CLOUD_LOBE_Y_STRETCH"] / lp
    worley = worley_fbm(px / lp, ly, pz / lp, octaves)
    shape = _lerp(perlin, worley, WORLEY_MIX)

    crown_noise = noise2(sx / (CELL * C["CLOUD_LOBE_DIV"]) + 29.0,
                         sz / (CELL * C["CLOUD_LOBE_DIV"]) + 13.0)
    crown = CROWN_START + CROWN_VARIANCE * (crown_noise - 0.5)
    envelope = smoothstep(0.0, 0.12, hf) * (1.0 - smoothstep(crown, 1.0, hf)) * rim

    belly = np.clip(1.0 - np.abs(hf - 0.34) * 2.0, 0.0, 1.0)
    iso_core = ISO_BASE - ISO_COVERAGE * np.sqrt(core) * mask - BULGE * belly
    iso = iso_core + (iso_top - iso_core) * (1.0 - envelope)
    d = smoothstep(iso, iso + SOFTNESS, shape)

    if detail:
        dp = CELL * C["CLOUD_DETAIL_DIV"]
        carve = 1.0 - worley_fbm(px / dp, py / dp, pz / dp, 1)
        erosion = carve * EROSION * (1.0 - d)
        d = np.clip((d - erosion) / np.maximum(1.0 - erosion, 1.0e-4), 0.0, 1.0)

    out[act] = d
    return out


def light_transmittance(tau):
    factors = (1.0, 0.45, 0.18)
    weights = (0.55, 0.30, 0.15)
    return sum(w * np.exp(-tau * f) for w, f in zip(weights, factors))
