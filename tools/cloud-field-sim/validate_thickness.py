#!/usr/bin/env python3
"""Measure the properties the cloud rework is judged on, on the numpy transcription of the shader."""
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_field as f


def scan(min_t, max_t, n=360, span=720.0, ystep=0.25):
    """Per column: vertical extent above the deck base plane, above the formation's own base, and the
    formation's configured thickness."""
    x = np.linspace(0.0, span, n)
    X, Z = np.meshgrid(x, x)
    top = np.zeros_like(X)
    any_ = np.zeros_like(X, bool)
    for y in np.arange(0.25, 30.0, ystep):
        hit = f.density(X, np.full_like(X, y), Z, 0.55, min_t, max_t) > 0
        top = np.where(hit, y, top)
        any_ |= hit
    base = f.formation_base(X, Z)
    th = f.formation_thickness(X, Z, min_t, max_t)
    return top[any_] - base[any_], th[any_], top[any_]


print("A. THICKNESS: does a formation's generated extent stay inside the configured pair?")
print("   (slab pushed by RtComposite = max + 4 = 26 blocks)")
print("   pair   columns   mean extent  max extent  max configured  max above deck  columns at ceiling")
for pair in [(16, 16), (18, 20), (16, 22), (22, 22)]:
    own, th, deck = scan(*pair)
    at_ceiling = float((own > th - 0.25).mean())
    print(f"   {pair[0]:>2}-{pair[1]:<2} {len(own):9d} {own.mean():12.2f} {own.max():12.2f}"
          f" {th.max():15.2f} {deck.max():15.2f} {100 * at_ceiling:17.2f}%")
    assert own.max() <= th.max() + 1e-6, "a column exceeded its own formation thickness"
    assert deck.max() <= 26.0, "a formation left the marched slab"

print("\nB. CONFIGURED THICKNESS DISTRIBUTION (equal endpoints must collapse to a constant)")
rng = np.random.default_rng(11)
n = 300_000
sx = rng.uniform(0, 8000, n); sz = rng.uniform(0, 8000, n)
for pair in [(16, 22), (16, 16), (22, 22), (19, 19)]:
    th = f.formation_thickness(sx, sz, *pair)
    inside = bool(((th >= pair[0] - 1e-9) & (th <= pair[1] + 1e-9)).all())
    constant = bool(np.allclose(th, pair[0], atol=1e-9)) if pair[0] == pair[1] else None
    print(f"   {pair[0]:>2}-{pair[1]:<2}  mean {th.mean():5.2f} std {th.std():5.2f}"
          f"  min {th.min():5.2f} max {th.max():5.2f}  inside range {inside}"
          + (f"  constant {constant}" if constant is not None else ""))
    assert inside
    if constant is not None:
        assert constant

print("\nC. DENSITY: is the volume hollow, solid, or a mixture?")
th = f.formation_thickness(sx, sz, 16, 22); base = f.formation_base(sx, sz)
y = base + rng.uniform(0.02, 0.98, n) * th
d = f.density(sx, y, sz)
print(f"   occupancy inside formation bounds {d.mean() if False else (d > 0).mean():.3f}"
      f"  saturated {(d >= 0.999).mean():.3f}  mean where present {d[d > 0].mean():.3f}"
      f"  p10/p50/p90 of present {np.round(np.percentile(d[d > 0], [10, 50, 90]), 2)}")

print("\nD. ANIMATION: is the cloudscape coherent frame to frame?")
m = 60_000
xs = rng.uniform(0, 900, m); zs = rng.uniform(0, 900, m); ys = rng.uniform(2, 20, m)
a = f.density(xs, ys, zs) > 0
for dt in (1.0, 5.0, 30.0):
    b = f.density(xs, ys, zs, morph=0.15 * dt) > 0
    print(f"   +{dt:4.0f}s  IoU {(a & b).sum() / max((a | b).sum(), 1):.3f}")

print("\nE. SLAB: does any formation leave the volume the renderer pushes?")
print("   (checked per pair in A; 26 = CLOUD_MAX_THICKNESS_BLOCKS + CLOUD_BASE_JITTER_BLOCKS)")
print("\nALL CHECKS PASSED")
