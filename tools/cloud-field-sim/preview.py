#!/usr/bin/env python3
"""Offline ray-marched preview of the cloud field in shaders/world/clouds.slang.

Mirrors cloudMarch's structure (Beer-Lambert extinction, dual-lobe HG phase, powder term, three-term
multiple-scattering) but single-scattered and with a stand-in sky, so it is a silhouette/contrast
check, NOT a picture of what the game will draw. Used only to compare tunings against each other.

Usage: python3 preview.py [outdir]   (default /tmp/cloudsim)
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_field as f

SIGMA = 0.115
HG_G, HG_BACK = 0.72, -0.28
POWDER = 0.7
DECK_BOTTOM, DECK_TOP = 0.0, 30.0


def hg(cos_t, g):
    g2 = g * g
    d = 1.0 + g2 - 2.0 * g * cos_t
    return (1.0 - g2) / (4.0 * np.pi * np.maximum(d * np.sqrt(np.maximum(d, 1e-6)), 1e-4))


def phase(cos_t):
    return hg(cos_t, HG_BACK) + (hg(cos_t, HG_G) - hg(cos_t, HG_BACK)) * 0.7


def ms(tau):
    return sum(w * np.exp(-tau * fac) for w, fac in zip((0.55, 0.30, 0.15), (1.0, 0.45, 0.18)))


def sky(elev):
    up = np.clip(elev, 0, 1)[:, None]
    return np.array([0.60, 0.67, 0.75])[None, :] * (1 - up) + np.array([0.22, 0.40, 0.70])[None, :] * up


def render(cam, target, w=360, h=240, fov=72, steps=24, lsteps=5, sun=(0.45, 0.70, 0.20),
           min_t=16.0, max_t=22.0, iso_top=None, coverage=0.55):
    cam = np.asarray(cam, float)
    fwd = np.asarray(target, float) - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    th = np.tan(np.radians(fov) * 0.5)
    xs = (np.arange(w) + 0.5) / w * 2 - 1
    ys = 1 - (np.arange(h) + 0.5) / h * 2
    X, Y = np.meshgrid(xs * th * (w / h), ys * th)
    dirs = (fwd[None, None, :] + X[..., None] * right + Y[..., None] * up)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs.reshape(-1, 3)
    org = np.broadcast_to(cam, dirs.shape).copy()

    light = np.asarray(sun, float); light /= np.linalg.norm(light)
    dy = np.where(np.abs(dirs[:, 1]) < 1e-5, 1e-5, dirs[:, 1])
    ta = (DECK_BOTTOM - org[:, 1]) / dy
    tb = (DECK_TOP - org[:, 1]) / dy
    ten = np.maximum(np.minimum(ta, tb), 0.0)
    tex = np.minimum(np.maximum(ta, tb), 2500.0)
    valid = tex > ten + 0.01
    step = (tex - ten) / steps
    dither = ((np.arange(dirs.shape[0]) * 2654435761) % 1009) / 1009.0

    kw = dict(min_t=min_t, max_t=max_t, iso_top=iso_top, cov=coverage)
    ph = phase(np.clip(dirs @ light, -1, 1))
    col = np.zeros((dirs.shape[0], 3))
    trans = np.ones(dirs.shape[0])
    for i in range(steps):
        p = org + dirs * (ten + step * (i + dither))[:, None]
        d = f.density(p[:, 0], p[:, 1], p[:, 2], **kw)
        live = d > 0
        if not live.any():
            continue
        lstep = 26.0 / (lsteps * max(abs(light[1]), 0.35))
        tau = np.zeros_like(d)
        for k in range(lsteps):
            s = lstep * (k + 0.5)
            tau += f.density(p[:, 0] + light[0] * s, p[:, 1] + light[1] * s,
                             p[:, 2] + light[2] * s, detail=False, **kw)
        lt = ms(tau * SIGMA * lstep)
        powder = 1.0 - POWDER * np.exp(-d * 4.0)
        sunl = (lt * ph * powder * 26.0)[:, None] * np.array([1.0, 0.97, 0.90])
        amb = 0.42 * (0.25 + 0.75 * np.clip(p[:, 1] / DECK_TOP, 0, 1))[:, None] \
            * sky(np.clip(0.4 + 0.6 * np.clip(p[:, 1] / DECK_TOP, 0, 1), 0, 1))
        ext = np.exp(-d * SIGMA * step)
        col += (trans * (1 - ext))[:, None] * (sunl + amb)
        trans *= ext
        if np.all(trans < 0.01):
            break

    bg = sky(np.clip(dirs[:, 1] * 2.5, 0, 1))
    col += trans[:, None] * bg
    col = np.where(valid[:, None], col, bg)
    img = (1 - np.exp(-np.clip(col, 0, None) * 1.7)) ** (1 / 2.2)
    return (np.clip(img, 0, 1).reshape(h, w, 3) * 255).astype(np.uint8)


VIEWS = {
    "under": ((360, -55, 380), (400, 12, 540), 68),
    "above": ((340, 95, 300), (400, 8, 470), 72),
}


def main(out_dir="/tmp/cloudsim"):
    os.makedirs(out_dir, exist_ok=True)
    for name, (cam, tgt, fov) in VIEWS.items():
        new = render(cam, tgt, fov=fov)
        old = render(cam, tgt, fov=fov, iso_top=0.72)
        Image.fromarray(new).save(f"{out_dir}/{name}_iso1.0.png")
        Image.fromarray(old).save(f"{out_dir}/{name}_iso0.72.png")
        print(f"{name}: mean iso1.0 {new.mean():.1f}  iso0.72 {old.mean():.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
