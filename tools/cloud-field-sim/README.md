# Cloud field simulator

Offline tooling used while developing the volumetric cloud model in `shaders/world/clouds.slang`. It is
**not** part of the build and nothing in `src/` or `shaders/` depends on it.

Three tools:

| file | what it does |
|---|---|
| `cloud_field.py` | A numpy transcription of the volumetric density function: same uint32 lattice hash, same periods, same stage order. All constants are **parsed out of `clouds.slang` at import**, so a measurement can never describe an older tuning — if a constant moves, the numbers move with it. |
| `validate_thickness.py` | Measures the properties the rework is judged on: per-formation configured thickness (and that `min == max` is exactly constant), per-column vertical extent above a formation's own base, whether any column reaches its ceiling, whether formations stay inside the slab the renderer marches, density statistics, and animation coherence over time. Asserts, so it fails rather than reporting a bad number. |
| `preview.py` | Ray-marched preview images of the density field (Beer-Lambert, dual-lobe HG, powder, multiple scattering, stand-in sky). For comparing tunings; not a picture of the game. |

Also `slang_compile.py`, a ctypes driver for the Slang compiler bundled in the `slangpy` wheel — useful
where the Vulkan SDK (and therefore `slangc`) is unavailable. Compiling a shader is not the same as
running it, but it is a real front-end + SPIR-V codegen pass.

```bash
pip install numpy pillow slangpy          # slangpy only for slang_compile.py
python3 tools/cloud-field-sim/validate_thickness.py     # ~2 min
python3 tools/cloud-field-sim/preview.py /tmp/clouds     # ~40 s
python3 tools/cloud-field-sim/slang_compile.py shaders/world/*.slang
```

## Why this exists

The CI job that builds this repository runs without a GPU and cannot observe a rendered frame, so a
shader change can be compiled but not seen. These tools turn the parts that are *checkable* — does a
formation's extent match the sliders, does the volume close inside its own bounds, is the field really
three-dimensional — into numbers. They are a stand-in for observation, not a substitute: the
transcription agrees with the shader because it was written against it line by line, and a change to the
shader's structure has to be mirrored here by hand.
