#!/usr/bin/env python3
"""Compile Slang sources with the compiler bundled in the `slangpy` wheel.

The environment has no Vulkan SDK, so `slangc` is unavailable. slangpy ships libslang-compiler, which
exposes the ordinary C API; this drives it directly through ctypes to run the same front-end CI's
`compileShaders` task runs, so a shader edit is at least type-checked and code-generated rather than
eyeballed.

Usage: python3 slang_compile.py file.slang [more.slang ...]
"""
import ctypes
import glob
import os
import sys

import slangpy

HERE = os.path.dirname(slangpy.__file__)
LIB = glob.glob(os.path.join(HERE, "libslang-compiler*.so*"))[0]
SLANG_STAGE_FRAGMENT = 5
SLANG_TARGET_SPIRV = 4


class SlangPath(ctypes.Structure):
    _fields_ = [("length", ctypes.c_size_t), ("chars", ctypes.POINTER(ctypes.c_char))]


lib = ctypes.CDLL(LIB)
# This build of libslang exposes the classic C API (spCreateSession/spCreateCompileRequest) rather
# than the newer global-session entry points.
lib.spCreateSession.argtypes = [ctypes.c_void_p]
lib.spCreateSession.restype = ctypes.c_void_p
lib.spCreateCompileRequest.argtypes = [ctypes.c_void_p]
lib.spCreateCompileRequest.restype = ctypes.c_void_p
lib.spDestroySession.argtypes = [ctypes.c_void_p]
lib.spAddCodeGenTarget.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.spAddSearchPath.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.spAddTranslationUnit.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.spAddTranslationUnitSourceFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.spCompile.argtypes = [ctypes.c_void_p]
lib.spCompile.restype = ctypes.c_int
lib.spGetDiagnosticOutput.argtypes = [ctypes.c_void_p]
lib.spGetDiagnosticOutput.restype = ctypes.c_char_p
lib.spDestroyCompileRequest.argtypes = [ctypes.c_void_p]


def compile_file(path):
    session = lib.spCreateSession(None)
    if not session:
        raise SystemExit("failed to create session")
    req = lib.spCreateCompileRequest(session)
    if not req:
        raise SystemExit("failed to create compile request")
    lib.spAddCodeGenTarget(req, SLANG_TARGET_SPIRV)
    lib.spAddSearchPath(req, os.path.dirname(os.path.abspath(path)).encode())
    lib.spAddSearchPath(req, os.path.dirname(os.path.abspath(path)).encode() + b"/..")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for extra in ("shaders/world", "shaders"):
        lib.spAddSearchPath(req, os.path.join(repo, extra).encode())
    tu = lib.spAddTranslationUnit(req, SLANG_STAGE_FRAGMENT)
    lib.spAddTranslationUnitSourceFile(req, os.path.abspath(path).encode())
    result = lib.spCompile(req)
    diag = (lib.spGetDiagnosticOutput(req) or b"").decode(errors="replace")
    lib.spDestroyCompileRequest(req)
    lib.spDestroySession(session)
    return result, diag


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    failures = 0
    for path in paths:
        result, diag = compile_file(path)
        errors = [line for line in diag.splitlines() if "error" in line.lower()]
        if result != 0 or errors:
            failures += 1
            print(f"FAIL {path}")
            print("\n".join(errors[:20]) or diag[:2000])
        else:
            warnings = [line for line in diag.splitlines() if "warning" in line.lower()]
            suffix = f" ({len(warnings)} warnings)" if warnings else ""
            print(f"OK   {path}{suffix}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
