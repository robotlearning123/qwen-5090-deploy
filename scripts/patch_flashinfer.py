#!/usr/bin/env python3
"""Two fixes so FlashInfer JIT works with the venv CUDA 13.3 toolchain.

1. Accept the venv CUDA 13.3 nvcc vs 13.0 headers (CCCL compat check).
2. The venv nvidia/cu13 wheel ships libs under lib/ with a versioned
   libcudart.so.13 only, but flashinfer's build.ninja links with
   -L$cuda_home/lib64 -lcudart, which resolves to nothing. Create the
   lib64 dir symlink and the unversioned libcudart.so symlink (2026-08-17:
   without these the fp4_gemm_cutlass_sm120 link step fails with
   "ld: cannot find -lcudart" and EngineCore dies).
"""

from __future__ import annotations

import os
from pathlib import Path

FLAG = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
NEEDLE = '"-DFLASHINFER_ENABLE_FP4_E2M1",'


def patch_cccl_flag() -> int:
    import flashinfer

    path = Path(flashinfer.__file__).resolve().parent / "compilation_context.py"
    text = path.read_text()
    if FLAG in text:
        print(f"already patched {path}")
        return 0
    if NEEDLE not in text:
        print(f"needle missing in {path}")
        return 1
    path.write_text(text.replace(NEEDLE, NEEDLE + f'\n        "{FLAG}",', 1))
    print(f"patched {path}")
    return 0


def patch_cu13_link_paths() -> int:
    # venv site-packages/nvidia/cu13, independent of where the venv lives.
    import sysconfig

    sp = Path(sysconfig.get_paths()["purelib"])
    cu13 = sp / "nvidia" / "cu13"
    if not (cu13 / "lib" / "libcudart.so.13").exists():
        print(f"no versioned cudart under {cu13}/lib; nothing to do")
        return 0
    made = []
    lib64 = cu13 / "lib64"
    if not lib64.exists():
        os.symlink("lib", lib64)
        made.append(str(lib64))
    devlink = cu13 / "lib" / "libcudart.so"
    if not devlink.exists():
        os.symlink("libcudart.so.13", devlink)
        made.append(str(devlink))
    print("cu13 link paths ok" + (f"; created: {', '.join(made)}" if made else ""))
    return 0


def main() -> int:
    rc = patch_cccl_flag()
    rc |= patch_cu13_link_paths()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
