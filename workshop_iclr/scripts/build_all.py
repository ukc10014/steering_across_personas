#!/usr/bin/env python3
"""Rebuild every workshop figure from the cached analysis outputs. CPU only.

    source /workspace/bootstrap.sh
    python workshop_iclr/scripts/build_all.py

Each figure script is independent and can be run on its own; this just runs them in
order and reports which inputs were missing. Nothing here launches a forward pass --
every input is a JSON written by a script in scripts/, and those are listed in each
figure's own module docstring.
"""
from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIGURES = [
    ("fig0_schematic", "pipeline schematic"),
    ("fig1_decomposition", "where the intervention goes"),
    ("fig2_ctp", "C x T x P decomposition"),
    ("fig3_dose_and_control", "dose response + matched untrained control"),
    ("fig4_shared_direction", "the shared direction untrained arms miss"),
    ("figA1_dose_calibration", "appendix: the three dose axes"),
    ("figA2_diagnostics", "appendix: per-cell CTP and global maps"),
    ("figA3_layer20", "appendix: layer-20 replication"),
    ("figA4_rotation_control", "appendix: shift rotation vs an untrained arm"),
    ("figA5_signed_validity", "appendix: why there is no signed OCT-style figure"),
]


def main() -> int:
    ok, missing, failed = [], [], []
    for mod, what in FIGURES:
        if not (HERE / f"{mod}.py").exists():
            missing.append((mod, "script not present"))
            continue
        print(f"\n== {mod}  ({what})")
        try:
            runpy.run_path(str(HERE / f"{mod}.py"), run_name="__main__")
            ok.append(mod)
        except (FileNotFoundError, KeyError) as e:
            # KeyError too: a partial or reduced analysis JSON is a missing input, not a
            # broken figure script, and should not read as a failure.
            print(f"  SKIPPED -- missing input: {e}")
            missing.append((mod, str(e)))
        except Exception:
            traceback.print_exc()
            failed.append(mod)

    print(f"\n{'=' * 70}\nbuilt {len(ok)}; skipped {len(missing)}; failed {len(failed)}")
    for m, why in missing:
        print(f"  skipped {m}: {why}")
    for m in failed:
        print(f"  FAILED  {m}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
