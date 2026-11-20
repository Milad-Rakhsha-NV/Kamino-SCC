"""Measure De Saxcé APGD schedules on physically joint-free box problems."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
STACK = ROOT / "results/scc_stack_accuracy_time"
APGD = ROOT / "results/apgd_coulomb_convergence"
sys.path[:0] = [str(APGD / "matched_time"), str(STACK / "scripts")]
import benchmark_frozen_stack as base
import measure
from study import Case, face, GDT, TOL

MODES = ("frozen_1", "frozen_4", "frozen_16", "frozen_64", "frozen_256", "lagged_control", "extrapolated")
CAPS = (16, 32, 64, 128, 256, 512, 1024, 4096)


def check_gpu():
    entries = measure.gpu_state()
    assert all(
        line.split(",")[0].strip() == str(os.getpid())
        or "/usr/lib/x86_64-linux-gnu/dcv/dcvagent" in line
        for line in entries.splitlines() if line.strip()
    ), entries
    return entries


def ordinary_stack(api, boxes):
    snapshot = STACK / "raw/initial" / f"boxes{boxes}_ratio0_worlds1.npz"
    b = base.build_problem(api, boxes, 0, 1, 0.75, "cuda:0", restore=snapshot)
    assert b["nb"] == 0 and b["nc"] == 4 * boxes, (b["nb"], b["nc"])
    with np.load(snapshot) as old:
        for key in ("j_all", "free", "inv_mass"):
            np.testing.assert_array_equal(old[key], b[key])
    reference = base.accuracy(b, b["reference"])
    assert reference["error"] <= 1e-7, reference
    return Case(f"stack{boxes}_unwelded", b["matrix"], b["rhs"][0], b["mu"],
                b["reference"][0], np.zeros(b["n"]), GDT,
                hashlib.sha256(snapshot.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = base.load_api(args.upstream.resolve())
    wp = api["wp"]
    wp.init()
    from newton._src.solvers.kamino._src.solvers.scc import apgd

    gpu_start = check_gpu()
    cases = [ordinary_stack(api, 5), ordinary_stack(api, 10), face(10.0, 4.0)]
    events = [wp.Event(device="cuda:0", enable_timing=True) for _ in range(2)]
    rng = random.Random(27019)
    rows = []
    with tempfile.TemporaryDirectory() as scratch, args.output.open("x", buffering=1) as output:
        modules = measure.prototypes(apgd, Path(scratch), wp)
        wp.load_module(module=apgd.kernels, device="cuda:0")
        for case in cases:
            settings = [(mode, cap) for mode in MODES for cap in CAPS
                        if not mode.startswith("frozen_") or cap >= int(mode.split("_")[1])]
            rng.shuffle(settings)
            for mode, cap in settings:
                run = measure.capture([case], mode, modules, wp, cap=cap)
                graph = run[0]
                for _ in range(2):
                    wp.capture_launch(graph)
                wp.synchronize()
                check_gpu()
                scores, elapsed = [], []
                for _ in range(7):
                    wp.record_event(events[0])
                    wp.capture_launch(graph)
                    wp.record_event(events[1])
                    elapsed.append(wp.get_event_elapsed_time(*events, synchronize=True))
                    scores.append(measure.score(run, [case])[0])
                check_gpu()
                assert all(np.isfinite(s["r"]) for s in scores)
                row = dict(case=case.name, mode=mode, cap=cap, contacts=len(case.b)//3,
                           correction_period=int(mode.split("_")[1]) if mode.startswith("frozen_") else 1,
                           median_ms=float(np.median(elapsed)), worst_residual=max(s["r"] for s in scores),
                           worst_velocity_error_mps=max(s["velocity_error"] for s in scores),
                           max_accepted_steps=max(s["steps"] for s in scores),
                           passed=all(s["r"] <= TOL and not s["failed"] for s in scores),
                           time_ms=elapsed, scores=scores)
                output.write(json.dumps(row)+"\n")
                rows.append(row)
            print(case.name, len(rows), flush=True)
    manifest = dict(
        newton_revision=base.REVISION, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu_start=gpu_start, gpu_end=check_gpu(), target=TOL, caps=CAPS, seed=27019,
        cases={c.name:{'contacts':len(c.b)//3,'mu':c.mu,'source':c.source} for c in cases},
        gpu=subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"], text=True
        ).strip(),
    )
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
