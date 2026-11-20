"""Build the walking, contact-convergence, and demonstration figures."""
from collections import defaultdict
import json
import math
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
FIGURES = HOME / "figures"

plt.rcParams.update(
    {"font.size": 8.6, "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False}
)
SCHUR, PADMM, ALTERNATING = "#216784", "#ac4c35", "#787d82"


def robot_rows(task):
    directory = ROOT / "results" / task / "final_capture" / "heldout"
    records = [
        json.loads(p.read_text()) for p in sorted(directory.glob("walk*.json"))
        if not p.stem.endswith("production")
    ]
    assert len(records) == 44
    return records


def walking():
    fig, axes = plt.subplots(2, 2, figsize=(6.95, 2.84), sharex=True)
    tasks = (("Dr Legs", "scc_dr_legs_walk_rerun"), ("Lanky Dog", "scc_lanky_dog_walk"))
    methods = (("schur", SCHUR, "o"), ("padmm", PADMM, "s"), ("alternating", ALTERNATING, "^"))
    for column, (name, task) in enumerate(tasks):
        records = robot_rows(task)
        axes[0, column].set_title(name)
        for method, color, marker in methods:
            for row, key in enumerate(("residual_max", "foot_tangent_error_mps")):
                ax = axes[row, column]
                reference = records if row == 0 else [r for r in records if r["reference_valid"]]
                by_budget = defaultdict(list)
                for record in reference:
                    for result in record["rows"]:
                        if result["mode"] == method and result["finite"]:
                            by_budget[result["budget"]].append(result[key] * (1000 if row else 1))
                assert by_budget and all(len(v) == len(reference) for v in by_budget.values())
                budgets = sorted(by_budget)
                q = np.array([np.quantile(by_budget[n], [0.1, .5, .9]) for n in budgets])
                ax.fill_between(budgets, q[:, 0], q[:, 2], color=color, alpha=.15, linewidth=0)
                ax.plot(budgets, q[:, 1], marker=marker, markersize=2.3, color=color,
                        linewidth=1.15, linestyle="--" if method == "alternating" else "-")
                ax.set(xscale="log", yscale="log")
                ax.grid(axis="y", linewidth=.35, color="#dedede")
                ax.set_xticks([1, 4, 16, 64, 256, 1024], labels=[1, 4, 16, 64, 256, 1024])
    for a in axes[0]:
        a.axhline(1e-2, color="#7d8183", lw=.7, ls=":")
    for a in axes[1]:
        a.axhline(5, color="#7d8183", lw=.7, ls=":")
        a.set_xlabel("Solver iteration budget")
    axes[0, 0].set_ylabel("Physical residual")
    axes[1, 0].set_ylabel("Foot error (mm/s)")
    fig.legend(handles=[
        Line2D([0], [0], color=color, marker=marker, markersize=3,
               linestyle="--" if method == "alternating" else "-", label=label)
        for (method, color, marker), label in zip(methods, ("Schur SCC", "P-ADMM", "Alternating SCC"))
    ], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.52, 1.03))
    fig.subplots_adjust(top=.84,bottom=.167,left=.115,right=.987,wspace=.23,hspace=.19)
    fig.savefig(FIGURES / "walking_convergence.png", dpi=540)
    plt.close(fig)


def contacts():
    rows = [json.loads(line) for line in
            (HOME/"contact_data/measurements_full.jsonl").read_text().splitlines()]
    assert len(rows) == 150
    cases = ("stack5_unwelded", "stack10_unwelded", "face_v10_mu4")
    modes = ("frozen_16", "lagged_control", "extrapolated")
    colors = ("#787d82", "#ac4c35", "#216784")
    best = {}
    fig, axes = plt.subplots(1, 2, figsize=(3.43, 1.66), sharey=True)
    for axis, name, label in zip(axes, cases, ("5 boxes", "10 boxes")):
        for series, color in zip(modes, colors, strict=True):
            trace = sorted((r for r in rows if r["case"] == name and r["mode"] == series
                            and r["cap"] <= 1024), key=lambda r:r["cap"])
            axis.loglog([r["cap"] for r in trace], [r["worst_residual"] for r in trace],
                        ".-", color=color, lw=1., markersize=3.)
        axis.axhline(1e-5, color="#72777a", ls=":", lw=.75)
        axis.set_title(label, pad=3)
        axis.set_xticks([16, 64, 256, 1024], labels=[16, 64, 256, 1024])
        axis.set_ylim(3e-6, .5)
    axes[0].set_ylabel("Coulomb residual")
    fig.supxlabel("Accepted-update cap", y=.008, fontsize=8.4)
    fig.legend(handles=[Line2D([0],[0], color=color, label=label, marker=".", markersize=2.5)
                        for color,label in zip(colors,("Frozen (16)", "Accepted", "Extrapolated"))],
               loc="upper center", bbox_to_anchor=(.54,1.035), ncol=3, fontsize=7.2,
               frameon=False, columnspacing=.8, handlelength=1.35)
    fig.subplots_adjust(top=.72,bottom=.255,left=.22,right=.966,wspace=.24)
    fig.savefig(FIGURES/"contact_convergence.png",dpi=540)
    plt.close(fig)
    for name in cases:
        best[name] = {}
        for mode in ("frozen", "lagged_control", "extrapolated"):
            options = [r for r in rows if r["case"] == name and r["passed"] and
                       (r["mode"].startswith("frozen_") if mode == "frozen" else r["mode"]==mode)]
            best[name][mode] = min(options,key=lambda r:r["median_ms"])
    (HOME/"contact_data/selected.json").write_text(json.dumps(best,indent=2)+"\n")


def video_frame(path, second, box):
    stream = imageio_ffmpeg.read_frames(str(path), input_params=["-ss", str(second)],
                                      output_params=["-frames:v","1"])
    meta = next(stream)
    frame = Image.frombytes("RGB",meta["size"],next(stream))
    stream.close()
    x0,y0,x1,y1 = box
    left, top = round(x0*frame.width), round(y0*frame.height)
    right, bottom = round(x1*frame.width), round(y1*frame.height)
    image=frame.crop((left, top, right, bottom))
    return image.resize((320, 220), Image.Resampling.LANCZOS)


def teaser():
    media = ROOT / "research/rss_2027/results_compendium/media"
    g1 = ROOT.parent / "isaaclab-dvi/logs/dvi_develop_matrix/2026-10-14/g1_apgd/g1_apgd.mp4"
    frames = (
        (media/"go2_hardware.mp4",8,(.10,.27,.85,.56)),
        (g1,5,(.35,.43,.65,.83)),
        (media/"lanky_dog.mp4",5,(.24,.29,.77,.78)),
        (media/"dr_legs.mp4",5,(.24,.27,.77,.76)),
    )
    canvas = Image.new("RGB",(4*320+3*6,220),"white")
    for i, (path, second, crop) in enumerate(frames):
        canvas.paste(video_frame(path,second,crop),(i*326,0))
    canvas.save(FIGURES/"robots.jpg",quality=94,optimize=True)


if __name__ == "__main__":
    walking()
    contacts()
    teaser()
