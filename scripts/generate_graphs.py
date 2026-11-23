from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GRAPH_DIR = RESULTS / "graphs"
SIZES = [512, 1024, 2048, 4096]
ALL_SIZES = SIZES + [8192]
PROCESSES = [1, 4, 9, 16]
ALGORITHMS = [
    ("oned_gemm", "1D GEMM", "oned_gemm_float.csv", "oned_gemm_double.csv"),
    ("summa", "SUMMA", "summa_float.csv", "summa_double.csv"),
    ("cannon", "Cannon", "cannon_float.csv", "cannon_double.csv"),
    ("summa_overlap", "SUMMA overlap", "summa_overlap_float.csv", "summa_overlap_double.csv"),
    ("cannon_overlap", "Cannon overlap", "cannon_overlap_float.csv", "cannon_overlap_double.csv"),
]
COLORS = {
    "1D GEMM": "#3568a8",
    "SUMMA": "#d17b0f",
    "Cannon": "#42855b",
    "SUMMA overlap": "#9b5ba5",
    "Cannon overlap": "#c44e52",
}
MARKERS = {"1D GEMM": "o", "SUMMA": "s", "Cannon": "^", "SUMMA overlap": "D", "Cannon overlap": "P"}


def validate_and_load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw: dict[str, pd.DataFrame] = {}
    expected = {(n, p) for n in ALL_SIZES for p in PROCESSES}
    for key, _, single_file, double_file in ALGORITHMS:
        for precision, filename in (("Single", single_file), ("Double", double_file)):
            path = RESULTS / filename
            df = pd.read_csv(path)
            required = {
                "Matrix_Size", "Processes", "Precision", "Execution_Time",
                "Communication_Time", "Computation_Time", "GFLOPS",
                "C00", "Max_Absolute_Error", "Max_Relative_Error", "Verification",
            }
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
            df["Matrix_Size"] = pd.to_numeric(df["Matrix_Size"], errors="raise").astype(int)
            df["Processes"] = pd.to_numeric(df["Processes"], errors="raise").astype(int)
            df["Max_Relative_Error"] = pd.to_numeric(df["Max_Relative_Error"], errors="raise")
            if len(df) != len(expected) or set(zip(df.Matrix_Size, df.Processes)) != expected:
                raise ValueError(f"{filename} does not contain exactly the expected 20 configurations")
            if not df.Precision.eq(precision).all():
                raise ValueError(f"{filename} contains a precision label inconsistent with its filename")
            if df.duplicated(["Matrix_Size", "Processes", "Precision"]).any():
                raise ValueError(f"{filename} contains duplicate configuration keys")
            if not (df["Verification"].str.upper().isin(["PASS", "PASSED"]).all() and (df.Max_Relative_Error < 1e-4).all()):
                raise ValueError(f"{filename} contains a failed or out-of-tolerance result")
            df["Algorithm"] = key
            df["Algorithm_Label"] = next(label for k, label, *_ in ALGORITHMS if k == key)
            raw[f"{key}_{precision}"] = df

    summary_path = RESULTS / "benchmark_summary.csv"
    summary = pd.read_csv(summary_path)
    needed = {
        "Program", "Matrix_Size", "Processes", "Precision", "Successful_Runs",
        "Median_Execution_Time", "Median_Communication_Time",
        "Median_Computation_Time", "Median_GFLOPS",
    }
    missing = needed.difference(summary.columns)
    if missing:
        raise ValueError(f"benchmark_summary.csv is missing columns: {sorted(missing)}")
    summary["Matrix_Size"] = pd.to_numeric(summary.Matrix_Size, errors="raise").astype(int)
    summary["Processes"] = pd.to_numeric(summary.Processes, errors="raise").astype(int)
    summary["Successful_Runs"] = pd.to_numeric(summary.Successful_Runs, errors="raise").astype(int)
    for col in ["Median_Execution_Time", "Median_Communication_Time", "Median_Computation_Time", "Median_GFLOPS"]:
        summary[col] = pd.to_numeric(summary[col], errors="raise")
    if len(summary) != 200 or summary.duplicated(["Program", "Matrix_Size", "Processes", "Precision"]).any():
        raise ValueError("benchmark_summary.csv must have 200 unique configuration rows")
    if not (summary.Successful_Runs.eq(1).all()):
        raise ValueError("Expected exactly one successful run per configuration")
    if not (summary.Matrix_Size.isin(ALL_SIZES).all() and summary.Processes.isin(PROCESSES).all()):
        raise ValueError("Summary contains an unexpected size or process count")

    program_to_key = {}
    for key, _label, *_ in ALGORITHMS:
        program_to_key[key] = key
        program_to_key[f"{key}_double"] = key
    key_to_label = {key: label for key, label, *_ in ALGORITHMS}
    summary["Algorithm"] = summary.Program.map(program_to_key)
    summary["Algorithm_Label"] = summary.Algorithm.map(key_to_label)
    if summary.Algorithm_Label.isna().any():
        raise ValueError("Summary contains an unknown program")
    expected_summary = {
        (key if precision == "Single" else f"{key}_double", n, p, precision)
        for key, *_ in ALGORITHMS
        for precision in ["Single", "Double"]
        for n in ALL_SIZES
        for p in PROCESSES
    }
    actual_summary = set(zip(summary.Program, summary.Matrix_Size, summary.Processes, summary.Precision))
    if actual_summary != expected_summary:
        raise ValueError("Summary does not cover all algorithm/precision/size/process combinations")
    # All sizes are plotted. 8192 came from a Mac, while 512–4096 came from
    # Windows; charts explicitly mark this platform boundary.
    return summary, raw


def save_figure(fig: plt.Figure, stem: str) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPH_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#d9dfe7", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def prep(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["Execution_Time"] = df.Median_Execution_Time
    df["Communication_Time"] = df.Median_Communication_Time
    df["Computation_Time"] = df.Median_Computation_Time
    df["GFLOPS"] = df.Median_GFLOPS
    df["Speedup"] = np.nan
    df["Efficiency_Percent"] = np.nan
    for (_, _, precision), inds in df.groupby(["Algorithm", "Matrix_Size", "Precision"]).groups.items():
        group = df.loc[inds]
        baseline_rows = group[group.Processes == 1]
        if len(baseline_rows) != 1:
            raise ValueError("Every algorithm/size/precision group must have one P=1 baseline")
        baseline = float(baseline_rows.iloc[0].Execution_Time)
        df.loc[inds, "Speedup"] = baseline / group.Execution_Time.to_numpy()
        df.loc[inds, "Efficiency_Percent"] = df.loc[inds, "Speedup"] / group.Processes.to_numpy() * 100.0
    # Define the one-process point exactly, as requested.
    df.loc[df.Processes == 1, "Speedup"] = 1.0
    df.loc[df.Processes == 1, "Efficiency_Percent"] = 100.0
    return df


def plot_execution_time(df: pd.DataFrame) -> None:
    for precision in ["Single", "Double"]:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharex=True)
        for ax, p in zip(axes.flat, PROCESSES):
            panel = df[(df.Precision == precision) & (df.Processes == p)]
            for _, label, *_ in ALGORITHMS:
                series = panel[panel.Algorithm_Label == label].sort_values("Matrix_Size")
                windows = series[series.Matrix_Size.isin(SIZES)]
                mac = series[series.Matrix_Size == 8192]
                ax.plot(windows.Matrix_Size, windows.Execution_Time, color=COLORS[label], marker=MARKERS[label], linewidth=1.8, markersize=5, label=label)
                ax.scatter(mac.Matrix_Size, mac.Execution_Time, color=COLORS[label], marker="X", s=55, zorder=4, label="8192, Mac" if label == "1D GEMM" else None)
            ax.set_title(f"{p} process{'es' if p != 1 else ''}")
            ax.set_xlabel("Matrix size, n")
            ax.set_ylabel("Execution time (s)")
            ax.set_xticks(ALL_SIZES, [f"{n:,}" for n in ALL_SIZES])
            ax.set_yscale("log")
            style_axes(ax)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
        fig.suptitle(f"Execution time vs matrix size — {precision} precision\nSingle runs; 8192 points are Mac, smaller sizes are Windows (log time axis)", y=1.04, fontsize=14, fontweight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(fig, f"execution_time_{precision.lower()}")


def plot_scaling(df: pd.DataFrame, metric: str, ylabel: str, stem: str, title: str, efficiency: bool = False) -> None:
    for precision in ["Single", "Double"]:
        fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharex=True)
        for ax, n in zip(axes.flat, ALL_SIZES):
            panel = df[(df.Precision == precision) & (df.Matrix_Size == n)]
            for _, label, *_ in ALGORITHMS:
                series = panel[panel.Algorithm_Label == label].sort_values("Processes")
                ax.plot(series.Processes, series[metric], color=COLORS[label], marker=MARKERS[label], linewidth=1.8, markersize=5, label=label)
            ax.set_title(f"n = {n:,}")
            ax.set_xlabel("MPI processes")
            ax.set_ylabel(ylabel)
            ax.set_xticks(PROCESSES)
            if efficiency:
                ax.axhline(100, color="#626a73", linestyle="--", linewidth=1, label="100% efficiency")
            style_axes(ax)
        axes.flat[-1].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5 if not efficiency else 6, frameon=False, bbox_to_anchor=(0.5, 0.955))
        fig.suptitle(f"{title} — {precision} precision\nDerived from single-run measurements", y=1.04, fontsize=14, fontweight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(fig, f"{stem}_{precision.lower()}")


def plot_gflops(df: pd.DataFrame) -> None:
    for precision in ["Single", "Double"]:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharex=True)
        for ax, p in zip(axes.flat, PROCESSES):
            panel = df[(df.Precision == precision) & (df.Processes == p)]
            for _, label, *_ in ALGORITHMS:
                series = panel[panel.Algorithm_Label == label].sort_values("Matrix_Size")
                windows = series[series.Matrix_Size.isin(SIZES)]
                mac = series[series.Matrix_Size == 8192]
                ax.plot(windows.Matrix_Size, windows.GFLOPS, color=COLORS[label], marker=MARKERS[label], linewidth=1.8, markersize=5, label=label)
                ax.scatter(mac.Matrix_Size, mac.GFLOPS, color=COLORS[label], marker="X", s=55, zorder=4, label="8192, Mac" if label == "1D GEMM" else None)
            ax.set_title(f"{p} process{'es' if p != 1 else ''}")
            ax.set_xlabel("Matrix size, n")
            ax.set_ylabel("Performance (GFLOP/s)")
            ax.set_xticks(ALL_SIZES, [f"{n:,}" for n in ALL_SIZES])
            style_axes(ax)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
        fig.suptitle(f"GFLOP/s vs matrix size — {precision} precision\nSingle runs; 8192 points are Mac, smaller sizes are Windows", y=1.04, fontsize=14, fontweight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(fig, f"gflops_{precision.lower()}")


def plot_compute_comm(df: pd.DataFrame) -> None:
    for precision in ["Single", "Double"]:
        for p in PROCESSES:
            fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharey=False)
            for ax, n in zip(axes.flat, ALL_SIZES):
                panel = df[(df.Precision == precision) & (df.Matrix_Size == n) & (df.Processes == p)].copy()
                panel["Algorithm_Order"] = panel.Algorithm_Label.map({label: i for i, (_, label, *_) in enumerate(ALGORITHMS)})
                panel = panel.sort_values("Algorithm_Order")
                x = np.arange(len(panel))
                colors = [COLORS[name] for name in panel.Algorithm_Label]
                ax.bar(x, panel.Computation_Time, color=colors, edgecolor="white", linewidth=0.5, label="Computation")
                ax.bar(x, panel.Communication_Time, bottom=panel.Computation_Time, color="#b9c1cb", edgecolor="white", linewidth=0.5, hatch="///", label="Communication")
                ax.set_xticks(x, panel.Algorithm_Label, rotation=18, ha="right", fontsize=8)
                ax.set_title(f"n = {n:,}")
                ax.set_ylabel("Time component (s)")
                style_axes(ax, "y")
            axes.flat[-1].axis("off")
            from matplotlib.patches import Patch
            legend = [Patch(facecolor="#777777", label="Computation"), Patch(facecolor="#b9c1cb", hatch="///", label="Communication")]
            fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
            fig.suptitle(f"Computation and communication — {precision} precision, P={p}\nSingle runs; 8192 is Mac, smaller sizes are Windows", y=1.04, fontsize=14, fontweight="semibold")
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            save_figure(fig, f"computation_communication_{precision.lower()}_p{p}")


def plot_overlap(df: pd.DataFrame, base: str, overlap: str, label: str, precision: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharex=True)
    for ax, n in zip(axes.flat, ALL_SIZES):
        panel = df[(df.Precision == precision) & (df.Matrix_Size == n) & (df.Algorithm.isin([base, overlap]))]
        for program, name, color, marker, linestyle in [
            (base, f"{label}", "#3568a8", "o", "-"),
            (overlap, f"{label} overlap", "#c44e52", "s", "--"),
        ]:
            series = panel[panel.Algorithm == program].sort_values("Processes")
            ax.plot(series.Processes, series.Execution_Time, label=name, color=color, marker=marker, linestyle=linestyle, linewidth=2, markersize=5)
        ax.set_title(f"n = {n:,}")
        ax.set_xlabel("MPI processes")
        ax.set_ylabel("Execution time (s)")
        ax.set_xticks(PROCESSES)
        style_axes(ax)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{label} overlap comparison — {precision} precision\nSingle runs; 8192 Mac results, smaller sizes Windows", y=1.04, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, f"{label.lower()}_overlap_comparison_{precision.lower()}")


def build_analysis(df: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Distributed GEMM performance analysis",
        "",
        "Charts include all 200 single-run measurements: 512–4096 were recorded on Windows and 8192 on a separate Mac. The 8192 measurements are visibly identified as Mac data; do not compare their absolute time or GFLOP/s directly with Windows points. Speedup and efficiency use each algorithm/precision/size's own P=1 result, so the 8192 scaling curves are within the Mac campaign. Each configuration ran once; summary medians are single measurements, not repeated-run averages. All raw CSV rows passed verification with maximum relative error below `1e-4`.",
        "",
        "## Best-performing implementation by size",
        "",
        "Winner means the shortest execution time among algorithms at the same matrix size, precision, and process count. This keeps the comparison fair. The winners can differ with precision or process count.",
        "",
        "| Matrix size | Precision | Processes | Fastest algorithm | Execution time (s) | GFLOP/s |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for n in ALL_SIZES:
        for precision in ["Single", "Double"]:
            for p in PROCESSES:
                group = df[(df.Matrix_Size == n) & (df.Precision == precision) & (df.Processes == p)]
                row = group.loc[group.Execution_Time.idxmin()]
                lines.append(f"| {n:,} | {precision} | {p} | {row.Algorithm_Label} | {row.Execution_Time:.6g} | {row.GFLOPS:.4g} |")

    lines += ["", "## Process count, speedup, and efficiency", ""]
    lines.append("Speedup is computed as `T1/Tp` within each algorithm, precision, and matrix size; parallel efficiency is `(speedup/processes) × 100`. The plotted P=1 speedup and efficiency are set to exactly 1 and 100%, respectively.")
    lines.append("")
    lines.append("| Precision | Size | Algorithm | P4 speedup | P9 speedup | P16 speedup | P16 efficiency |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for precision in ["Single", "Double"]:
        for n in ALL_SIZES:
            for _, label, *_ in ALGORITHMS:
                group = df[(df.Precision == precision) & (df.Matrix_Size == n) & (df.Algorithm_Label == label)].set_index("Processes")
                lines.append(f"| {precision} | {n:,} | {label} | {group.loc[4, 'Speedup']:.3f}× | {group.loc[9, 'Speedup']:.3f}× | {group.loc[16, 'Speedup']:.3f}× | {group.loc[16, 'Efficiency_Percent']:.1f}% |")

    lines += ["", "## Communication overhead", ""]
    comm_share = df["Communication_Time"] / (df["Communication_Time"] + df["Computation_Time"])
    lines.append(f"Across the 200 configurations from two machines, communication time is greater than computation time in {int((df.Communication_Time > df.Computation_Time).sum())} cases. The median communication share of the measured communication-plus-computation components is {comm_share.median() * 100:.1f}% (range {comm_share.min() * 100:.1f}%–{comm_share.max() * 100:.1f}%). This component share is descriptive; for overlap implementations communication and computation may occur concurrently, so their sum is not necessarily execution time.")
    lines.append("")
    for precision in ["Single", "Double"]:
        p16 = df[(df.Precision == precision) & (df.Processes == 16)]
        median_ratio = (p16.Communication_Time / p16.Execution_Time).median() * 100
        lines.append(f"- At 16 processes ({precision.lower()}), median reported communication time is {median_ratio:.1f}% of execution time.")

    lines += ["", "## Communication-computation overlap", ""]
    for base, overlap, label in [("summa", "summa_overlap", "SUMMA"), ("cannon", "cannon_overlap", "Cannon")]:
        for precision in ["Single", "Double"]:
            a = df[(df.Algorithm == base) & (df.Precision == precision)].set_index(["Matrix_Size", "Processes"])
            b = df[(df.Algorithm == overlap) & (df.Precision == precision)].set_index(["Matrix_Size", "Processes"])
            pct = (a.Execution_Time - b.Execution_Time) / a.Execution_Time * 100
            wins = int((pct > 0).sum())
            lines.append(f"- {label} ({precision.lower()}): overlap was faster in {wins}/20 matched size/process comparisons; median execution-time change was {pct.median():+.2f}% (positive means faster with overlap).")
    lines.append("These are comparisons of measured wall times only. Single-run results do not establish that observed differences are repeatable or statistically significant.")

    lines += ["", "## SUMMA versus Cannon", ""]
    for precision in ["Single", "Double"]:
        summa = df[df.Algorithm == "summa"].set_index(["Precision", "Matrix_Size", "Processes"])
        cannon = df[df.Algorithm == "cannon"].set_index(["Precision", "Matrix_Size", "Processes"])
        comparisons = pd.DataFrame({"summa": summa.Execution_Time, "cannon": cannon.Execution_Time}).loc[precision]
        summa_wins = int((comparisons.summa < comparisons.cannon).sum())
        cannon_wins = int((comparisons.cannon < comparisons.summa).sum())
        lines.append(f"- Plain SUMMA was faster in {summa_wins}/20 matched cases and Cannon in {cannon_wins}/20 ({precision.lower()}). No single algorithm dominated every tested size/process combination.")
    lines.append("Use the winner table above to see the specific size and process-count conditions; rankings are workload- and precision-dependent.")

    lines += ["", "## Single versus double precision", ""]
    for algorithm, label, *_ in ALGORITHMS:
        single = df[(df.Algorithm == algorithm) & (df.Precision == "Single")].set_index(["Matrix_Size", "Processes"])
        double = df[(df.Algorithm == algorithm) & (df.Precision == "Double")].set_index(["Matrix_Size", "Processes"])
        ratios = double.Execution_Time / single.Execution_Time
        lines.append(f"- {label}: the double/single execution-time ratio has a median of {ratios.median():.2f}× (range {ratios.min():.2f}×–{ratios.max():.2f}×) across matched configurations.")
    lines += [
        "",
        "## Measurement limitations",
        "",
        "- Every configuration has one measurement. The summary median therefore has no repeated samples behind it; system load, process placement, and timing noise can affect individual points.",
        "- The 512–4096 campaign used one Windows machine/runtime; the 8192 campaign used a separate Mac and Open MPI. Their performance values are kept separate. Neither is a multi-node study.",
        "- GFLOP/s uses each recorded execution time and the same `2n³` operation count. It is useful for within-campaign comparisons, but it does not separate the benchmark from system-level effects.",
        "- The stacked communication/computation figures show the reported timing components. In overlap variants, these components can overlap, so they should not be interpreted as additive portions of total execution time.",
        "",
        "## Inputs and generated figures",
        "",
        "The ten `results/*_float.csv` and `results/*_double.csv` files were read without modification and validated for all 200 rows. Every graph includes the 8192 observations. Absolute performance graphs identify the machine boundary; speedups use matched P=1 baselines from each size's own campaign.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    summary, raw = validate_and_load()
    df = prep(summary)

    plot_execution_time(df)
    plot_scaling(df, "Speedup", "Speedup (×)", "speedup", "Speedup vs process count")
    plot_scaling(df, "Efficiency_Percent", "Parallel efficiency (%)", "parallel_efficiency", "Parallel efficiency vs process count", efficiency=True)
    plot_gflops(df)
    plot_compute_comm(df)
    for precision in ["Single", "Double"]:
        plot_overlap(df, "summa", "summa_overlap", "SUMMA", precision)
        plot_overlap(df, "cannon", "cannon_overlap", "Cannon", precision)

    analysis = build_analysis(df, raw)
    documentation_dir = RESULTS / "documentation"
    documentation_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = documentation_dir / "graph_analysis.md"
    analysis_path.write_text(analysis, encoding="utf-8")
    print(f"Validated {sum(len(frame) for frame in raw.values())} raw benchmark rows; plotted {len(summary)} summary rows across both campaigns.")
    print(f"Generated {len(list(GRAPH_DIR.glob('*.png')))} PNG graphs at 300 DPI in {GRAPH_DIR}")
    print(f"Analysis written to {analysis_path}")


if __name__ == "__main__":
    main()
