import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    }
)

KERNEL_COLORS = {
    "reg": "#4e79a7",
    "warp": "#f28e2b",
    "warp_gs": "#e15759",
    "shared": "#59a14f",
    "shared_rt": "#7C856B",
    "cublas": "#b07aa1",
}
KERNEL_LABELS = {
    "reg": "Kernel 1 – registers",
    "warp": "Kernel 2 – warp",
    "warp_gs": "Kernel 3 – warp GS",
    "shared": "Kernel 4 – shared mem",
    "shared_rt": "Kernel 5 – 2D reg tiling",
    "cublas": "cuBLAS reference",
}


def kname(k):
    return KERNEL_LABELS.get(k, k)


def kcolor(k):
    return KERNEL_COLORS.get(k, "#aaaaaa")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def config_label(row):
    return f"{int(row['M'])}x{int(row['N'])}"


def save_plot(fig, out_path):
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def load_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    df["config"] = df.apply(config_label, axis=1)
    df["kernel"] = df["kernel"].astype(str)
    df["aspect_ratio"] = df["M"] / df["N"]
    df["shape"] = df.apply(lambda r: f"{int(r['M'])}x{int(r['N'])}", axis=1)

    df["efficiency_pct"] = 100.0 * df["gflops"] / df["cublas_gflops"]
    return df


def plot_vs_k(df, out_dir):

    configs = df[["M", "N"]].drop_duplicates().sort_values(["M", "N"]).values

    for M, N in configs:
        sub = (
            df[(df["M"] == M) & (df["N"] == N)].sort_values("k").reset_index(drop=True)
        )

        if sub.empty:
            continue

        out = os.path.join(out_dir, f"gflops_vs_k_{M}x{N}.png")
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2]},
        )

        fig.suptitle(
            f"Performance vs output columns k\n(M={M}, N={N})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        ax1.fill_between(
            sub["k"],
            sub["cublas_gflops"],
            alpha=0.12,
            color=kcolor("cublas"),
            label="_nolegend_",
        )

        ax1.plot(
            sub["k"],
            sub["cublas_gflops"],
            color=kcolor("cublas"),
            linewidth=2,
            linestyle="--",
            label=kname("cublas"),
            zorder=3,
        )

        kernels_seen = set()
        for i, row in sub.iterrows():
            kn = row["kernel"]
            color = kcolor(kn)
            label = kname(kn) if kn not in kernels_seen else "_nolegend_"
            kernels_seen.add(kn)
            if i < len(sub) - 1 and sub.iloc[i + 1]["kernel"] == kn:
                ax1.plot(
                    [row["k"], sub.iloc[i + 1]["k"]],
                    [row["gflops"], sub.iloc[i + 1]["gflops"]],
                    color=color,
                    linewidth=2.5,
                    zorder=4,
                )

            ax1.scatter(
                row["k"],
                row["gflops"],
                color=color,
                s=60,
                zorder=5,
                label=label,
            )

        regimes = [
            (1, 16, "reg", "Registers"),
            (17, 32, "warp_gs", "Warp/Warp-GS"),
            (33, 64, "shared", "Shared"),
            (65, 128, "shared_rt", "Shared+RT"),
        ]
        ymax = sub["cublas_gflops"].max() * 1.08
        for lo, hi, kn, label in regimes:
            ax1.axvspan(
                lo,
                hi,
                alpha=0.06,
                color=kcolor(kn),
            )

            ax1.text(
                (lo + hi) / 2,
                ymax * 0.97,
                label,
                ha="center",
                va="top",
                fontsize=8,
                color=kcolor(kn),
                fontweight="bold",
            )

        ax1.set_ylabel("GFLOPs/s")
        ax1.set_ylim(0, ymax)

        ax1.set_xscale("log", base=2)
        ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.grid(alpha=0.25, linestyle="--")

        ax1.legend(
            loc="upper left",
            fontsize=9,
            framealpha=0.9,
        )

        efficiency = 100.0 * sub["gflops"] / sub["cublas_gflops"]
        bar_colors = [kcolor(k) for k in sub["kernel"]]

        ax2.bar(
            sub["k"],
            efficiency,
            color=bar_colors,
            width=[k * 0.6 for k in sub["k"]],
            align="center",
        )

        ax2.axhline(
            100,
            color=kcolor("cublas"),
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

        ax2.set_ylabel("Efficiency %\nvs cuBLAS")
        ax2.set_xlabel("k  (log₂ scale)")
        ax2.set_ylim(0, max(110, efficiency.max() * 1.15))
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.grid(alpha=0.25, linestyle="--")

        plt.tight_layout()
        plt.savefig(
            out,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        print(f"Saved {out}")


def plot_kernel_phase_diagrams(df, out_dir):

    shapes = df["shape"].unique()
    ncols = 2
    nrows = int(np.ceil(len(shapes) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)
    for ax, shape in zip(axes, shapes):
        sub = df[df["shape"] == shape].sort_values("k")

        # cuBLAS
        ax.fill_between(
            sub["k"],
            sub["cublas_gflops"],
            alpha=0.12,
            color=kcolor("cublas"),
        )

        ax.plot(
            sub["k"],
            sub["cublas_gflops"],
            linestyle="--",
            linewidth=2,
            color=kcolor("cublas"),
            label="cuBLAS",
        )

        kernels_seen = set()
        for i, row in sub.iterrows():
            kn = row["kernel"]
            label = kname(kn) if kn not in kernels_seen else "_nolegend_"
            kernels_seen.add(kn)

            ax.scatter(
                row["k"],
                row["gflops"],
                s=70,
                color=kcolor(kn),
                label=label,
                zorder=5,
            )

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(shape)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8)

    fig.suptitle(
        "Kernel Phase Diagrams",
        fontsize=14,
        fontweight="bold",
    )

    fig.supxlabel("k")
    fig.supylabel("GFLOPs/s")

    plt.tight_layout()
    out = os.path.join(
        out_dir,
        "kernel_phase_diagrams.png",
    )

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


def plot_scaling_curves(df, out_dir):

    df = df.copy()
    df["problem_size"] = df["M"] * df["N"] * df["k"]
    fig, ax = plt.subplots(figsize=(10, 6))
    cublas = (
        df.groupby("problem_size")["cublas_gflops"]
        .max()
        .reset_index()
        .sort_values("problem_size")
    )

    ax.plot(
        cublas["problem_size"],
        cublas["cublas_gflops"],
        linestyle="--",
        linewidth=2,
        color=kcolor("cublas"),
        label="cuBLAS",
    )

    for kernel, grp in df.groupby("kernel"):
        grp = grp.sort_values("problem_size")

        ax.plot(
            grp["problem_size"],
            grp["gflops"],
            marker="o",
            linewidth=2,
            color=kcolor(kernel),
            label=kname(kernel),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Problem size = M × N × k")
    ax.set_ylabel("GFLOPs/s")
    ax.set_title(
        "Performance Scaling Curves",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(
        out_dir,
        "scaling_curves.png",
    )

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


def plot_roofline(df, out_dir):

    df = df.copy()
    flops = 2.0 * df["M"] * df["N"] * df["k"]
    bytes_moved = 4.0 * (df["M"] * df["N"] + df["N"] * df["k"] + df["M"] * df["k"])
    df["AI"] = flops / bytes_moved

    fig, ax = plt.subplots(figsize=(10, 6))
    for kernel, grp in df.groupby("kernel"):
        ax.scatter(
            grp["AI"],
            grp["gflops"],
            s=80,
            color=kcolor(kernel),
            label=kname(kernel),
            alpha=0.8,
        )

    ax.scatter(
        df["AI"],
        df["cublas_gflops"],
        marker="x",
        s=90,
        linewidths=2,
        color=kcolor("cublas"),
        label="cuBLAS",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("GFLOPs/s")
    ax.set_title(
        "Roofline-style Performance Plot",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()

    out = os.path.join(
        out_dir,
        "roofline_plot.png",
    )

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


def plot_speedup(df, out_dir):
    configs = df[["M", "N"]].drop_duplicates().sort_values(["M", "N"]).values

    for M, N in configs:
        sub = (
            df[(df["M"] == M) & (df["N"] == N)].sort_values("k").reset_index(drop=True)
        )
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"Speedup vs k  (M={M}, N={N})", fontsize=13, fontweight="bold")

        # Linea grigia di sfondo che connette tutti i punti
        ax.plot(
            sub["k"],
            sub["speedup"],
            color="#cccccc",
            linewidth=1.5,
            linestyle="-",
            zorder=1,
            label="_nolegend_",
        )

        # Linee colorate per segmenti dello stesso kernel
        kernels_seen = set()
        for i, row in sub.iterrows():
            kn = row["kernel"]
            label = kname(kn) if kn not in kernels_seen else "_nolegend_"
            kernels_seen.add(kn)

            if i < len(sub) - 1 and sub.iloc[i + 1]["kernel"] == kn:
                ax.plot(
                    [row["k"], sub.iloc[i + 1]["k"]],
                    [row["speedup"], sub.iloc[i + 1]["speedup"]],
                    color=kcolor(kn),
                    linewidth=2.5,
                    zorder=3,
                )

            ax.scatter(
                row["k"], row["speedup"], color=kcolor(kn), s=70, zorder=5, label=label
            )

        ax.set_xlabel("k  (log₂ scale)")
        ax.set_ylabel("Speedup")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.25, linestyle="--")

        plt.tight_layout()
        out = os.path.join(out_dir, f"speedup_vs_k_{M}x{N}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def plot_time_breakdown(df, out_dir):

    import matplotlib.pyplot as plt
    import numpy as np

    eq = df[df["M"] == df["N"]].copy()

    eq = eq.sort_values(["k", "M"])

    labels = [f"k={r.k}\n{r.M}" for _, r in eq.iterrows()]

    x = np.arange(len(eq))

    kernel_ms = eq["kernel_time"] * 1000
    mpi_ms = eq["mpi_time"] * 1000
    h2d_ms = eq["h2d_time"] * 1000
    d2h_ms = eq["d2h_time"] * 1000

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.bar(
        x,
        kernel_ms,
        label="Kernel GPU",
        alpha=0.9,
    )

    ax1.bar(
        x,
        mpi_ms,
        bottom=kernel_ms,
        label="MPI AllReduce",
        alpha=0.9,
    )

    ax1.bar(
        x,
        h2d_ms,
        bottom=kernel_ms + mpi_ms,
        label="H2D",
        alpha=0.9,
    )

    ax1.bar(
        x,
        d2h_ms,
        bottom=kernel_ms + mpi_ms + h2d_ms,
        label="D2H",
        alpha=0.9,
    )

    ax1.set_ylabel("Time (ms)")

    ax1.set_title(
        "Runtime Decomposition (Full)",
        fontsize=18,
        fontweight="bold",
    )

    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.bar(
        x,
        kernel_ms,
        label="Kernel GPU",
        alpha=0.9,
    )

    ax2.bar(
        x,
        mpi_ms,
        bottom=kernel_ms,
        label="MPI AllReduce",
        alpha=0.9,
    )

    ax2.bar(
        x,
        d2h_ms,
        bottom=kernel_ms + mpi_ms,
        label="D2H",
        alpha=0.9,
    )

    ax2.set_ylabel("Time (ms)")

    ax2.set_title(
        "Compute-side Breakdown (Zoomed)",
        fontsize=15,
        fontweight="bold",
    )

    ax2.grid(alpha=0.25)

    ax2.set_ylim(
        0,
        (kernel_ms + mpi_ms + d2h_ms).max() * 1.2,
    )
    ax2.set_xticks(x)

    ax2.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax2.set_xlabel("k and matrix size")
    plt.tight_layout()
    out = os.path.join(
        out_dir,
        "time_breakdown_split.png",
    )

    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark plotting tool")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--out", default="plots", help="Output directory")
    args = parser.parse_args()

    ensure_dir(args.out)

    df = load_data(args.csv)

    plot_vs_k(df, args.out)
    plot_kernel_phase_diagrams(df, args.out)
    plot_scaling_curves(df, args.out)
    plot_roofline(df, args.out)
    plot_speedup(df, args.out)
    plot_time_breakdown(df, args.out)

    print(f"\nAll plots saved to '{args.out}/'")
