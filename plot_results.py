import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    experiments = []
    with open("all_results.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            exp_id = int(row["experiment_id"])
            serial_lat = row["serial_latency"].replace(" ms", "").strip()
            parallel_lat = row["parallel_latency"].replace(" ms", "").strip()
            serial_lat = float(serial_lat) if serial_lat != "—" else None
            parallel_lat = float(parallel_lat) if parallel_lat != "—" else None
            status = row["status"].strip()
            obs = row["observation"].strip()
            label = obs.split(":")[0].split(".")[0].strip()
            if len(label) > 50:
                label = label[:47] + "..."
            experiments.append((exp_id, serial_lat, parallel_lat, status, label))

    experiments.sort(key=lambda x: x[0])

    total = len(experiments)
    kept = sum(1 for e in experiments if e[3] == "keep")

    # Build running best lines for serial and parallel (step functions through kept experiments)
    def build_running_best(exps, lat_idx):
        """Build step-function (x, y) for running best of kept experiments."""
        keep_exps = [(e[0], e[lat_idx]) for e in exps if e[3] == "keep" and e[lat_idx] is not None]
        if not keep_exps:
            return [], []
        xs, ys = [], []
        for i, (kx, ky) in enumerate(keep_exps):
            if i > 0:
                xs.append(kx)
                ys.append(keep_exps[i - 1][1])
            xs.append(kx)
            ys.append(ky)
        xs.append(exps[-1][0])
        ys.append(keep_exps[-1][1])
        return xs, ys

    serial_best_x, serial_best_y = build_running_best(experiments, 1)
    parallel_best_x, parallel_best_y = build_running_best(experiments, 2)

    # Separate kept serial/parallel points
    serial_keep_x = [e[0] for e in experiments if e[3] == "keep" and e[1] is not None]
    serial_keep_y = [e[1] for e in experiments if e[3] == "keep" and e[1] is not None]
    parallel_keep_x = [e[0] for e in experiments if e[3] == "keep" and e[2] is not None]
    parallel_keep_y = [e[2] for e in experiments if e[3] == "keep" and e[2] is not None]

    # Discarded points (use serial latency)
    discard_x = [e[0] for e in experiments if e[3] == "discard" and e[1] is not None]
    discard_y = [e[1] for e in experiments if e[3] == "discard" and e[1] is not None]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Serial running best (primary — core algorithm)
    if serial_best_x:
        ax.plot(serial_best_x, serial_best_y, color="#2ca02c", linewidth=2.5,
                zorder=2, label="Serial (core algorithm)", solid_capstyle="round")

    # Parallel running best (secondary — infrastructure)
    if parallel_best_x:
        ax.plot(parallel_best_x, parallel_best_y, color="#1565C0", linewidth=2,
                zorder=2, label="Parallel (6 threads)", linestyle="--",
                solid_capstyle="round")

    # Discarded points
    ax.scatter(discard_x, discard_y, color="#cccccc", s=30, zorder=3,
               alpha=0.6, edgecolors="none", label="Discarded")

    # Kept serial points
    ax.scatter(serial_keep_x, serial_keep_y, color="#2ca02c", s=70, zorder=4,
               edgecolors="white", linewidths=1.5, label="Kept (serial)")

    # Kept parallel points
    if parallel_keep_x:
        ax.scatter(parallel_keep_x, parallel_keep_y, color="#1565C0", s=50,
                   zorder=4, edgecolors="white", linewidths=1.5,
                   marker="D", label="Kept (parallel)")

    # Annotate kept points
    for e in experiments:
        if e[3] != "keep":
            continue
        # Serial annotation (always present for kept)
        if e[1] is not None:
            ax.annotate(
                e[4],
                (e[0], e[1]),
                textcoords="offset points",
                xytext=(8, 12),
                fontsize=7,
                fontstyle="italic",
                color="#2ca02c",
                alpha=0.85,
                rotation=20,
                ha="left",
                va="bottom",
            )
            lat_str = f"{e[1]:.2f}ms" if e[1] >= 1 else f"{e[1]:.4f}ms"
            ax.annotate(
                lat_str,
                (e[0], e[1]),
                textcoords="offset points",
                xytext=(0, -12),
                fontsize=7,
                fontweight="bold",
                color="#2ca02c",
                ha="center",
                va="top",
            )
        # Parallel annotation (only when present)
        if e[2] is not None:
            lat_str = f"{e[2]:.2f}ms" if e[2] >= 1 else f"{e[2]:.4f}ms"
            ax.annotate(
                lat_str,
                (e[0], e[2]),
                textcoords="offset points",
                xytext=(0, -12),
                fontsize=7,
                fontweight="bold",
                color="#1565C0",
                ha="center",
                va="top",
            )

    # Axis config
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Average Latency ms (lower is better)", fontsize=12)
    ax.set_title(
        f"Sparse Matrix Multiplication — Algorithm Evolution ({total} experiments, {kept} kept)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yscale("log")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}" if y >= 1 else f"{y:.2f}"
    ))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    exp_ids = [e[0] for e in experiments]
    ax.set_xlim(min(exp_ids) - 1, max(exp_ids) + 2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.grid(True, which="major", axis="both", alpha=0.15, linewidth=0.8)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9,
              edgecolor="#dddddd")

    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    ax.tick_params(colors="#666666")

    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=150, facecolor="white")
    print("Saved results_plot.png")


if __name__ == "__main__":
    main()
