import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    experiments = []
    with open("all_results.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            exp_id = int(row["experiment_id"])
            latency = float(row["average_latency"].replace(" ms", ""))
            status = row["status"].strip()
            obs = row["observation"].strip()
            # Extract a short label from observation (first phrase before : or .)
            label = obs.split(":")[0].split(".")[0].strip()
            if len(label) > 50:
                label = label[:47] + "..."
            experiments.append((exp_id, latency, status, label))

    experiments.sort(key=lambda x: x[0])

    total = len(experiments)
    kept = sum(1 for e in experiments if e[2] == "keep")

    # Build running best line (step function through kept experiments)
    keep_exps = [(e[0], e[1]) for e in experiments if e[2] == "keep"]
    running_best_x = []
    running_best_y = []
    for i, (kx, ky) in enumerate(keep_exps):
        if i > 0:
            # Horizontal line from previous keep to this x
            running_best_x.append(kx)
            running_best_y.append(keep_exps[i - 1][1])
        # Vertical drop to this keep's y
        running_best_x.append(kx)
        running_best_y.append(ky)
    # Extend to the last experiment
    if keep_exps:
        running_best_x.append(experiments[-1][0])
        running_best_y.append(keep_exps[-1][1])

    # Separate keep and discard
    discard_x = [e[0] for e in experiments if e[2] == "discard"]
    discard_y = [e[1] for e in experiments if e[2] == "discard"]
    keep_x = [e[0] for e in experiments if e[2] == "keep"]
    keep_y = [e[1] for e in experiments if e[2] == "keep"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Running best step line
    ax.plot(running_best_x, running_best_y, color="#2ca02c", linewidth=2,
            zorder=2, label="Running best", solid_capstyle="round")

    # Discarded points - light gray, small
    ax.scatter(discard_x, discard_y, color="#cccccc", s=30, zorder=3,
               alpha=0.6, edgecolors="none", label="Discarded")

    # Kept points - green, prominent with border
    ax.scatter(keep_x, keep_y, color="#2ca02c", s=70, zorder=4,
               edgecolors="white", linewidths=1.5, label="Kept")

    # Annotate kept points with observation label (green) + latency (blue)
    for e in experiments:
        if e[2] == "keep":
            # Green italic label
            ax.annotate(
                e[3],
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
            # Blue latency value
            if e[1] >= 1:
                lat_str = f"{e[1]:.2f}ms"
            else:
                lat_str = f"{e[1]:.4f}ms"
            ax.annotate(
                lat_str,
                (e[0], e[1]),
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
        f"Sparse Matrix Multiplication Optimization Progress: {total} Experiments, {kept} Kept Improvements",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yscale("log")

    # Nice y-axis tick formatting
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}" if y >= 1 else f"{y:.2f}"
    ))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # X ticks at every 5, minor at every 1
    exp_ids = [e[0] for e in experiments]
    ax.set_xlim(min(exp_ids) - 1, max(exp_ids) + 2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Grid
    ax.grid(True, which="major", axis="both", alpha=0.15, linewidth=0.8)

    # Legend - top right
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9,
              edgecolor="#dddddd")

    # Clean spines
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    ax.tick_params(colors="#666666")

    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=150, facecolor="white")
    print("Saved results_plot.png")


if __name__ == "__main__":
    main()
