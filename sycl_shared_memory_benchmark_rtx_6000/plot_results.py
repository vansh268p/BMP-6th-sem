"""Plot SYCL malloc_shared benchmark results without pandas."""

import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)


NUMERIC = {
    "size",
    "py_total_ms",
    "kernel_ms",
    "gflops",
    "h2d_ms",
    "d2h_ms",
    "sycl_total_ms",
    "py_total_ms_std",
}


def read_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in NUMERIC:
            if key in row and row[key] != "":
                row[key] = float(row[key])
        if "size" in row:
            row["size"] = int(row["size"])
        row["source_csv"] = path.name
        row["non_kernel_ms"] = float(row.get("py_total_ms", 0.0)) - float(row.get("kernel_ms", 0.0))
    return rows


def default_csvs():
    avg10 = sorted(RES.glob("*avg10.csv"))
    if avg10:
        return avg10
    return sorted(RES.glob("*.csv"))


def grouped(rows, key):
    out = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out


def plot(rows, y, ylabel, title, filename, log_y=False):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, sub in grouped(rows, "method").items():
        sub = sorted(sub, key=lambda r: r["size"])
        ax.plot(
            [r["size"] for r in sub],
            [float(r[y]) for r in sub],
            marker="o",
            linewidth=2,
            label=method,
        )
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Matrix size N (N x N, FP32)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def write_combined(rows):
    keys = sorted({k for row in rows for k in row.keys()})
    with (OUT / "sycl_shared_combined.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", type=Path, default=None)
    args = ap.parse_args()

    csvs = args.csv if args.csv else default_csvs()
    if not csvs:
        raise SystemExit("no CSV files in results/")

    rows = []
    for path in csvs:
        rows.extend(read_csv(path))
    rows = [r for r in rows if r.get("kind") == "malloc_shared_tiled"]

    plot(rows, "py_total_ms", "End-to-end Python call time (ms)",
         "SYCL malloc_shared tiled matmul: total time",
         "sycl_shared_total_time.png", log_y=True)
    plot(rows, "gflops", "Throughput (GFLOP/s)",
         "SYCL malloc_shared tiled matmul: throughput",
         "sycl_shared_gflops.png")
    plot(rows, "kernel_ms", "Kernel time (ms)",
         "SYCL malloc_shared tiled matmul: kernel time",
         "sycl_shared_kernel_time.png", log_y=True)
    plot(rows, "non_kernel_ms", "Total minus kernel time (ms)",
         "SYCL malloc_shared tiled matmul: integration overhead",
         "sycl_shared_integration_overhead.png", log_y=True)
    write_combined(rows)
    print(f"[ok] wrote plots to {OUT}")
    print("[ok] input CSVs:")
    for path in csvs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
