import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path


FEATURES = [
    "center_score_mean",
    "horizontal_score_mean",
    "vertical_score_mean",
    "cross_score_mean",
    "balance_score_mean",
    "spread_score_mean",
    "self_peak_score_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot top feature rankings from aggregated CSV"
    )
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def load_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def make_bar_svg(feature: str, rows, top_k: int):
    ranked = sorted(rows, key=lambda r: float(r[feature]), reverse=True)[:top_k]
    width, height = 1200, 700
    left, right, top, bottom = 240, 40, 60, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(float(r[feature]) for r in ranked) if ranked else 1.0
    bar_h = plot_h / max(len(ranked), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">Top {len(ranked)} by {feature}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black" stroke-width="1.5"/>',
    ]
    for i, row in enumerate(ranked):
        y = top + i * bar_h + 4
        val = float(row[feature])
        bw = 0 if max_val <= 0 else (val / max_val) * (plot_w - 10)
        label = f"L{int(row['layer']):02d} H{int(row['head']):02d}"
        lines.append(
            f'<text x="{left - 10}" y="{y + bar_h * 0.65:.2f}" text-anchor="end" font-size="13" font-family="Arial">{svg_escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left + 2}" y="{y:.2f}" width="{bw:.2f}" height="{max(bar_h - 8, 1):.2f}" fill="#4c72b0"/>'
        )
        lines.append(
            f'<text x="{left + bw + 10:.2f}" y="{y + bar_h * 0.65:.2f}" font-size="12" font-family="Arial">{val:.6f}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def save_svg_or_png(out_base: Path, svg_text: str):
    svg_path = out_base.with_suffix(".svg")
    svg_path.write_text(svg_text, encoding="utf-8")
    png_path = out_base.with_suffix(".png")
    if shutil.which("convert"):
        subprocess.run(["convert", str(svg_path), str(png_path)], check=True)
        return png_path
    return svg_path


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    rows = load_rows(csv_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else csv_path.parent / "feature_rankings"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    for feature in FEATURES:
        svg = make_bar_svg(feature, rows, args.top_k)
        out = save_svg_or_png(output_dir / feature, svg)
        summary_lines.append(f"{feature}: {out.name}")

    (output_dir / "README.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(f"Saved feature ranking plots to {output_dir}")


if __name__ == "__main__":
    main()
