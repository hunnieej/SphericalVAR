import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot rank vs head score")
    parser.add_argument("json_path", type=str, help="Head classification JSON path")
    parser.add_argument(
        "--output", type=str, default="", help="Optional output image path"
    )
    parser.add_argument("--title", type=str, default="", help="Optional title")
    parser.add_argument("--log_y", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def load_results(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No 'results' found in {json_path}")
    results = sorted(results, key=lambda x: x.get("rank", 10**9))
    return data, results


def alpha_cutoffs(alpha_values, total):
    out = []
    for alpha in alpha_values or []:
        k = max(1, int(math.ceil(total * float(alpha))))
        out.append((float(alpha), k))
    return out


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_svg(data, results, title, log_y):
    width, height = 1200, 700
    left, right, top, bottom = 90, 40, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    ranks = [int(item["rank"]) for item in results]
    scores = [float(item["mean_query_variance"]) for item in results]
    labels = [item.get("label", "") for item in results]
    n = len(results)

    if log_y:
        positive = [s for s in scores if s > 0]
        ymin = min(positive) if positive else 1e-12
        ymax = max(scores) if scores else 1.0
        yvals = [math.log10(max(s, ymin)) for s in scores]
        y0 = math.log10(ymin)
        y1 = math.log10(ymax)
    else:
        ymin = min(scores) if scores else 0.0
        ymax = max(scores) if scores else 1.0
        yvals = scores
        y0, y1 = ymin, ymax

    if abs(y1 - y0) < 1e-12:
        y1 = y0 + 1.0

    def x_map(rank):
        if n <= 1:
            return left
        return left + (rank - 1) * plot_w / (n - 1)

    def y_map(y):
        return top + plot_h - (y - y0) * plot_h / (y1 - y0)

    pts = [f"{x_map(r):.2f},{y_map(y):.2f}" for r, y in zip(ranks, yvals)]
    structural_pts = [
        (x_map(r), y_map(y))
        for r, y, label in zip(ranks, yvals, labels)
        if label == "structural"
    ]
    contextual_pts = [
        (x_map(r), y_map(y))
        for r, y, label in zip(ranks, yvals, labels)
        if label != "structural"
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black" stroke-width="1.5"/>',
    ]

    for i in range(6):
        frac = i / 5
        y = top + plot_h - frac * plot_h
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        raw = y0 + frac * (y1 - y0)
        label = f"1e{raw:.1f}" if log_y else f"{raw:.3g}"
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{svg_escape(label)}</text>'
        )

    xticks = 8
    for i in range(xticks + 1):
        rank = 1 + i * (n - 1) / xticks if n > 1 else 1
        x = x_map(rank)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eeeeee" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="12" font-family="Arial">{int(round(rank))}</text>'
        )

    for alpha, cutoff in alpha_cutoffs(data.get("alpha_values"), n):
        x = x_map(cutoff)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#55a868" stroke-width="1.5" stroke-dasharray="5,4"/>'
        )
        lines.append(
            f'<text x="{x - 4:.2f}" y="{top + 18}" text-anchor="end" font-size="11" font-family="Arial" fill="#55a868" transform="rotate(-90 {x - 4:.2f},{top + 18})">alpha={alpha:.1f}</text>'
        )

    lines.append(
        f'<polyline fill="none" stroke="black" stroke-width="1.4" points="{" ".join(pts)}"/>'
    )

    for x, y in contextual_pts:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="#4c72b0"/>')
    for x, y in structural_pts:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="#c44e52"/>')

    lines.extend(
        [
            f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="16" font-family="Arial">Rank</text>',
            f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 22,{height / 2:.1f})">Mean Query Variance</text>',
            f'<circle cx="{width - 185}" cy="{top + 18}" r="5" fill="#c44e52"/><text x="{width - 172}" y="{top + 22}" font-size="13" font-family="Arial">structural</text>',
            f'<circle cx="{width - 95}" cy="{top + 18}" r="5" fill="#4c72b0"/><text x="{width - 82}" y="{top + 22}" font-size="13" font-family="Arial">contextual</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    data, results = load_results(json_path)
    title = args.title or f"Rank vs Mean Query Variance - {json_path.name}"

    output_path = (
        Path(args.output)
        if args.output
        else json_path.with_name(f"{json_path.stem}_rank_vs_score.png")
    )
    svg_path = output_path.with_suffix(".svg")
    svg = build_svg(data, results, title, bool(args.log_y))
    svg_path.write_text(svg, encoding="utf-8")

    if output_path.suffix.lower() == ".svg":
        print(f"Saved plot to {output_path}")
        return

    if shutil.which("convert"):
        subprocess.run(["convert", str(svg_path), str(output_path)], check=True)
        print(f"Saved plot to {output_path}")
    else:
        print(
            f"Saved SVG plot to {svg_path} (ImageMagick 'convert' not found for PNG export)"
        )


if __name__ == "__main__":
    main()
