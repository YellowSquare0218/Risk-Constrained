import csv
from pathlib import Path


OUT_DIR = Path("outputs")


COLORS = {
    "feedback_only": "#c44e52",
    "physics_prior_only": "#4c72b0",
    "feedback_physics": "#55a868",
    "full_risk_gate": "#8172b3",
}
LABELS = {
    "feedback_only": "Feedback only",
    "physics_prior_only": "Physics only",
    "feedback_physics": "Feedback + physics",
    "full_risk_gate": "Full risk gate",
}


def read_summary(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score_by_budget(rows, method):
    selected = [r for r in rows if r["method"] == method]
    selected.sort(key=lambda r: int(r["budget"]))
    return [int(r["budget"]) for r in selected], [float(r["score_mean"]) for r in selected]


def row_at(rows, method, budget=20):
    for row in rows:
        if row["method"] == method and int(row["budget"]) == budget:
            return row
    raise KeyError((method, budget))


def line_path(xs, ys, x_min, x_max, y_min, y_max, left, top, width, height):
    pts = []
    for x, y in zip(xs, ys):
        px = left + (x - x_min) / max(1e-9, x_max - x_min) * width
        py = top + (y_max - y) / max(1e-9, y_max - y_min) * height
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}.title{font-size:15px;font-weight:700}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:11px}</style>',
    ]


def draw_axes(lines, left, top, width, height, x_ticks, y_ticks, x_min, x_max, y_min, y_max, title, x_label, y_label):
    lines.append(f'<text class="title" x="{left + width / 2:.1f}" y="{top - 18}" text-anchor="middle">{title}</text>')
    for y in y_ticks:
        py = top + (y_max - y) / max(1e-9, y_max - y_min) * height
        lines.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{left + width}" y2="{py:.1f}"/>')
        lines.append(f'<text x="{left - 8}" y="{py + 4:.1f}" text-anchor="end">{y:.3f}</text>')
    for x in x_ticks:
        px = left + (x - x_min) / max(1e-9, x_max - x_min) * width
        lines.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + height}"/>')
        lines.append(f'<text x="{px:.1f}" y="{top + height + 18}" text-anchor="middle">{x}</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + height}"/>')
    lines.append(f'<text x="{left + width / 2:.1f}" y="{top + height + 40}" text-anchor="middle">{x_label}</text>')
    lines.append(f'<text x="{left - 48}" y="{top + height / 2:.1f}" text-anchor="middle" transform="rotate(-90 {left - 48} {top + height / 2:.1f})">{y_label}</text>')


def plot_series(lines, xs, ys, method, x_min, x_max, y_min, y_max, left, top, width, height):
    color = COLORS[method]
    pts = line_path(xs, ys, x_min, x_max, y_min, y_max, left, top, width, height)
    lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for x, y in zip(xs, ys):
        px = left + (x - x_min) / max(1e-9, x_max - x_min) * width
        py = top + (y_max - y) / max(1e-9, y_max - y_min) * height
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>')


def make_budget_curve():
    panels = [
        ("Clean feedback", read_summary(OUT_DIR / "offline_feedback_simulation_v2_24splits_summary.csv")),
        ("Noisy feedback", read_summary(OUT_DIR / "offline_feedback_simulation_v2_24splits_noisy_summary.csv")),
    ]
    methods = ["feedback_only", "physics_prior_only", "feedback_physics", "full_risk_gate"]
    width, height = 980, 390
    lines = svg_header(width, height)
    panel_specs = [(72, 58, 380, 235), (552, 58, 380, 235)]
    all_scores = []
    for _, rows in panels:
        for method in methods:
            _, ys = score_by_budget(rows, method)
            all_scores.extend(ys)
    y_min = min(all_scores) - 0.001
    y_max = max(all_scores) + 0.001
    y_ticks = [round(y_min + i * (y_max - y_min) / 4, 3) for i in range(5)]
    for (title, rows), (left, top, pw, ph) in zip(panels, panel_specs):
        draw_axes(lines, left, top, pw, ph, [0, 5, 10, 15, 20], y_ticks, 0, 20, y_min, y_max, title, "Feedback budget", "Score")
        for method in methods:
            xs, ys = score_by_budget(rows, method)
            plot_series(lines, xs, ys, method, 0, 20, y_min, y_max, left, top, pw, ph)
    legend_x, legend_y = 650, 330
    for i, method in enumerate(methods):
        y = legend_y + i * 15
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{COLORS[method]}" stroke-width="2.5"/>')
        lines.append(f'<text class="legend" x="{legend_x + 34}" y="{y + 4}">{LABELS[method]}</text>')
    lines.append("</svg>")
    out = OUT_DIR / "paper_fig1_feedback_budget_curve.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def make_robustness_curve():
    settings = [
        ("Clean", OUT_DIR / "offline_feedback_simulation_v2_24splits_summary.csv"),
        ("0.0004+flip", OUT_DIR / "offline_feedback_simulation_v2_24splits_noise0004_flip005_summary.csv"),
        ("0.0008+flip", OUT_DIR / "offline_feedback_simulation_v2_24splits_noisy_summary.csv"),
        ("0.0016+flip", OUT_DIR / "offline_feedback_simulation_v2_24splits_noise0016_flip005_summary.csv"),
    ]
    methods = ["feedback_only", "physics_prior_only", "feedback_physics", "full_risk_gate"]
    values_by_method = {}
    all_values = []
    for method in methods:
        values = []
        for _, path in settings:
            rows = read_summary(path)
            values.append(float(row_at(rows, method, 20)["score_mean"]))
        values_by_method[method] = values
        all_values.extend(values)
    width, height = 760, 430
    left, top, pw, ph = 82, 62, 590, 250
    lines = svg_header(width, height)
    y_min = min(all_values) - 0.001
    y_max = max(all_values) + 0.001
    y_ticks = [round(y_min + i * (y_max - y_min) / 4, 3) for i in range(5)]
    x_ticks = list(range(len(settings)))
    draw_axes(lines, left, top, pw, ph, x_ticks, y_ticks, 0, len(settings) - 1, y_min, y_max, "Robustness to noisy aggregate feedback", "Noise setting", "Budget-20 score")
    for i, (label, _) in enumerate(settings):
        px = left + i / (len(settings) - 1) * pw
        lines.append(f'<text x="{px:.1f}" y="{top + ph + 34}" text-anchor="middle" transform="rotate(16 {px:.1f} {top + ph + 34})">{label}</text>')
    for method in methods:
        xs = list(range(len(settings)))
        plot_series(lines, xs, values_by_method[method], method, 0, len(settings) - 1, y_min, y_max, left, top, pw, ph)
    legend_x, legend_y = 505, 344
    for i, method in enumerate(methods):
        y = legend_y + i * 15
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{COLORS[method]}" stroke-width="2.5"/>')
        lines.append(f'<text class="legend" x="{legend_x + 34}" y="{y + 4}">{LABELS[method]}</text>')
    lines.append("</svg>")
    out = OUT_DIR / "paper_fig2_robustness_sweep.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    for path in [make_budget_curve(), make_robustness_curve()]:
        print(path)


if __name__ == "__main__":
    main()
