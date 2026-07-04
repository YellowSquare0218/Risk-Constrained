import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("outputs")

METHODS = ["feedback_only", "physics_prior_only", "feedback_physics", "full_risk_gate"]
METHOD_LABELS = {
    "feedback_only": "Feedback only",
    "physics_prior_only": "Biomechanical prior",
    "feedback_physics": "Feedback + bio gate",
    "full_risk_gate": "Full risk gate",
}
COLORS = {
    "feedback_only": "#C44E52",
    "physics_prior_only": "#4C72B0",
    "feedback_physics": "#55A868",
    "full_risk_gate": "#8172B3",
}
RGB = {k: tuple(int(v[i : i + 2], 16) for i in (1, 3, 5)) for k, v in COLORS.items()}
INK = "#202124"
GRID = "#DADCE0"
MUTED = "#5F6368"


def font(size=14, bold=False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summary_value(rows, method, budget, field="score_mean"):
    for row in rows:
        if row["method"] == method and int(row["budget"]) == budget:
            return float(row[field])
    raise KeyError((method, budget, field))


def escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}.title{font-size:18px;font-weight:700}.axis{font-size:12px}.tick{font-size:11px;fill:#5F6368}.legend{font-size:12px}.panel{font-size:15px;font-weight:700}.note{font-size:11px;fill:#5F6368}</style>',
        ]

    def text(self, x, y, text, cls="", anchor="start", rotate=None):
        transform = f' transform="rotate({rotate[0]} {rotate[1]} {rotate[2]})"' if rotate else ""
        self.lines.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{transform}>{escape(text)}</text>')

    def line(self, x1, y1, x2, y2, color=INK, width=1, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>')

    def polyline(self, pts, color, width=2.5):
        s = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.lines.append(f'<polyline points="{s}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')

    def circle(self, x, y, r, color):
        self.lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"/>')

    def rect(self, x, y, w, h, fill="none", stroke=INK, width=1):
        self.lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def save(self, path):
        self.lines.append("</svg>")
        path.write_text("\n".join(self.lines), encoding="utf-8")


def scale_point(x, y, x_min, x_max, y_min, y_max, left, top, width, height):
    px = left + (x - x_min) / (x_max - x_min) * width
    py = top + (y_max - y) / (y_max - y_min) * height
    return px, py


def draw_axes_svg(svg, left, top, width, height, x_ticks, y_ticks, x_min, x_max, y_min, y_max, x_label, y_label):
    for y in y_ticks:
        _, py = scale_point(x_min, y, x_min, x_max, y_min, y_max, left, top, width, height)
        svg.line(left, py, left + width, py, GRID, 1)
        svg.text(left - 8, py + 4, f"{y:.3f}", "tick", "end")
    for x in x_ticks:
        px, _ = scale_point(x, y_min, x_min, x_max, y_min, y_max, left, top, width, height)
        svg.line(px, top, px, top + height, "#EEF0F2", 1)
        svg.text(px, top + height + 18, str(x), "tick", "middle")
    svg.line(left, top + height, left + width, top + height, INK, 1.2)
    svg.line(left, top, left, top + height, INK, 1.2)
    svg.text(left + width / 2, top + height + 42, x_label, "axis", "middle")
    svg.text(left - 52, top + height / 2, y_label, "axis", "middle", rotate=(-90, left - 52, top + height / 2))


def draw_axes_png(draw, left, top, width, height, x_ticks, y_ticks, x_min, x_max, y_min, y_max, x_label, y_label, f_tick, f_axis):
    for y in y_ticks:
        _, py = scale_point(x_min, y, x_min, x_max, y_min, y_max, left, top, width, height)
        draw.line((left, py, left + width, py), fill="#DADCE0", width=1)
        draw.text((left - 8, py - 7), f"{y:.3f}", fill="#5F6368", font=f_tick, anchor="ra")
    for x in x_ticks:
        px, _ = scale_point(x, y_min, x_min, x_max, y_min, y_max, left, top, width, height)
        draw.line((px, top, px, top + height), fill="#EEF0F2", width=1)
        draw.text((px, top + height + 6), str(x), fill="#5F6368", font=f_tick, anchor="ma")
    draw.line((left, top + height, left + width, top + height), fill="#202124", width=2)
    draw.line((left, top, left, top + height), fill="#202124", width=2)
    draw.text((left + width / 2, top + height + 32), x_label, fill="#202124", font=f_axis, anchor="mm")
    draw.text((left, top - 24), y_label, fill="#202124", font=f_axis, anchor="la")


def make_budget_curve():
    clean = read_csv(OUT / "offline_feedback_simulation_v5_samesplit_24splits_summary.csv")
    noisy = read_csv(OUT / "offline_feedback_simulation_v5_samesplit_24splits_noisy_summary.csv")
    panels = [("A", "Clean feedback", clean), ("B", "Noisy feedback", noisy)]
    budgets = [0, 1, 3, 5, 10, 20]
    all_scores = [summary_value(rows, m, b) for _, _, rows in panels for m in METHODS for b in budgets]
    y_min = math.floor((min(all_scores) - 0.001) * 1000) / 1000
    y_max = math.ceil((max(all_scores) + 0.001) * 1000) / 1000
    y_ticks = [y_min + i * (y_max - y_min) / 4 for i in range(5)]

    svg = Svg(1000, 430)
    svg.text(500, 28, "Feedback-budget curves under clean and noisy aggregate feedback", "title", "middle")
    panel_boxes = [(78, 72, 385, 240), (570, 72, 385, 240)]
    for (letter, title, rows), (left, top, w, h) in zip(panels, panel_boxes):
        svg.text(left, top - 24, f"{letter}. {title}", "panel")
        draw_axes_svg(svg, left, top, w, h, [0, 5, 10, 15, 20], y_ticks, 0, 20, y_min, y_max, "Feedback budget", "Score")
        for method in METHODS:
            pts = [scale_point(b, summary_value(rows, method, b), 0, 20, y_min, y_max, left, top, w, h) for b in budgets]
            svg.polyline(pts, COLORS[method], 2.6)
            for x, y in pts:
                svg.circle(x, y, 3.6, COLORS[method])
    lx, ly = 290, 368
    for i, method in enumerate(METHODS):
        x = lx + (i % 2) * 250
        y = ly + (i // 2) * 22
        svg.line(x, y, x + 28, y, COLORS[method], 3)
        svg.text(x + 36, y + 4, METHOD_LABELS[method], "legend")
    svg.text(500, 414, "Higher is better. Curves show mean score over 24 pseudo-test splits.", "note", "middle")
    svg_path = OUT / "paper_fig1_budget_curve_polished.svg"
    svg.save(svg_path)

    img = Image.new("RGB", (2000, 860), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_panel, f_axis, f_tick, f_legend = font(34, True), font(26, True), font(22), font(20), font(22)
    draw.text((1000, 30), "Feedback-budget curves under clean and noisy aggregate feedback", fill="#202124", font=f_title, anchor="ma")
    for (letter, title, rows), (left, top, w, h) in zip(panels, [(156, 144, 770, 480), (1140, 144, 770, 480)]):
        draw.text((left, top - 48), f"{letter}. {title}", fill="#202124", font=f_panel)
        draw_axes_png(draw, left, top, w, h, [0, 5, 10, 15, 20], y_ticks, 0, 20, y_min, y_max, "Feedback budget", "Score", f_tick, f_axis)
        for method in METHODS:
            pts = [scale_point(b, summary_value(rows, method, b), 0, 20, y_min, y_max, left, top, w, h) for b in budgets]
            pts = [(int(x), int(y)) for x, y in pts]
            draw.line(pts, fill=RGB[method], width=5, joint="curve")
            for x, y in pts:
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=RGB[method])
    for i, method in enumerate(METHODS):
        x = 580 + (i % 2) * 500
        y = 735 + (i // 2) * 42
        draw.line((x, y, x + 56, y), fill=RGB[method], width=6)
        draw.text((x + 72, y), METHOD_LABELS[method], fill="#202124", font=f_legend, anchor="lm")
    draw.text((1000, 828), "Higher is better. Curves show mean score over 24 pseudo-test splits.", fill="#5F6368", font=font(20), anchor="mm")
    png_path = OUT / "paper_fig1_budget_curve_polished.png"
    img.save(png_path, dpi=(300, 300))
    return svg_path, png_path


def quartiles(values):
    vals = sorted(values)
    n = len(vals)
    def pct(p):
        k = (n - 1) * p
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - k) + vals[hi] * (k - lo)
    return vals[0], pct(0.25), pct(0.5), pct(0.75), vals[-1]


def make_gain_boxplot():
    rows = read_csv(OUT / "paper_per_split_budget20_gains_samesplit.csv")
    scenarios = ["Clean", "Noisy", "Feedback-50", "Feedback-50 noisy"]
    methods = ["feedback_only", "physics_prior_only", "full_risk_gate"]
    grouped = {(s, m): [] for s in scenarios for m in methods}
    for row in rows:
        key = (row["scenario"], row["method"])
        if key in grouped:
            grouped[key].append(float(row["gain_over_initial"]))
    values = [v for arr in grouped.values() for v in arr]
    y_min = math.floor((min(values) - 0.001) * 1000) / 1000
    y_max = math.ceil((max(values) + 0.001) * 1000) / 1000
    y_ticks = [y_min + i * (y_max - y_min) / 5 for i in range(6)]

    svg = Svg(980, 470)
    svg.text(490, 28, "Split-level gains at budget 20", "title", "middle")
    left, top, w, h = 86, 70, 820, 280
    x_min, x_max = -0.5, len(scenarios) - 0.5
    for y in y_ticks:
        _, py = scale_point(x_min, y, x_min, x_max, y_min, y_max, left, top, w, h)
        svg.line(left, py, left + w, py, GRID, 1)
        svg.text(left - 8, py + 4, f"{y:.3f}", "tick", "end")
    svg.line(left, top + h, left + w, top + h, INK, 1.2)
    svg.line(left, top, left, top + h, INK, 1.2)
    svg.text(left, top - 12, "Gain over initial", "axis")
    svg.text(left + w / 2, top + h + 42, "Feedback setting", "axis", "middle")
    for i, scenario in enumerate(scenarios):
        px, _ = scale_point(i, y_min, x_min, x_max, y_min, y_max, left, top, w, h)
        svg.line(px, top, px, top + h, "#EEF0F2", 1)
        svg.text(px, top + h + 20, scenario.replace(" ", "\n"), "tick", "middle")
    offsets = {"feedback_only": -0.17, "physics_prior_only": 0.0, "full_risk_gate": 0.17}
    box_w = 30
    for i, scenario in enumerate(scenarios):
        for method in methods:
            vals = grouped[(scenario, method)]
            mn, q1, med, q3, mx = quartiles(vals)
            cx, _ = scale_point(i + offsets[method], y_min, x_min, x_max, y_min, y_max, left, top, w, h)
            y_mn = scale_point(0, mn, 0, 1, y_min, y_max, left, top, w, h)[1]
            y_q1 = scale_point(0, q1, 0, 1, y_min, y_max, left, top, w, h)[1]
            y_med = scale_point(0, med, 0, 1, y_min, y_max, left, top, w, h)[1]
            y_q3 = scale_point(0, q3, 0, 1, y_min, y_max, left, top, w, h)[1]
            y_mx = scale_point(0, mx, 0, 1, y_min, y_max, left, top, w, h)[1]
            color = COLORS[method]
            svg.line(cx, y_mn, cx, y_mx, color, 1.5)
            svg.rect(cx - box_w / 2, y_q3, box_w, y_q1 - y_q3, fill="#FFFFFF", stroke=color, width=2)
            svg.line(cx - box_w / 2, y_med, cx + box_w / 2, y_med, color, 2)
            svg.line(cx - box_w / 4, y_mn, cx + box_w / 4, y_mn, color, 1.5)
            svg.line(cx - box_w / 4, y_mx, cx + box_w / 4, y_mx, color, 1.5)
    lx, ly = 190, 410
    for i, method in enumerate(methods):
        x = lx + i * 250
        svg.rect(x, ly - 12, 18, 18, fill="#FFFFFF", stroke=COLORS[method], width=2)
        svg.text(x + 28, ly + 2, METHOD_LABELS[method], "legend")
    svg.text(490, 452, "Boxes show median and interquartile range over 24 pseudo-test splits.", "note", "middle")
    svg_path = OUT / "paper_fig2_split_gain_boxplot_polished.svg"
    svg.save(svg_path)

    img = Image.new("RGB", (1960, 940), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_axis, f_tick, f_legend = font(34, True), font(22), font(20), font(22)
    draw.text((980, 32), "Split-level gains at budget 20", fill="#202124", font=f_title, anchor="ma")
    left2, top2, w2, h2 = 172, 140, 1640, 560
    for y in y_ticks:
        _, py = scale_point(x_min, y, x_min, x_max, y_min, y_max, left2, top2, w2, h2)
        draw.line((left2, py, left2 + w2, py), fill="#DADCE0", width=1)
        draw.text((left2 - 8, py - 7), f"{y:.3f}", fill="#5F6368", font=f_tick, anchor="ra")
    draw.line((left2, top2 + h2, left2 + w2, top2 + h2), fill="#202124", width=2)
    draw.line((left2, top2, left2, top2 + h2), fill="#202124", width=2)
    draw.text((left2, top2 - 28), "Gain over initial", fill="#202124", font=f_axis, anchor="la")
    draw.text((left2 + w2 / 2, top2 + h2 + 72), "Feedback setting", fill="#202124", font=f_axis, anchor="mm")
    for i, scenario in enumerate(scenarios):
        px, _ = scale_point(i, y_min, x_min, x_max, y_min, y_max, left2, top2, w2, h2)
        draw.line((px, top2, px, top2 + h2), fill="#EEF0F2", width=1)
        draw.text((px, top2 + h2 + 38), scenario, fill="#5F6368", font=f_tick, anchor="ma")
    for i, scenario in enumerate(scenarios):
        for method in methods:
            vals = grouped[(scenario, method)]
            mn, q1, med, q3, mx = quartiles(vals)
            cx, _ = scale_point(i + offsets[method], y_min, x_min, x_max, y_min, y_max, left2, top2, w2, h2)
            y_mn = scale_point(0, mn, 0, 1, y_min, y_max, left2, top2, w2, h2)[1]
            y_q1 = scale_point(0, q1, 0, 1, y_min, y_max, left2, top2, w2, h2)[1]
            y_med = scale_point(0, med, 0, 1, y_min, y_max, left2, top2, w2, h2)[1]
            y_q3 = scale_point(0, q3, 0, 1, y_min, y_max, left2, top2, w2, h2)[1]
            y_mx = scale_point(0, mx, 0, 1, y_min, y_max, left2, top2, w2, h2)[1]
            c = RGB[method]
            cx = int(cx)
            draw.line((cx, y_mn, cx, y_mx), fill=c, width=3)
            draw.rectangle((cx - 30, y_q3, cx + 30, y_q1), outline=c, width=4)
            draw.line((cx - 30, y_med, cx + 30, y_med), fill=c, width=4)
            draw.line((cx - 14, y_mn, cx + 14, y_mn), fill=c, width=3)
            draw.line((cx - 14, y_mx, cx + 14, y_mx), fill=c, width=3)
    for i, method in enumerate(methods):
        x = 380 + i * 500
        y = 820
        draw.rectangle((x, y - 20, x + 36, y + 16), outline=RGB[method], width=4)
        draw.text((x + 54, y), METHOD_LABELS[method], fill="#202124", font=f_legend, anchor="lm")
    draw.text((980, 900), "Boxes show median and interquartile range over 24 pseudo-test splits.", fill="#5F6368", font=font(20), anchor="mm")
    png_path = OUT / "paper_fig2_split_gain_boxplot_polished.png"
    img.save(png_path, dpi=(300, 300))
    return svg_path, png_path


def main():
    paths = []
    paths.extend(make_budget_curve())
    paths.extend(make_gain_boxplot())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
