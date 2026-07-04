import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("outputs")
SOURCES = [
    OUT / "offline_feedback_simulation_v5_samesplit_24splits_noisy.json",
    OUT / "offline_feedback_simulation_v6_mainsplit_feedback50_full_noisy.json",
]
CASE_FILE = OUT / "offline_feedback_case_studies_samesplit_2026-06-08.json"

W, H = 1200, 455
SCALE = 2

INK = "#1F2933"
MUTED = "#667085"
GRID = "#E5E7EB"
BORDER = "#D0D7DE"
GRAY = "#E9EDF2"
GRAY_DARK = "#9AA4B2"
GREEN = "#2B8C5A"
GREEN_LIGHT = "#CFE8DA"
RED = "#B94A50"
RED_LIGHT = "#F1C7CB"
BLUE = "#4979B5"
BLUE_LIGHT = "#D9E7F6"
PURPLE = "#7464AC"
AMBER = "#C9842E"


def font(size=10, bold=False):
    names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size * SCALE)
        except OSError:
            continue
    return ImageFont.load_default()


def escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rgba(hex_color, alpha=255):
    named = {"white": "#FFFFFF", "black": "#000000"}
    hex_color = named.get(hex_color, hex_color).lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def full_gate_allows(row):
    p_after = float(row["p_after"])
    return p_after >= 0.42 and (bool(row["total_agrees"]) or p_after >= 0.48)


def iter_feedback_only_accepted():
    for path in SOURCES:
        report = json.loads(path.read_text(encoding="utf-8"))
        for split in report["splits"]:
            test_ids = split.get("test_ids_ordered", split.get("test_ids", []))
            for row in split["traces"].get("feedback_only_budget20", []):
                if not row.get("accepted"):
                    continue
                item = dict(row)
                row_idx = int(item["row_idx"])
                item["source"] = path.stem
                item["seed"] = int(split["seed"])
                item["patient_id"] = int(test_ids[row_idx]) if row_idx < len(test_ids) else None
                item["evaluation_delta"] = float(item.get("evaluation_delta", item.get("aggregate_delta", 0.0)))
                item["observed_delta"] = float(item.get("observed_delta", 0.0))
                item["p_after"] = float(item.get("p_after", 0.0))
                item["would_pass_full_gate"] = full_gate_allows(item)
                yield item


def load_cases():
    return json.loads(CASE_FILE.read_text(encoding="utf-8"))


def summarize_points(points):
    harmful = [p for p in points if p["evaluation_delta"] <= 0]
    beneficial = [p for p in points if p["evaluation_delta"] > 0]
    harmful_blocked = [p for p in harmful if not p["would_pass_full_gate"]]
    harmful_pass = [p for p in harmful if p["would_pass_full_gate"]]
    beneficial_blocked = [p for p in beneficial if not p["would_pass_full_gate"]]
    beneficial_pass = [p for p in beneficial if p["would_pass_full_gate"]]
    return {
        "accepted": len(points),
        "harmful": len(harmful),
        "harmful_blocked": len(harmful_blocked),
        "harmful_pass": len(harmful_pass),
        "beneficial": len(beneficial),
        "beneficial_blocked": len(beneficial_blocked),
        "beneficial_pass": len(beneficial_pass),
        "blocked": len(harmful_blocked) + len(beneficial_blocked),
        "passed": len(harmful_pass) + len(beneficial_pass),
        "harmful_blocked_pct": len(harmful_blocked) / max(1, len(harmful)),
        "beneficial_pass_pct": len(beneficial_pass) / max(1, len(beneficial)),
    }


def bezier_points(x1, y1, x2, y2, n=28):
    c1x = x1 + (x2 - x1) * 0.45
    c2x = x1 + (x2 - x1) * 0.55
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * x1 + 3 * mt**2 * t * c1x + 3 * mt * t**2 * c2x + t**3 * x2
        y = mt**3 * y1 + 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2 + t**3 * y2
        pts.append((x, y))
    return pts


def ribbon_polygon(x1, y1a, y1b, x2, y2a, y2b):
    top = bezier_points(x1, y1a, x2, y2a)
    bot = bezier_points(x2, y2b, x1, y1b)
    return top + bot


class Svg:
    def __init__(self, width=W, height=H):
        self.lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            (
                "<style>"
                "text{font-family:Arial,Helvetica,sans-serif;fill:#1F2933}"
                ".panel{font-size:20px;font-weight:700}"
                ".head{font-size:15px;font-weight:700}"
                ".small{font-size:10px}"
                ".note{font-size:9.5px;fill:#667085}"
                ".metric{font-size:23px;font-weight:700}"
                "</style>"
            ),
        ]

    def text(self, x, y, s, cls="small", anchor="start"):
        self.lines.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{escape(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke="none", width=1, radius=0, opacity=None):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{op}/>'
        )

    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>')

    def circle(self, x, y, r, fill, stroke="white", width=0.8):
        self.lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def ribbon(self, x1, y1a, y1b, x2, y2a, y2b, fill, opacity=0.58):
        c1x = x1 + (x2 - x1) * 0.45
        c2x = x1 + (x2 - x1) * 0.55
        d = (
            f"M {x1:.1f} {y1a:.1f} "
            f"C {c1x:.1f} {y1a:.1f}, {c2x:.1f} {y2a:.1f}, {x2:.1f} {y2a:.1f} "
            f"L {x2:.1f} {y2b:.1f} "
            f"C {c2x:.1f} {y2b:.1f}, {c1x:.1f} {y1b:.1f}, {x1:.1f} {y1b:.1f} Z"
        )
        self.lines.append(f'<path d="{d}" fill="{fill}" opacity="{opacity}"/>')

    def save(self, path):
        self.lines.append("</svg>")
        path.write_text("\n".join(self.lines), encoding="utf-8")


class Png:
    def __init__(self):
        self.img = Image.new("RGBA", (W * SCALE, H * SCALE), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)

    def xy(self, x, y):
        return int(round(x * SCALE)), int(round(y * SCALE))

    def text(self, x, y, s, size=10, bold=False, fill=INK, anchor="la"):
        self.draw.text(self.xy(x, y), s, font=font(size, bold), fill=rgba(fill), anchor=anchor)

    def rect(self, x, y, w, h, fill, outline=None, width=1, radius=0):
        box = (*self.xy(x, y), *self.xy(x + w, y + h))
        if radius:
            self.draw.rounded_rectangle(
                box,
                radius=int(radius * SCALE),
                fill=rgba(fill),
                outline=rgba(outline) if outline else None,
                width=max(1, int(width * SCALE)),
            )
        else:
            self.draw.rectangle(
                box,
                fill=rgba(fill),
                outline=rgba(outline) if outline else None,
                width=max(1, int(width * SCALE)),
            )

    def line(self, x1, y1, x2, y2, fill=GRID, width=1):
        self.draw.line((*self.xy(x1, y1), *self.xy(x2, y2)), fill=rgba(fill), width=max(1, int(width * SCALE)))

    def ribbon(self, x1, y1a, y1b, x2, y2a, y2b, fill, alpha=125):
        pts = ribbon_polygon(x1, y1a, y1b, x2, y2a, y2b)
        self.draw.polygon([(int(x * SCALE), int(y * SCALE)) for x, y in pts], fill=rgba(fill, alpha))

    def circle(self, x, y, r, fill, outline="white"):
        box = (*self.xy(x - r, y - r), *self.xy(x + r, y + r))
        self.draw.ellipse(box, fill=rgba(fill), outline=rgba(outline), width=max(1, int(0.8 * SCALE)))

    def save(self, path):
        self.img.convert("RGB").save(path, dpi=(300, 300))


def layout(summary):
    top, total_h = 92, 260
    unit = total_h / summary["accepted"]
    x_src, x_mid, x_dec = 74, 300, 548
    bw = 34
    gap = 18

    b = summary["beneficial"] * unit
    h = summary["harmful"] * unit
    bb = summary["beneficial_blocked"] * unit
    bp = summary["beneficial_pass"] * unit
    hb = summary["harmful_blocked"] * unit
    hp = summary["harmful_pass"] * unit
    blocked = summary["blocked"] * unit
    passed = summary["passed"] * unit
    y_pass = top + blocked + gap

    return {
        "top": top,
        "total_h": total_h,
        "unit": unit,
        "bw": bw,
        "x_src": x_src,
        "x_mid": x_mid,
        "x_dec": x_dec,
        "source_beneficial": (top, top + b),
        "source_harmful": (top + b, top + b + h),
        "mid_beneficial": (top, top + b),
        "mid_harmful": (top + b, top + b + h),
        "mid_beneficial_blocked": (top, top + bb),
        "mid_beneficial_pass": (top + bb, top + bb + bp),
        "mid_harmful_blocked": (top + b, top + b + hb),
        "mid_harmful_pass": (top + b + hb, top + b + hb + hp),
        "dec_blocked": (top, top + blocked),
        "dec_blocked_beneficial": (top, top + bb),
        "dec_blocked_harmful": (top + bb, top + bb + hb),
        "dec_passed": (y_pass, y_pass + passed),
        "dec_passed_beneficial": (y_pass, y_pass + bp),
        "dec_passed_harmful": (y_pass + bp, y_pass + bp + hp),
    }


def draw_flow(canvas, summary, is_svg=False):
    L = layout(summary)
    bw = L["bw"]
    x0, x1, x2 = L["x_src"], L["x_mid"], L["x_dec"]

    canvas.text(35, 43, "A", "panel" if is_svg else 20, True if not is_svg else "start")
    if is_svg:
        canvas.text(62, 43, "Decision audit of feedback-only accepted updates", "head")
    else:
        canvas.text(62, 29, "Decision audit of feedback-only accepted updates", 15, True)

    # Source to hidden outcome.
    canvas.ribbon(x0 + bw, *L["source_beneficial"], x1, *L["mid_beneficial"], GREEN, 0.28 if is_svg else 72)
    canvas.ribbon(x0 + bw, *L["source_harmful"], x1, *L["mid_harmful"], RED, 0.28 if is_svg else 72)

    # Hidden outcome to gate decision.
    canvas.ribbon(x1 + bw, *L["mid_beneficial_blocked"], x2, *L["dec_blocked_beneficial"], GREEN, 0.48 if is_svg else 122)
    canvas.ribbon(x1 + bw, *L["mid_harmful_blocked"], x2, *L["dec_blocked_harmful"], RED, 0.48 if is_svg else 122)
    canvas.ribbon(x1 + bw, *L["mid_beneficial_pass"], x2, *L["dec_passed_beneficial"], GREEN, 0.66 if is_svg else 165)
    canvas.ribbon(x1 + bw, *L["mid_harmful_pass"], x2, *L["dec_passed_harmful"], RED, 0.66 if is_svg else 165)

    # Bars.
    canvas.rect(x0, L["top"], bw, L["total_h"], GRAY, stroke=BORDER if is_svg else None)
    y1, y2 = L["mid_beneficial"]
    canvas.rect(x1, y1, bw, y2 - y1, GREEN_LIGHT, stroke="white" if is_svg else None)
    y1, y2 = L["mid_harmful"]
    canvas.rect(x1, y1, bw, y2 - y1, RED_LIGHT, stroke="white" if is_svg else None)
    y1, y2 = L["dec_blocked"]
    canvas.rect(x2, y1, bw, y2 - y1, BLUE_LIGHT, stroke=BORDER if is_svg else None)
    y1, y2 = L["dec_passed"]
    canvas.rect(x2, y1, bw, y2 - y1, GRAY, stroke=BORDER if is_svg else None)

    # Segment markers on decision bar.
    y_bb = L["dec_blocked_beneficial"][1]
    y_bp = L["dec_passed_beneficial"][1]
    canvas.line(x2, y_bb, x2 + bw, y_bb, "white", 1.2)
    canvas.line(x2, y_bp, x2 + bw, y_bp, "white", 1.2)

    # Column labels.
    if is_svg:
        canvas.text(x0 + bw / 2, 72, "feedback-only", "small", "middle")
        canvas.text(x0 + bw / 2, 86, f"accepted {summary['accepted']}", "note", "middle")
        canvas.text(x1 + bw / 2, 72, "hidden outcome", "small", "middle")
        canvas.text(x2 + bw / 2, 72, "full gate", "small", "middle")
        canvas.text(x1 + bw + 8, L["mid_beneficial"][0] + 19, f"beneficial {summary['beneficial']}", "small")
        canvas.text(x1 + bw + 8, L["mid_harmful"][0] + 19, f"harmful {summary['harmful']}", "small")
        canvas.text(x2 + bw + 8, L["dec_blocked"][0] + 22, f"blocked {summary['blocked']}", "small")
        canvas.text(x2 + bw + 8, L["dec_passed"][0] + 22, f"passed {summary['passed']}", "small")
        canvas.text(70, 386, f"Full gate blocks {summary['harmful_blocked']}/{summary['harmful']} hidden-harmful accepts and passes {summary['beneficial_pass']}/{summary['beneficial']} hidden-beneficial accepts.", "note")
    else:
        canvas.text(x0 + bw / 2, 68, "feedback-only", 10, fill=INK, anchor="ma")
        canvas.text(x0 + bw / 2, 84, f"accepted {summary['accepted']}", 9, fill=MUTED, anchor="ma")
        canvas.text(x1 + bw / 2, 68, "hidden outcome", 10, fill=INK, anchor="ma")
        canvas.text(x2 + bw / 2, 68, "full gate", 10, fill=INK, anchor="ma")
        canvas.text(x1 + bw + 8, L["mid_beneficial"][0] + 14, f"beneficial {summary['beneficial']}", 10, fill=INK)
        canvas.text(x1 + bw + 8, L["mid_harmful"][0] + 14, f"harmful {summary['harmful']}", 10, fill=INK)
        canvas.text(x2 + bw + 8, L["dec_blocked"][0] + 17, f"blocked {summary['blocked']}", 10, fill=INK)
        canvas.text(x2 + bw + 8, L["dec_passed"][0] + 17, f"passed {summary['passed']}", 10, fill=INK)
        canvas.text(70, 386, f"Full gate blocks {summary['harmful_blocked']}/{summary['harmful']} hidden-harmful accepts and passes {summary['beneficial_pass']}/{summary['beneficial']} hidden-beneficial accepts.", 9, fill=MUTED)


def case_line(case):
    total = "yes" if case["total_agrees"] else "no"
    return [
        f"{float(case['observed_delta']):+.6f}",
        f"{float(case['p_after']):.3f}",
        total,
        f"{float(case['evaluation_delta']):+.6f}",
    ]


def draw_rule_table(canvas, cases, is_svg=False):
    x, y = 715, 43
    if is_svg:
        canvas.text(682, 43, "B", "panel")
        canvas.text(710, 43, "Gate rule and trace audit", "head")
        canvas.text(710, 79, "Accept candidate only if observed delta > 0 and", "small")
        canvas.rect(710, 94, 125, 28, BLUE_LIGHT, stroke=BORDER, radius=4)
        canvas.text(722, 112, "total agrees: p >= 0.42", "small")
        canvas.rect(846, 94, 145, 28, BLUE_LIGHT, stroke=BORDER, radius=4)
        canvas.text(858, 112, "total disagrees: p >= 0.48", "small")
        canvas.text(710, 158, "Trace", "note")
        canvas.text(875, 158, "obs delta", "note")
        canvas.text(955, 158, "p", "note")
        canvas.text(1005, 158, "total", "note")
        canvas.text(1065, 158, "hidden delta", "note")
        canvas.line(710, 168, 1140, 168, BORDER, 1)
    else:
        canvas.text(682, 29, "B", 20, True)
        canvas.text(710, 29, "Gate rule and trace audit", 15, True)
        canvas.text(710, 70, "Accept candidate only if observed delta > 0 and", 10, fill=INK)
        canvas.rect(710, 86, 125, 28, BLUE_LIGHT, outline=BORDER, radius=4)
        canvas.text(722, 105, "total agrees: p >= 0.42", 9, fill=INK)
        canvas.rect(846, 86, 145, 28, BLUE_LIGHT, outline=BORDER, radius=4)
        canvas.text(858, 105, "total disagrees: p >= 0.48", 9, fill=INK)
        canvas.text(710, 153, "Trace", 9, fill=MUTED)
        canvas.text(875, 153, "obs delta", 9, fill=MUTED)
        canvas.text(955, 153, "p", 9, fill=MUTED)
        canvas.text(1005, 153, "total", 9, fill=MUTED)
        canvas.text(1065, 153, "hidden delta", 9, fill=MUTED)
        canvas.line(710, 163, 1140, 163, BORDER, 1)

    rows = [
        ("Supported correction", cases["full_method_success"], GREEN, "accept"),
        ("Residual risk", cases["feedback_only_harmful_accept"], RED, "pass"),
        ("Noisy trap", cases["risk_gate_rejected_feedback_trap"], PURPLE, "reject"),
    ]
    y0 = 190
    for idx, (name, case, color, action) in enumerate(rows):
        yy = y0 + idx * 58
        vals = case_line(case)
        label = f"P{case['patient_id']} {case['column']} {case['before']}->{case['after']}"
        if is_svg:
            canvas.circle(716, yy - 4, 4, color, stroke=color)
            canvas.text(728, yy, name, "small")
            canvas.text(728, yy + 17, label, "note")
            canvas.text(875, yy, vals[0], "small")
            canvas.text(955, yy, vals[1], "small")
            canvas.text(1008, yy, vals[2], "small")
            canvas.text(1065, yy, vals[3], "small")
            canvas.text(1138, yy, action, "small", "end")
            canvas.line(710, yy + 28, 1140, yy + 28, GRID, 0.8)
        else:
            canvas.circle(716, yy - 6, 4, color, outline=color)
            canvas.text(728, yy - 12, name, 10, True)
            canvas.text(728, yy + 7, label, 9, fill=MUTED)
            canvas.text(875, yy - 12, vals[0], 9, fill=INK)
            canvas.text(955, yy - 12, vals[1], 9, fill=INK)
            canvas.text(1008, yy - 12, vals[2], 9, fill=INK)
            canvas.text(1065, yy - 12, vals[3], 9, fill=INK)
            canvas.text(1138, yy - 12, action, 9, True, fill=color, anchor="ra")
            canvas.line(710, yy + 28, 1140, yy + 28, GRID, 0.8)

    if is_svg:
        canvas.text(710, 406, "The audit exposes the intended behavior and the residual risk in one view.", "note")
    else:
        canvas.text(710, 406, "The audit exposes intended behavior and residual risk in one view.", 9, fill=MUTED)


class PngCanvas:
    def __init__(self):
        self.img = Image.new("RGBA", (W * SCALE, H * SCALE), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)

    def text(self, x, y, s, size=10, bold=False, fill=INK, anchor="la"):
        self.draw.text((int(x * SCALE), int(y * SCALE)), s, font=font(size, bold), fill=rgba(fill), anchor=anchor)

    def rect(self, x, y, w, h, fill, outline=None, width=1, radius=0, stroke=None):
        if stroke is not None:
            outline = stroke
        box = (int(x * SCALE), int(y * SCALE), int((x + w) * SCALE), int((y + h) * SCALE))
        if radius:
            self.draw.rounded_rectangle(box, radius=int(radius * SCALE), fill=rgba(fill), outline=rgba(outline) if outline else None, width=max(1, int(width * SCALE)))
        else:
            self.draw.rectangle(box, fill=rgba(fill), outline=rgba(outline) if outline else None, width=max(1, int(width * SCALE)))

    def line(self, x1, y1, x2, y2, fill=GRID, width=1):
        self.draw.line((int(x1 * SCALE), int(y1 * SCALE), int(x2 * SCALE), int(y2 * SCALE)), fill=rgba(fill), width=max(1, int(width * SCALE)))

    def circle(self, x, y, r, fill, outline="white"):
        box = (int((x - r) * SCALE), int((y - r) * SCALE), int((x + r) * SCALE), int((y + r) * SCALE))
        self.draw.ellipse(box, fill=rgba(fill), outline=rgba(outline), width=max(1, int(0.8 * SCALE)))

    def ribbon(self, x1, y1a, y1b, x2, y2a, y2b, fill, alpha=125):
        pts = [(int(x * SCALE), int(y * SCALE)) for x, y in ribbon_polygon(x1, y1a, y1b, x2, y2a, y2b)]
        self.draw.polygon(pts, fill=rgba(fill, alpha))

    def save(self, path):
        self.img.convert("RGB").save(path, dpi=(300, 300))


def make_svg(points, cases, summary):
    svg = Svg()
    draw_flow(svg, summary, is_svg=True)
    draw_rule_table(svg, cases, is_svg=True)
    svg.save(OUT / "paper_fig3_gate_interpretability.svg")


def make_png(points, cases, summary):
    canvas = PngCanvas()
    draw_flow(canvas, summary, is_svg=False)
    draw_rule_table(canvas, cases, is_svg=False)
    canvas.save(OUT / "paper_fig3_gate_interpretability.png")


def main():
    points = list(iter_feedback_only_accepted())
    cases = load_cases()
    summary = summarize_points(points)
    (OUT / "paper_fig3_gate_interpretability_summary.json").write_text(
        json.dumps({"summary": summary}, indent=2), encoding="utf-8"
    )
    make_svg(points, cases, summary)
    make_png(points, cases, summary)
    print(json.dumps({"figure_png": str(OUT / "paper_fig3_gate_interpretability.png"), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
