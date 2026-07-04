import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
VARIANTS = [
    ("relaxed", 0.40, 0.46),
    ("default", 0.42, 0.48),
    ("conservative", 0.44, 0.50),
]
SETTINGS = [
    ("clean", 20260526, 0.0, 0.0),
    ("noisy", 20260526, 0.0008, 0.05),
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_gate(original_gate, lower_threshold, strong_threshold):
    def gate(method, flip):
        if method == "full_risk_gate":
            return flip["p_after"] >= lower_threshold and (flip["total_agrees"] or flip["p_after"] >= strong_threshold)
        return original_gate(method, flip)

    return gate


def summarize(rows):
    out = []
    keys = sorted({(row["setting"], row["variant"]) for row in rows})
    for setting, variant in keys:
        subset = [row for row in rows if row["setting"] == setting and row["variant"] == variant]
        out.append(
            {
                "setting": setting,
                "variant": variant,
                "lower_threshold": subset[0]["lower_threshold"],
                "strong_threshold": subset[0]["strong_threshold"],
                "splits": len(subset),
                "score_mean": float(np.mean([r["score"] for r in subset])),
                "score_std": float(np.std([r["score"] for r in subset])),
                "gain_mean": float(np.mean([r["score_gain"] for r in subset])),
                "gain_std": float(np.std([r["score_gain"] for r in subset])),
                "item_accuracy_mean": float(np.mean([r["item_accuracy"] for r in subset])),
                "total_rmse_mean": float(np.mean([r["total_rmse"] for r in subset])),
                "changed_labels_mean": float(np.mean([r["changed_labels"] for r in subset])),
                "accepted_updates_mean": float(np.mean([r["accepted_updates"] for r in subset])),
                "harmful_accepted_mean": float(np.mean([r["harmful_accepted_updates"] for r in subset])),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=24)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--out-prefix", default="offline_threshold_sensitivity_v1")
    args = parser.parse_args()

    base_ids, y = sim.load_track1()
    original_gate = sim.gate_allows
    all_rows = []
    split_reports = []
    try:
        for setting, seed_base, feedback_noise, flip_feedback_prob in SETTINGS:
            for variant, lower, strong in VARIANTS:
                ids = base_ids.copy()
                sim.gate_allows = make_gate(original_gate, lower, strong)
                for i in range(args.splits):
                    seed = seed_base + i * 17
                    report = sim.run_split(
                        ids,
                        y,
                        seed,
                        args.test_fraction,
                        [args.budget],
                        feedback_noise=feedback_noise,
                        flip_feedback_prob=flip_feedback_prob,
                    )
                    split_reports.append(
                        {
                            "setting": setting,
                            "variant": variant,
                            "seed": seed,
                            "report": report,
                        }
                    )
                    row = next(
                        item
                        for item in report["rows"]
                        if item["method"] == "full_risk_gate" and int(item["budget"]) == args.budget
                    )
                    row = dict(row)
                    row.update(
                        {
                            "setting": setting,
                            "variant": variant,
                            "lower_threshold": lower,
                            "strong_threshold": strong,
                        }
                    )
                    all_rows.append(row)
                    print(f"completed {setting} {variant} split {i + 1}/{args.splits}", flush=True)
    finally:
        sim.gate_allows = original_gate

    summary = summarize(all_rows)
    row_csv = OUT_DIR / f"{args.out_prefix}_rows.csv"
    summary_csv = OUT_DIR / f"{args.out_prefix}_summary.csv"
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    write_csv(row_csv, all_rows)
    write_csv(summary_csv, summary)
    json_path.write_text(
        json.dumps({"config": vars(args), "variants": VARIANTS, "settings": SETTINGS, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Full Gate Threshold Sensitivity",
        "",
        f"- Splits: `{args.splits}`",
        f"- Budget: `{args.budget}`",
        "",
        "| setting | variant | thresholds | score | gain | item acc | total rmse | changed | harmful |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"clean": 0, "noisy": 1}
    for row in sorted(summary, key=lambda r: (order.get(r["setting"], 99), r["variant"])):
        lines.append(
            f"| {row['setting']} | {row['variant']} | {row['lower_threshold']:.2f}/{row['strong_threshold']:.2f} | "
            f"{row['score_mean']:.5f} +/- {row['score_std']:.5f} | {row['gain_mean']:+.5f} | "
            f"{row['item_accuracy_mean']:.5f} | {row['total_rmse_mean']:.3f} | "
            f"{row['changed_labels_mean']:.2f} | {row['harmful_accepted_mean']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
