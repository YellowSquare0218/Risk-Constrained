import json
from pathlib import Path


OUT_DIR = Path("outputs")
SOURCE_FILES = [
    OUT_DIR / "offline_feedback_simulation_v5_samesplit_24splits.json",
    OUT_DIR / "offline_feedback_simulation_v5_samesplit_24splits_noisy.json",
    OUT_DIR / "offline_feedback_simulation_v6_mainsplit_feedback50_full.json",
    OUT_DIR / "offline_feedback_simulation_v6_mainsplit_feedback50_full_noisy.json",
]
OUT_JSON = OUT_DIR / "offline_feedback_case_studies_samesplit_2026-06-08.json"
OUT_MD = OUT_DIR / "offline_feedback_case_studies_samesplit_2026-06-08.md"


def full_gate_would_allow(row):
    return float(row["p_after"]) >= 0.42 and (bool(row["total_agrees"]) or float(row["p_after"]) >= 0.48)


def iter_trace_rows(report, source_name, trace_name):
    for split in report["splits"]:
        test_ids = split.get("test_ids_ordered", split.get("test_ids", []))
        for row in split["traces"].get(trace_name, []):
            item = dict(row)
            row_idx = int(item["row_idx"])
            item["source"] = source_name
            item["seed"] = int(split["seed"])
            item["trace"] = trace_name
            item["patient_id"] = int(test_ids[row_idx]) if row_idx < len(test_ids) else None
            item["evaluation_delta"] = float(item.get("evaluation_delta", item.get("aggregate_delta", 0.0)))
            item["aggregate_delta"] = float(item.get("aggregate_delta", 0.0))
            item["observed_delta"] = float(item.get("observed_delta", 0.0))
            item["p_after"] = float(item.get("p_after", 0.0))
            yield item


def load_cases():
    rows = []
    for path in SOURCE_FILES:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        source_name = path.stem
        rows.extend(iter_trace_rows(report, source_name, "full_risk_gate_budget20"))
        rows.extend(iter_trace_rows(report, source_name, "feedback_only_budget20"))
    return rows


def select_cases(rows):
    full_success = [
        r
        for r in rows
        if r["trace"] == "full_risk_gate_budget20" and r.get("accepted") and r["evaluation_delta"] > 0
    ]
    feedback_harm = [
        r
        for r in rows
        if r["trace"] == "feedback_only_budget20" and r.get("accepted") and r["evaluation_delta"] <= 0
    ]
    rejected_by_full = [r for r in feedback_harm if not full_gate_would_allow(r)]
    full_success.sort(key=lambda r: (r["evaluation_delta"], r["p_after"]), reverse=True)
    feedback_harm.sort(key=lambda r: (r["evaluation_delta"], -r["p_after"]))
    rejected_by_full.sort(key=lambda r: (r["evaluation_delta"], -r["p_after"]))
    return {
        "full_method_success": full_success[0] if full_success else None,
        "feedback_only_harmful_accept": feedback_harm[0] if feedback_harm else None,
        "risk_gate_rejected_feedback_trap": rejected_by_full[0] if rejected_by_full else None,
        "counts": {
            "full_success_candidates": len(full_success),
            "feedback_harmful_accepts": len(feedback_harm),
            "harmful_accepts_rejected_by_full_gate": len(rejected_by_full),
        },
    }


def describe_case(row):
    if row is None:
        return "- Not found."
    return (
        f"- Source: `{row['source']}`, seed `{row['seed']}`, patient `{row['patient_id']}`, "
        f"`{row['column']}: {row['before']} -> {row['after']}`; "
        f"p_after `{row['p_after']:.4f}`, total_agrees `{row['total_agrees']}`, "
        f"feedback delta `{row['aggregate_delta']:+.6f}`, observed delta `{row['observed_delta']:+.6f}`, "
        f"evaluation delta `{row['evaluation_delta']:+.6f}`."
    )


def main():
    rows = load_cases()
    cases = select_cases(rows)
    OUT_JSON.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    lines = [
        "# Offline Feedback Case Studies",
        "",
        "These cases are selected automatically from the 24-split clean/noisy and Feedback-50/full simulation traces.",
        "",
        "## Counts",
        "",
        f"- Full-method accepted positive updates: `{cases['counts']['full_success_candidates']}`",
        f"- Feedback-only harmful accepted updates: `{cases['counts']['feedback_harmful_accepts']}`",
        f"- Harmful feedback-only updates rejected by the full gate: `{cases['counts']['harmful_accepts_rejected_by_full_gate']}`",
        "",
        "## Full Method Successful Update",
        "",
        describe_case(cases["full_method_success"]),
        "",
        "## Feedback-Only Harmful Accepted Update",
        "",
        describe_case(cases["feedback_only_harmful_accept"]),
        "",
        "## Risk-Gate Rejected Feedback Trap",
        "",
        describe_case(cases["risk_gate_rejected_feedback_trap"]),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "md": str(OUT_MD), "counts": cases["counts"]}, indent=2))


if __name__ == "__main__":
    main()
