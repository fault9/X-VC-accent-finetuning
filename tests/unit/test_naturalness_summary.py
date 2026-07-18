import csv

from scripts.summarize_naturalness_eval import summarize


def test_summary_reports_per_target_deltas(tmp_path):
    metrics = tmp_path / "metrics.csv"
    rows = [
        {"step": "base", "target": "voice_a", "mos_pred": "3.0", "wer": "0.02", "sim_cosine": "0.70"},
        {"step": "base", "target": "voice_b", "mos_pred": "3.2", "wer": "0.01", "sim_cosine": "0.72"},
        {"step": "50", "target": "voice_a", "mos_pred": "3.2", "wer": "0.03", "sim_cosine": "0.71"},
        {"step": "50", "target": "voice_b", "mos_pred": "3.3", "wer": "0.01", "sim_cosine": "0.74"},
    ]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "summary.csv"
    result = summarize(metrics, output)
    voice_a = next(row for row in result if row["step"] == "50" and row["target"] == "voice_a")
    overall = next(row for row in result if row["step"] == "50" and row["target"] == "__overall__")
    assert voice_a["mos_delta_vs_base"] == "0.2000"
    assert overall["mos_delta_vs_base"] == "0.1500"
    assert output.is_file()
