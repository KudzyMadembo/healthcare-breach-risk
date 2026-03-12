"""Inspect CICIoMT2024 datasets for temporal sequencing suitability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INTERIM_RELATIVE_PATH = Path("data/interim/train_dataset.csv")
PROCESSED_RELATIVE_PATH = Path("data/processed/train_processed.csv")
REPORT_RELATIVE_PATH = Path("reports/sequence_inspection.txt")
TEMPORAL_KEYWORDS = ("time", "timestamp", "flow_duration", "iat", "packet", "seq")


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with pandas."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def find_temporal_candidate_columns(df: pd.DataFrame) -> list[str]:
    """Return columns whose names suggest temporal or ordering information."""
    candidates: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in TEMPORAL_KEYWORDS):
            candidates.append(col)
    return candidates


def format_candidate_details(df: pd.DataFrame, columns: list[str], sample_rows: int = 5) -> list[str]:
    """Create text lines with dtype and sample values for candidate columns."""
    lines: list[str] = []
    if not columns:
        lines.append("No temporal candidate columns found.")
        return lines

    for col in columns:
        dtype = str(df[col].dtype)
        samples = df[col].dropna().head(sample_rows).tolist()
        lines.append(f"- {col} | dtype={dtype} | sample_values={samples}")
    return lines


def format_source_file_counts(df: pd.DataFrame, top_n: int = 20) -> list[str]:
    """Return source_file counts as text lines."""
    if "source_file" not in df.columns:
        return ["source_file column not found."]

    counts = df["source_file"].value_counts(dropna=False)
    lines = [f"Unique source_file values: {counts.shape[0]}"]
    lines.append(f"Top {min(top_n, counts.shape[0])} source_file counts:")
    for source_file, count in counts.head(top_n).items():
        lines.append(f"- {source_file}: {int(count)}")
    return lines


def assess_temporal_suitability(interim_df: pd.DataFrame, processed_df: pd.DataFrame) -> str:
    """Provide a practical recommendation for temporal sequence modeling."""
    interim_candidates = find_temporal_candidate_columns(interim_df)
    processed_candidates = find_temporal_candidate_columns(processed_df)
    has_source_file = "source_file" in interim_df.columns
    has_ordering_signal = bool(interim_candidates)

    if has_source_file and has_ordering_signal:
        return (
            "Potentially suitable for temporal grouping: source_file exists and temporal-like "
            "columns are present. For true temporal modeling, group by source_file and sort "
            "within each group using a reliable timestamp/order column."
        )
    if has_source_file and not has_ordering_signal:
        return (
            "Partially suitable: source_file exists, but no strong temporal/order columns were "
            "detected by name. Sequence windows may be pseudo-temporal unless a true ordering "
            "field is identified."
        )
    if not has_source_file and has_ordering_signal:
        return (
            "Partially suitable: temporal-like columns exist, but source_file grouping key is "
            "missing in the interim dataset. Consider deriving session/flow grouping keys."
        )
    return (
        "Limited suitability for true temporal modeling: no clear source grouping key or "
        "temporal/order columns detected by naming heuristics."
    )


def build_report(interim_df: pd.DataFrame, processed_df: pd.DataFrame) -> str:
    """Build complete inspection report as plain text."""
    interim_candidates = find_temporal_candidate_columns(interim_df)
    processed_candidates = find_temporal_candidate_columns(processed_df)

    lines: list[str] = []
    lines.append("CICIoMT2024 Sequence Inspection")
    lines.append("=" * 32)
    lines.append("")
    lines.append(f"Interim dataset shape: {interim_df.shape}")
    lines.append(f"Processed dataset shape: {processed_df.shape}")
    lines.append("")
    lines.append("Interim columns:")
    lines.extend([f"- {col}" for col in interim_df.columns.tolist()])
    lines.append("")
    lines.append("Processed columns:")
    lines.extend([f"- {col}" for col in processed_df.columns.tolist()])
    lines.append("")
    lines.append("Temporal candidate columns in interim dataset:")
    lines.extend(format_candidate_details(interim_df, interim_candidates))
    lines.append("")
    lines.append("Temporal candidate columns in processed dataset:")
    lines.extend(format_candidate_details(processed_df, processed_candidates))
    lines.append("")
    lines.append("Counts by source_file (interim):")
    lines.extend(format_source_file_counts(interim_df))
    lines.append("")
    lines.append("Temporal sequencing suitability assessment:")
    lines.append(assess_temporal_suitability(interim_df, processed_df))
    lines.append("")
    return "\n".join(lines)


def run_inspection(base_path: str) -> str:
    """Run inspection and save summary report."""
    root = Path(base_path).resolve()
    interim_path = root / INTERIM_RELATIVE_PATH
    processed_path = root / PROCESSED_RELATIVE_PATH
    report_path = root / REPORT_RELATIVE_PATH

    interim_df = load_csv(interim_path)
    processed_df = load_csv(processed_path)
    report_text = build_report(interim_df, processed_df)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_text


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = run_inspection(str(project_root))
    print(output)
    print(f"Saved inspection report to: {project_root / REPORT_RELATIVE_PATH}")
