"""Exact-match agreement and Cohen's kappa across runs, using few-shot and zero-shot files.

Expected zero-based column indices in both files:
    1 = example ID
    2 = run (0, 1, 2)
    3 = k (ignored for the zero-shot file; forced to 0)
   10 = raw model output

Example:
    python run_agreement_two_files.py fewshot.xlsx zeroshot.xlsx
    python run_agreement_two_files.py fewshot.xlsx zeroshot.csv

Exports:
    agreement_summary.csv
    agreement_by_item.csv
    excluded_items.csv
    diagnostics_by_k.csv
    dimension_kappa_by_k.csv
"""

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ID_COL = 1
RUN_COL = 2
K_COL = 3
RESPONSE_COL = 10

RUNS = ("0", "1", "2")
COMPARISONS = {
    "0_vs_1": ("0", "1"),
    "0_vs_2": ("0", "2"),
    "1_vs_2": ("1", "2"),
    "all_3": RUNS,
}
METRICS = ("boolean", "dimensions", "dimensions_positive", "combined")


def clean_key(value):
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_run(value):
    label = clean_key(value).casefold()
    match = re.fullmatch(r"(?:run[\s_:\-]*)?([012])(?:st|nd|rd)?(?:[\s_:\-]*run)?", label)
    return match.group(1) if match else ""


def extract_types_list(text, start):
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "[":
        raise ValueError("types does not start with '['")

    depth = 0
    quote = None
    escaped = False
    for j in range(i, len(text)):
        char = text[j]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    raise ValueError("Unclosed types list")


def normalize_dimension_name(value):
    """Canonicalize spelling variants of the 'overt behaviour' dimension."""
    name = re.sub(r"\s+", " ", value).strip().casefold()
    # All these refer to the same dimension, regardless of capitalization:
    # behaviour, behavior, (overt) behaviour, overt behaviour.
    # 'bevaviour' is also accepted as a common typo.
    if re.fullmatch(r"(?:\(?overt\)?\s+)?(?:behaviou?r|bevaviour)", name):
        return "overt behaviour"
    return name


def parse_response(value):
    """Read is_process and types directly, ignoring malformed utterance text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Empty model response")

    boolean_matches = list(re.finditer(
        r'''(?<!\w)["']?is_process["']?\s*:\s*["']?(true|false|1|0)["']?(?!\w)''',
        value, flags=re.IGNORECASE,
    ))
    types_matches = list(re.finditer(
        r'''(?<!\w)["']?types["']?\s*:''',
        value, flags=re.IGNORECASE,
    ))
    if not boolean_matches:
        raise ValueError("Cannot extract is_process")
    if not types_matches:
        raise ValueError("Cannot extract types")

    is_process = boolean_matches[-1].group(1).casefold() in ("true", "1")
    types_text = extract_types_list(value, types_matches[-1].end())

    try:
        types = json.loads(types_text)
    except json.JSONDecodeError:
        try:
            types = ast.literal_eval(types_text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Cannot parse types list: {types_text}") from exc

    if not isinstance(types, (list, tuple)) or not all(isinstance(x, str) for x in types):
        raise ValueError(f"types must be a list of strings: {types_text}")

    normalized_types = frozenset(normalize_dimension_name(x) for x in types if x.strip())
    return (is_process, normalized_types)


def load_table(path, sheet):
    """Detect actual file content, not just its extension.

    Supports .xlsx, legacy .xls (requires xlrd), and CSV text even if
    that CSV has mistakenly been named with an .xlsx extension.
    """
    path = Path(path)
    with path.open("rb") as stream:
        signature = stream.read(8)

    if signature.startswith(b"PK\x03\x04"):
        print(f"Reading Excel workbook: {path}")
        try:
            return pd.read_excel(
                path, sheet_name=sheet, header=None,
                dtype=object, engine="openpyxl",
            )
        except Exception as exc:
            raise ValueError(f"Cannot read Excel workbook {path}: {exc}") from exc

    if signature.startswith(b"\xd0\xcf\x11\xe0"):
        print(f"Reading legacy .xls workbook: {path}")
        try:
            return pd.read_excel(
                path, sheet_name=sheet, header=None,
                dtype=object, engine="xlrd",
            )
        except ImportError as exc:
            raise RuntimeError("Install xlrd to read .xls: pip install xlrd") from exc

    if not signature:
        raise ValueError(f"The file is empty: {path}")

    # A file named .xlsx may actually contain CSV text.
    print(f"Reading text/CSV file: {path}")
    last_error = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            data = pd.read_csv(
                path, sep=None, engine="python", header=None,
                dtype=object, encoding=encoding,
            )
            return data
        except (UnicodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(
        f"Unsupported or unreadable format for {path}. "
        f"Use a real .xlsx workbook or a CSV file. Last error: {last_error}"
    )


def read_workbook(path, sheet, force_k_zero, groups, exclusions):
    df = load_table(path, sheet)
    run_counts = Counter()
    if df.shape[1] <= RESPONSE_COL:
        raise ValueError(f"{path}: expected at least 11 columns (output in Excel column K)")

    for index, row in df.iterrows():
        excel_row = index + 1
        example_id = clean_key(row.iloc[ID_COL])
        original_k = clean_key(row.iloc[K_COL])
        raw_run = clean_key(row.iloc[RUN_COL])
        response = row.iloc[RESPONSE_COL]

        # Skip a header row if it exists inside the sheet.
        if (example_id.casefold() in {"id", "example_id", "example id"}
                and raw_run.casefold() in {"run", "run_id", "run number"}):
            continue
        if not example_id and not raw_run and not original_k and pd.isna(response):
            continue

        k = "0" if force_k_zero else original_k
        if not example_id or not k:
            exclusions.append({"source_file": path.name, "excel_row": excel_row,
                               "id": example_id, "k": k, "run": raw_run,
                               "reason": "missing_id_or_k", "details": "Cannot match this record"})
            continue

        run = normalize_run(raw_run)
        if not run:
            exclusions.append({"source_file": path.name, "excel_row": excel_row,
                               "id": example_id, "k": k, "run": raw_run,
                               "reason": "invalid_run", "details": "Expected run 0, 1 or 2"})
            continue

        run_counts[run] += 1
        group = groups.setdefault((example_id, k), {r: [] for r in RUNS})
        entry = {"source_file": path.name, "excel_row": excel_row, "prediction": None}
        try:
            entry["prediction"] = parse_response(response)
        except (ValueError, TypeError) as exc:
            exclusions.append({"source_file": path.name, "excel_row": excel_row,
                               "id": example_id, "k": k, "run": run,
                               "reason": "parse_error", "details": str(exc)})
        group[run].append(entry)

    print(
        f"Rows by run in {path.name}: "
        + ", ".join(f"run {r}={run_counts[r]}" for r in RUNS)
    )
    if force_k_zero and sum(run_counts.values()) and sum(run_counts[r] > 0 for r in RUNS) < 3:
        print(
            "WARNING: The zero-shot input does not contain all three run labels (0, 1, 2). "
            "Agreement across all three runs requires three recorded predictions "
            "for each ID; the script will not invent missing runs."
        )


def make_agreement(predictions):
    bools = [p[0] for p in predictions]
    dimensions = [p[1] for p in predictions]
    boolean_agree = int(all(x == bools[0] for x in bools[1:]))
    dimension_agree = int(all(x == dimensions[0] for x in dimensions[1:]))
    return {
        "boolean": boolean_agree,
        "dimensions": dimension_agree,
        "dimensions_positive": dimension_agree if all(bools) else None,
        "combined": int(boolean_agree and dimension_agree),
    }


def evaluate(groups, exclusions):
    items = []
    for (example_id, k), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        item = {"id": example_id, "k": k}
        valid = {}
        for run, entries in group.items():
            if not entries:
                status = "missing"
                exclusions.append({"source_file": "", "excel_row": "", "id": example_id,
                                   "k": k, "run": run, "reason": "missing_run", "details": "No record"})
            elif len(entries) > 1:
                status = "duplicate"
                exclusions.append({"source_file": ", ".join(sorted({e["source_file"] for e in entries})),
                                   "excel_row": ", ".join(str(e["excel_row"]) for e in entries),
                                   "id": example_id, "k": k, "run": run,
                                   "reason": "duplicate_run", "details": "All duplicates excluded"})
            elif entries[0]["prediction"] is None:
                status = "parse_error"
            else:
                status = "valid"
                valid[run] = entries[0]["prediction"]
            item[f"run_{run}_status"] = status
            item[f"run_{run}_is_process"] = valid[run][0] if run in valid else None
            item[f"run_{run}_types"] = (
                json.dumps(sorted(valid[run][1]), ensure_ascii=False) if run in valid else ""
            )

        item["n_valid_runs"] = len(valid)
        for comparison, required_runs in COMPARISONS.items():
            agreements = (make_agreement([valid[r] for r in required_runs])
                          if all(r in valid for r in required_runs) else {})
            for metric in METRICS:
                item[f"{metric}_{comparison}"] = agreements.get(metric)
        items.append(item)
    return items


def summarize(items):
    ks = sorted({item["k"] for item in items},
                key=lambda value: (0, int(value)) if re.fullmatch(r"-?\d+", value)
                else (1, value))
    records = []
    for k in [*ks, "OVERALL"]:
        subset = items if k == "OVERALL" else [item for item in items if item["k"] == k]
        for comparison in COMPARISONS:
            compared = [item for item in subset if item[f"boolean_{comparison}"] is not None]
            positive = [item for item in compared if item[f"dimensions_positive_{comparison}"] is not None]
            record = {
                "k": k,
                "comparison": comparison,
                "n_items_compared": len(compared),
                "n_positive_dimension_items": len(positive),
            }
            for metric in METRICS:
                eligible = positive if metric == "dimensions_positive" else compared
                record[f"{metric}_agreement_pct"] = (
                    round(100 * sum(item[f"{metric}_{comparison}"] for item in eligible) / len(eligible), 2)
                    if eligible else None
                )
            records.append(record)
    return records


def make_diagnostics(items):
    """Explain missing agreement values without treating them as 0% agreement."""
    records = []
    for k in sorted({item["k"] for item in items},
                    key=lambda v: (0, int(v)) if re.fullmatch(r"-?\d+", v) else (1, v)):
        subset = [item for item in items if item["k"] == k]
        record = {"k": k, "n_unique_ids": len(subset)}
        for run in RUNS:
            for status in ("valid", "missing", "duplicate", "parse_error"):
                record[f"run_{run}_{status}"] = sum(
                    item[f"run_{run}_status"] == status for item in subset
                )
        for label in COMPARISONS:
            record[f"n_valid_{label}"] = sum(
                item[f"boolean_{label}"] is not None for item in subset
            )
        issues = []
        if record["n_valid_all_3"] == 0:
            if any(record[f"run_{r}_valid"] == 0 for r in RUNS):
                issues.append("At least one run has no valid predictions")
            if any(record[f"run_{r}_parse_error"] for r in RUNS):
                issues.append("Unparseable model responses; check output column K")
            if any(record[f"run_{r}_duplicate"] for r in RUNS):
                issues.append("Duplicate (ID, k, run) records")
            if any(record[f"run_{r}_missing"] for r in RUNS):
                issues.append("Runs do not share the same IDs or runs are missing")
        record["diagnosis"] = "; ".join(issues) if issues else "Valid comparisons available"
        records.append(record)
    return records


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for two paired binary label sequences.

    Returns None when there are no examples or when expected agreement is 1.
    This happens, for example, if neither run ever predicts a dimension.
    """
    n = len(labels_a)
    if n == 0:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    p_a = sum(labels_a) / n
    p_b = sum(labels_b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if abs(1 - expected) < 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def available_for_comparison(item, required_runs):
    return all(item[f"run_{r}_status"] == "valid" for r in required_runs)


def kappa_for_runs(rows, run_a, run_b, dimension=None):
    """Compute kappa from the SAME matched examples in two runs."""
    if dimension is None:
        labels_a = [bool(row[f"run_{run_a}_is_process"]) for row in rows]
        labels_b = [bool(row[f"run_{run_b}_is_process"]) for row in rows]
    else:
        labels_a = [dimension in json.loads(row[f"run_{run_a}_types"]) for row in rows]
        labels_b = [dimension in json.loads(row[f"run_{run_b}_types"]) for row in rows]
    return cohen_kappa(labels_a, labels_b)


def add_boolean_kappa(summary, items):
    """Add pairwise Boolean kappa and a 3-pair mean for all_3.

    The all_3 value is NOT itself a Cohen's kappa coefficient; it is
    the mean of the three pairwise kappas on common triple-valid IDs.
    """
    for record in summary:
        k = record["k"]
        comparison = record["comparison"]
        subset = items if k == "OVERALL" else [row for row in items if row["k"] == k]
        required = COMPARISONS[comparison]
        rows = [row for row in subset if available_for_comparison(row, required)]

        record["boolean_cohens_kappa"] = None
        record["boolean_mean_pairwise_kappa"] = None
        if comparison == "all_3":
            kappas = [kappa_for_runs(rows, a, b) for a, b in
                      (("0", "1"), ("0", "2"), ("1", "2"))]
            if all(value is not None for value in kappas):
                record["boolean_mean_pairwise_kappa"] = round(sum(kappas) / 3, 4)
        else:
            value = kappa_for_runs(rows, *required)
            record["boolean_cohens_kappa"] = round(value, 4) if value is not None else None


def make_dimension_kappa(items):
    """Calculate a binary Cohen's kappa per predicted dimension and k.

    Dimension order is ignored; all valid matched examples are included,
    including items where is_process=False (empty dimensions = absent).
    """
    dimensions = sorted({
        dimension
        for row in items
        if row["k"] != "0"
        for run in RUNS
        if row[f"run_{run}_status"] == "valid"
        for dimension in json.loads(row[f"run_{run}_types"])
    })
    # Zero-shot has only one run, so it cannot produce Cohen's kappa.
    # Omit k=0 entirely from the per-dimension Kappa CSV.
    ks = sorted({row["k"] for row in items if row["k"] != "0"},
                key=lambda v: (0, int(v)) if re.fullmatch(r"-?\d+", v) else (1, v))
    records = []
    for k in [*ks, "OVERALL"]:
        # Kappa results cover only few-shot data, including the OVERALL row.
        subset = ([row for row in items if row["k"] != "0"] if k == "OVERALL"
                  else [row for row in items if row["k"] == k])
        for dimension in dimensions:
            for comparison, required_runs in COMPARISONS.items():
                rows = [row for row in subset if available_for_comparison(row, required_runs)]
                record = {
                    "k": k, "dimension": dimension, "comparison": comparison,
                    "n_items_compared": len(rows),
                }
                for run in RUNS:
                    record[f"n_positive_run_{run}"] = (
                        sum(dimension in json.loads(row[f"run_{run}_types"]) for row in rows)
                        if run in required_runs else None
                    )
                record["cohens_kappa"] = None
                record["mean_pairwise_kappa"] = None
                record["n_defined_pairs"] = None
                if comparison == "all_3":
                    kappas = [kappa_for_runs(rows, a, b, dimension) for a, b in
                              (("0", "1"), ("0", "2"), ("1", "2"))]
                    record["n_defined_pairs"] = sum(value is not None for value in kappas)
                    if all(value is not None for value in kappas):
                        record["mean_pairwise_kappa"] = round(sum(kappas) / 3, 4)
                else:
                    value = kappa_for_runs(rows, *required_runs, dimension=dimension)
                    if value is not None:
                        record["cohens_kappa"] = round(value, 4)
                records.append(record)
    return records



def make_wide_dimension_kappa(records):
    """One row per (k, dimension), with run counts and kappas side by side.

    The run counts are computed on the all-three-valid subset. Pairwise
    kappas retain their original pairwise-valid comparison populations.
    An undefined kappa is written as 'undefined', never as 0.
    """
    wide = {}
    for record in records:
        key = (record["k"], record["dimension"])
        row = wide.setdefault(key, {
            "k": record["k"],
            "dimension": record["dimension"],
            "n_valid_all_3": 0,
            "predicted_run_0": None,
            "predicted_run_1": None,
            "predicted_run_2": None,
            "kappa_0_vs_1": None,
            "kappa_0_vs_2": None,
            "kappa_1_vs_2": None,
            "mean_pairwise_kappa": None,
        })
        comparison = record["comparison"]
        if comparison == "all_3":
            row["n_valid_all_3"] = record["n_items_compared"]
            for run in RUNS:
                row[f"predicted_run_{run}"] = record[f"n_positive_run_{run}"]
            row["mean_pairwise_kappa"] = record["mean_pairwise_kappa"]
        else:
            row[f"kappa_{comparison}"] = record["cohens_kappa"]

    output = sorted(
        wide.values(),
        key=lambda row: (
            (1, 0) if row["k"] == "OVERALL" else
            (0, int(row["k"])) if re.fullmatch(r"-?\d+", row["k"]) else
            (0, row["k"]),
            row["dimension"],
        ),
    )
    for row in output:
        for column in (
            "kappa_0_vs_1", "kappa_0_vs_2", "kappa_1_vs_2",
            "mean_pairwise_kappa",
        ):
            if row[column] is None:
                row[column] = "undefined"

    columns = [
        "k", "dimension", "n_valid_all_3",
        "predicted_run_0", "predicted_run_1", "predicted_run_2",
        "kappa_0_vs_1", "kappa_0_vs_2", "kappa_1_vs_2",
        "mean_pairwise_kappa",
    ]
    return pd.DataFrame(output, columns=columns)


def main():
    parser = argparse.ArgumentParser(description="Exact-match agreement and per-dimension Cohen's kappa")
    parser.add_argument("fewshot_file", type=Path, help="Excel/CSV containing k=1,5,10,...")
    parser.add_argument("zeroshot_file", type=Path, help="Separate Excel/CSV for k=0")
    parser.add_argument("--fewshot-sheet", default="0", help="Sheet index or name (default 0)")
    parser.add_argument("--zeroshot-sheet", default="0", help="Sheet index or name (default 0)")
    parser.add_argument("--output", type=Path, default=Path("robustness_results"))
    args = parser.parse_args()

    for path in (args.fewshot_file, args.zeroshot_file):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    if args.fewshot_file.resolve() == args.zeroshot_file.resolve():
        parser.error(
            "The same file was supplied twice. "
            "The first argument must be the FEW-SHOT workbook and "
            "the second must be the separate ZERO-SHOT workbook."
        )

    def sheet_arg(value):
        return int(value) if value.isdecimal() else value

    groups = {}
    exclusions = []
    read_workbook(args.fewshot_file, sheet_arg(args.fewshot_sheet), False, groups, exclusions)
    read_workbook(args.zeroshot_file, sheet_arg(args.zeroshot_sheet), True, groups, exclusions)
    if not groups:
        raise SystemExit("No valid (ID, k) groups found in either workbook")

    items = evaluate(groups, exclusions)
    # List unique normalized dimensions across all valid responses, both files.
    available_dimensions = sorted({
        dim
        for item in items
        for run in RUNS
        if item[f"run_{run}_status"] == "valid"
        for dim in json.loads(item[f"run_{run}_types"])
    })
    print(f"\nAVAILABLE DIMENSIONS ({len(available_dimensions)}):")
    if available_dimensions:
        for dimension in available_dimensions:
            print(f"  - {dimension}")
    else:
        print("  None found in valid model responses")

    summary = summarize(items)
    add_boolean_kappa(summary, items)
    dimension_kappa = make_dimension_kappa(items)
    # Macro-average: give each dimension equal weight; omit undefined kappas.
    for record in summary:
        matching = [row for row in dimension_kappa
                    if row["k"] == record["k"]
                    and row["comparison"] == record["comparison"]]
        metric = ("mean_pairwise_kappa" if record["comparison"] == "all_3"
                  else "cohens_kappa")
        defined = [row[metric] for row in matching if row[metric] is not None]
        record["dimension_macro_kappa"] = (
            round(sum(defined) / len(defined), 4) if defined else None
        )
        record["n_dimensions_with_defined_kappa"] = len(defined)
    diagnostics = make_diagnostics(items)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output / "agreement_summary.csv", index=False)
    make_wide_dimension_kappa(dimension_kappa).to_csv(
        args.output / "dimension_kappa_by_k.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(args.output / "diagnostics_by_k.csv", index=False)
    item_frame = pd.DataFrame(items)
    for comparison in COMPARISONS:
        for metric in METRICS:
            column = f"{metric}_{comparison}"
            item_frame[column] = item_frame[column].astype("Int64")
    item_frame.to_csv(args.output / "agreement_by_item.csv", index=False)
    pd.DataFrame(exclusions, columns=["source_file", "excel_row", "id", "k", "run", "reason", "details"]).to_csv(
        args.output / "excluded_items.csv", index=False
    )

    print("\nINPUT DIAGNOSTICS BY k")
    diagnostics_frame = pd.DataFrame(diagnostics)
    display_columns = [
        "k", "n_unique_ids", "run_0_valid", "run_1_valid", "run_2_valid",
        "n_valid_0_vs_1", "n_valid_0_vs_2", "n_valid_1_vs_2", "n_valid_all_3", "diagnosis",
    ]
    print(diagnostics_frame[display_columns].to_string(index=False))
    zero_diagnostics = next((r for r in diagnostics if r["k"] == "0"), None)
    if zero_diagnostics is not None and zero_diagnostics["n_valid_all_3"] == 0:
        print("\nZERO-SHOT WARNING: None of the k=0 examples has all three valid runs.")
        print("Inspect diagnostics_by_k.csv and excluded_items.csv (filter k=0).")
        print("A missing final '}' in a model response is NOT a parsing error if ")
        print("'is_process' and the complete 'types' list are present.")
    print("\nOVERALL AGREEMENT (%)")
    print(pd.DataFrame(summary).query('k == "OVERALL"').to_string(index=False))
    print(f"\nSaved to: {args.output.resolve()}")
    print("  agreement_summary.csv\n  agreement_by_item.csv\n  excluded_items.csv\n  diagnostics_by_k.csv\n  dimension_kappa_by_k.csv")


if __name__ == "__main__":
    main()
