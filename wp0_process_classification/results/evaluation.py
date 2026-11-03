
import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


# ==================================================
# CONFIGURATION
# ==================================================

# Zero-based column indices
ID_COL = 1
RUN_COL = 2
K_COL = 3
GT_BOOL_COL = 5
GT_DIMS_COL = 6
PRED_COL = 10

RUNS = ("0", "1", "2")


# ==================================================
# NORMALIZATION
# ==================================================

def clean(value):
    if value is None or (
        isinstance(value, (int, float, np.integer, np.floating))
        and pd.isna(value)
    ):
        return ""

    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def normalize_run(value):
    match = re.fullmatch(
        r"(?:run[\s_:-]*)?([012])"
        r"(?:st|nd|rd)?(?:[\s_:-]*run)?",
        clean(value).casefold(),
    )

    return match.group(1) if match else ""


def normalize_dimension(name):
    name = re.sub(r"\s+", " ", name).strip().casefold()

    # Normalize behaviour variants
    if re.fullmatch(
        r"(?:\(?overt\)?\s+)?(?:behaviou?r|bevaviour)",
        name,
    ):
        return "overt behaviour"

    # Normalize self variants
    if name in {"self", "sense of self"}:
        return "sense of self"

    return name


def parse_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if (
        isinstance(value, (int, float, np.integer, np.floating))
        and not pd.isna(value)
        and value in (0, 1)
    ):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().casefold()

        if normalized in ("true", "false", "0", "1"):
            return normalized in ("true", "1")

    raise ValueError(
        f"Invalid Boolean: {value!r}"
    )


def parse_dimensions(value, *, missing_allowed):
    if value is None or (
        isinstance(value, (int, float, np.integer, np.floating))
        and pd.isna(value)
    ):
        if missing_allowed:
            return frozenset()

        raise ValueError(
            "Missing ground-truth dimensions"
        )

    if isinstance(value, str):
        value = value.strip()

        if not value:
            if missing_allowed:
                return frozenset()

            raise ValueError(
                "Missing ground-truth dimensions"
            )

        try:
            value = json.loads(value)

        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)

            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"Invalid dimension list: {value!r}"
                ) from error

    if (
        not isinstance(value, (list, tuple, set))
        or not all(isinstance(x, str) for x in value)
    ):
        raise ValueError(
            f"Dimensions must be a list of strings: {value!r}"
        )

    return frozenset(
        normalize_dimension(x)
        for x in value
        if x.strip()
    )


# ==================================================
# MODEL RESPONSE PARSING
# ==================================================

def extract_list(text, start):
    """
    Extract types even when surrounding JSON is invalid.
    """

    while start < len(text) and text[start].isspace():
        start += 1

    if start >= len(text) or text[start] != "[":
        raise ValueError(
            "types does not start with ["
        )

    depth = 0
    quote = None
    escaped = False

    for i in range(start, len(text)):
        char = text[i]

        if quote:
            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == quote:
                quote = None

        elif char in "\"'":
            quote = char

        elif char == "[":
            depth += 1

        elif char == "]":
            depth -= 1

            if depth == 0:
                return text[start:i + 1]

    raise ValueError(
        "Unclosed types list"
    )


def parse_prediction(text):
    """
    Extract only is_process and types.

    The utterance is ignored.
    JSON does not need to be valid.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Empty model response"
        )

    bool_matches = list(
        re.finditer(
            r'''(?<!\w)["']?is_process["']?\s*:\s*["']?(true|false|1|0)["']?(?!\w)''',
            text,
            flags=re.IGNORECASE,
        )
    )

    types_matches = list(
        re.finditer(
            r'''(?<!\w)["']?types["']?\s*:''',
            text,
            flags=re.IGNORECASE,
        )
    )

    if not bool_matches or not types_matches:
        raise ValueError(
            "Could not find is_process and types"
        )

    flag = parse_boolean(
        bool_matches[-1].group(1)
    )

    dims_text = extract_list(
        text,
        types_matches[-1].end(),
    )

    dims = parse_dimensions(
        dims_text,
        missing_allowed=False,
    )

    return flag, dims


# ==================================================
# READ EXCEL FILES
# ==================================================

def read_file(
    path,
    sheet,
    *,
    zeroshot,
    candidates,
    exclusions,
):
    sheet = (
        int(sheet)
        if str(sheet).isdigit()
        else sheet
    )

    df = pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        dtype=object,
    )

    if df.shape[1] <= PRED_COL:
        raise ValueError(
            f"{path}: expected at least 11 columns"
        )

    run_counts = Counter()

    for ix, row in df.iterrows():
        example_id = clean(
            row.iloc[ID_COL]
        )

        # Skip header
        if example_id.casefold() in (
            "id", "example_id", "example id"
        ):
            continue

        run = normalize_run(
            row.iloc[RUN_COL]
        )

        k = (
            "0"
            if zeroshot
            else clean(row.iloc[K_COL])
        )

        source = {
            "source_file": path.name,
            "excel_row": ix + 1,
            "id": example_id,
            "k": k,
            "run": clean(row.iloc[RUN_COL]),
        }

        # Skip empty rows
        if (
            not example_id
            and not clean(row.iloc[RUN_COL])
            and pd.isna(row.iloc[PRED_COL])
        ):
            continue

        if not example_id or not k or not run:
            exclusions.append({
                **source,
                "reason": "invalid_id_k_or_run",
                "details": "Expected ID, k, run 0/1/2",
            })
            continue

        run_counts[run] += 1
        source["run"] = run

        key = (example_id, k, run)

        try:
            # Ground truth
            gt_bool = parse_boolean(
                row.iloc[GT_BOOL_COL]
            )

            gt_dims = parse_dimensions(
                row.iloc[GT_DIMS_COL],
                missing_allowed=not gt_bool,
            )

            if not gt_bool and gt_dims:
                raise ValueError(
                    "Ground truth False with nonempty dimensions"
                )

            # Model prediction
            pred_bool, pred_dims = parse_prediction(
                row.iloc[PRED_COL]
            )

            record = {
                **source,
                "gt_bool": gt_bool,
                "gt_dims": gt_dims,
                "pred_bool": pred_bool,
                "pred_dims": pred_dims,
            }

            candidates[key].append(record)

        except (ValueError, TypeError) as exc:
            candidates[key].append(None)

            exclusions.append({
                **source,
                "reason": "parse_or_gt_error",
                "details": str(exc),
            })

    print(
        f"{path.name}: "
        + ", ".join(
            f"run {r}={run_counts[r]}"
            for r in RUNS
        )
    )


# ==================================================
# METRICS
# ==================================================

def kappa_binary(y_true, y_pred):
    """
    Cohen's Kappa between two binary label sequences.

    NaN is returned when mathematically undefined.
    """

    if len(y_true) == 0:
        return math.nan

    a = np.asarray(y_true, dtype=int)
    b = np.asarray(y_pred, dtype=int)

    observed = np.mean(a == b)

    p_a = a.mean()
    p_b = b.mean()

    expected = (
        p_a * p_b
        + (1 - p_a) * (1 - p_b)
    )

    if np.isclose(expected, 1):
        return math.nan

    value = float(
        (observed - expected)
        / (1 - expected)
    )

    if abs(value) < 1e-12:
        return 0.0

    return value


def prf(y_true, y_pred, average):
    """
    Precision, recall and F1.

    Undefined precision/recall/F1 values become 0.
    """

    labels = (
        [0, 1]
        if average in ("micro", "macro")
        and np.asarray(y_true).ndim == 1
        else None
    )

    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=average,
        zero_division=0,
    )

    return float(p), float(r), float(f)


# ==================================================
# EVALUATE ONE RUN / K
# ==================================================

def evaluate_scope(
    records,
    k,
    run,
    scope,
    vocabulary,
    boolean_results,
    dimension_summaries,
    dimension_results,
):
    n = len(records)

    # ----------------------------------------------
    # BOOLEAN EVALUATION
    # ----------------------------------------------

    if scope == "all":
        truths = np.array([
            int(r["gt_bool"])
            for r in records
        ])

        preds = np.array([
            int(r["pred_bool"])
            for r in records
        ])

        p, recall, f1 = prf(
            truths, preds, "binary"
        )

        pma, rma, fma = prf(
            truths, preds, "macro"
        )

        pmi, rmi, fmi = prf(
            truths, preds, "micro"
        )

        boolean_results.append({
            "k": k,
            "run": run,
            "n_evaluated": n,
            "gt_positive": int(truths.sum()),
            "pred_positive": int(preds.sum()),

            "precision_positive": p,
            "recall_positive": recall,
            "f1_positive": f1,

            "precision_macro": pma,
            "recall_macro": rma,
            "f1_macro": fma,

            "precision_micro": pmi,
            "recall_micro": rmi,
            "f1_micro": fmi,

            "accuracy": float(
                np.mean(truths == preds)
            ),

            "cohens_kappa": kappa_binary(
                truths, preds
            ),
        })

    # ----------------------------------------------
    # DIMENSION EVALUATION
    # ----------------------------------------------

    # Dimensions occurring in this subset
    active_dims = sorted(
        set().union(*(
            r["gt_dims"] | r["pred_dims"]
            for r in records
        ))
    )

    if active_dims:
        yt = np.array([
            [
                int(d in r["gt_dims"])
                for d in active_dims
            ]
            for r in records
        ])

        yp = np.array([
            [
                int(d in r["pred_dims"])
                for d in active_dims
            ]
            for r in records
        ])

        pmi, rmi, fmi = prf(
            yt, yp, "micro"
        )

        pma, rma, fma = prf(
            yt, yp, "macro"
        )

    else:
        pmi = rmi = fmi = math.nan
        pma = rma = fma = math.nan

    dimension_summaries.append({
        "k": k,
        "run": run,
        "scope": scope,
        "n_evaluated": n,

        "n_active_dimensions": len(active_dims),

        "precision_micro": pmi,
        "recall_micro": rmi,
        "f1_micro": fmi,

        "precision_macro": pma,
        "recall_macro": rma,
        "f1_macro": fma,

        "exact_set_accuracy": sum(
            r["gt_dims"] == r["pred_dims"]
            for r in records
        ) / n,
    })

    # ----------------------------------------------
    # PER-DIMENSION EVALUATION
    # ----------------------------------------------

    defined_kappas = []

    for dimension in vocabulary:
        yt = np.array([
            int(dimension in r["gt_dims"])
            for r in records
        ])

        yp = np.array([
            int(dimension in r["pred_dims"])
            for r in records
        ])

        p, recall, f1 = prf(
            yt, yp, "binary"
        )

        value_kappa = kappa_binary(
            yt, yp
        )

        if not math.isnan(value_kappa):
            defined_kappas.append(
                value_kappa
            )

        dimension_results.append({
            "k": k,
            "run": run,
            "scope": scope,
            "dimension": dimension,
            "n_evaluated": n,

            "gt_positive": int(yt.sum()),
            "pred_positive": int(yp.sum()),

            "tp": int(
                np.sum((yt == 1) & (yp == 1))
            ),

            "fp": int(
                np.sum((yt == 0) & (yp == 1))
            ),

            "fn": int(
                np.sum((yt == 1) & (yp == 0))
            ),

            "tn": int(
                np.sum((yt == 0) & (yp == 0))
            ),

            "precision": p,
            "recall": recall,
            "f1": f1,
            "cohens_kappa": value_kappa,
        })

    # Macro-average of defined dimension Kappas
    dimension_summaries[-1][
        "dimension_macro_kappa"
    ] = (
        float(np.mean(defined_kappas))
        if defined_kappas
        else math.nan
    )

    dimension_summaries[-1][
        "n_dimensions_with_defined_kappa"
    ] = len(defined_kappas)


# ==================================================
# AVERAGE ACROSS THREE RUNS
# ==================================================

def mean_over_three_runs(df, group_cols):
    """
    Average metrics across the three runs.

    Zero-shot is excluded because it has one run.
    """

    selected = df[
        df["k"] != "OVERALL_ALL_AVAILABLE"
    ]

    metric_cols = [
        col for col in df.columns
        if (
            col.startswith((
                "precision",
                "recall",
                "f1",
            ))
            or col in (
                "accuracy",
                "cohens_kappa",
                "dimension_macro_kappa",
                "exact_set_accuracy",
            )
        )
    ]

    output = []

    for keys, part in selected.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):

        if (
            set(part["run"].astype(str)) != set(RUNS)
            or len(part) != 3
        ):
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(
            zip(group_cols, keys)
        )

        row["n_runs"] = 3

        row["min_n_evaluated"] = int(
            part["n_evaluated"].min()
        )

        row["max_n_evaluated"] = int(
            part["n_evaluated"].max()
        )

        for col in metric_cols:
            row["mean_" + col] = (
                part[col].mean(skipna=True)
            )

            if "kappa" in col:
                row["n_defined_" + col] = int(
                    part[col].notna().sum()
                )

        output.append(row)

    return pd.DataFrame(output)


# ==================================================
# MAIN
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model predictions "
            "against ground truth"
        )
    )

    parser.add_argument(
        "fewshot_file",
        type=Path,
    )

    parser.add_argument(
        "zeroshot_file",
        type=Path,
    )

    parser.add_argument(
        "--fewshot-sheet",
        default="0",
    )

    parser.add_argument(
        "--zeroshot-sheet",
        default="0",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results"),
    )

    args = parser.parse_args()

    if (
        args.fewshot_file.resolve()
        == args.zeroshot_file.resolve()
    ):
        parser.error(
            "Supply two different files"
        )

    for path in (
        args.fewshot_file,
        args.zeroshot_file,
    ):
        if not path.is_file():
            parser.error(
                f"File not found: {path}"
            )

    candidates = defaultdict(list)
    exclusions = []

    # ----------------------------------------------
    # READ BOTH DATASETS
    # ----------------------------------------------

    read_file(
        args.fewshot_file,
        args.fewshot_sheet,
        zeroshot=False,
        candidates=candidates,
        exclusions=exclusions,
    )

    read_file(
        args.zeroshot_file,
        args.zeroshot_sheet,
        zeroshot=True,
        candidates=candidates,
        exclusions=exclusions,
    )

    records = []

    for (
        example_id, k, run
    ), entries in candidates.items():

        if len(entries) != 1:
            exclusions.append({
                "source_file": "",
                "excel_row": "",
                "id": example_id,
                "k": k,
                "run": run,
                "reason": "duplicate_id_k_run",
                "details": "All duplicates excluded",
            })

        elif entries[0] is not None:
            records.append(entries[0])

    if not records:
        raise SystemExit(
            "No valid examples found"
        )

    # ----------------------------------------------
    # PRINT AVAILABLE DIMENSIONS
    # ----------------------------------------------

    vocabulary = sorted(
        set().union(*(
            r["gt_dims"] | r["pred_dims"]
            for r in records
        ))
    )

    print(
        f"\nAVAILABLE DIMENSIONS "
        f"({len(vocabulary)}):"
    )

    for dimension in vocabulary:
        print(f"  - {dimension}")

    # ----------------------------------------------
    # PREPARE EVALUATION GROUPS
    # ----------------------------------------------

    k_values = sorted(
        {r["k"] for r in records},
        key=lambda v: (
            (0, int(v))
            if v.isdigit()
            else (1, v)
        ),
    )

    slices = []

    for k in k_values:
        for run in RUNS:
            subset = [
                r for r in records
                if r["k"] == k
                and r["run"] == run
            ]

            if subset:
                slices.append(
                    (k, run, subset)
                )

    # Overall few-shot evaluation for each run
    for run in RUNS:
        subset = [
            r for r in records
            if r["run"] == run
            and r["k"] != "0"
        ]

        if subset:
            slices.append((
                "OVERALL_FEWSHOT",
                run,
                subset,
            ))

    # Optional overall for run 0 including zero-shot
    subset = [
        r for r in records
        if r["run"] == "0"
    ]

    if subset:
        slices.append((
            "OVERALL_ALL_AVAILABLE",
            "0",
            subset,
        ))

    # ----------------------------------------------
    # CALCULATE ALL METRICS
    # ----------------------------------------------

    boolean_results = []
    dimension_summaries = []
    dimension_results = []

    for k, run, subset in slices:

        # Evaluate all valid examples
        evaluate_scope(
            subset,
            k,
            run,
            "all",
            vocabulary,
            boolean_results,
            dimension_summaries,
            dimension_results,
        )

        # Evaluate dimensions on GT-positive examples
        gt_positive = [
            r for r in subset
            if r["gt_bool"]
        ]

        if gt_positive:
            evaluate_scope(
                gt_positive,
                k,
                run,
                "gt_positive",
                vocabulary,
                boolean_results,
                dimension_summaries,
                dimension_results,
            )

    # ----------------------------------------------
    # CREATE RESULT TABLES
    # ----------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    boolean_table = pd.DataFrame(
        boolean_results
    )

    dim_summary_table = pd.DataFrame(
        dimension_summaries
    )

    per_dim_table = pd.DataFrame(
        dimension_results
    )

    tables = {
        "boolean_per_run": boolean_table,

        "dimensions_summary": dim_summary_table,

        "per_dimension": per_dim_table,

        "boolean_mean_3runs": mean_over_three_runs(
            boolean_table,
            ["k"],
        ),

        "dimensions_mean_3runs": mean_over_three_runs(
            dim_summary_table,
            ["k", "scope"],
        ),

        "dimension_mean_3runs": mean_over_three_runs(
            per_dim_table,
            ["k", "scope", "dimension"],
        ),

        "excluded_rows": pd.DataFrame(
            exclusions,
            columns=[
                "source_file",
                "excel_row",
                "id",
                "k",
                "run",
                "reason",
                "details",
            ],
        ),

        "evaluated_rows": pd.DataFrame([
            {
                **{
                    field: r[field]
                    for field in (
                        "source_file",
                        "excel_row",
                        "id",
                        "k",
                        "run",
                    )
                },

                "gt_is_process": r["gt_bool"],
                "pred_is_process": r["pred_bool"],

                "gt_types": json.dumps(
                    sorted(r["gt_dims"])
                ),

                "pred_types": json.dumps(
                    sorted(r["pred_dims"])
                ),

                "boolean_correct": (
                    r["gt_bool"] == r["pred_bool"]
                ),

                "types_exact_match": (
                    r["gt_dims"] == r["pred_dims"]
                ),
            }
            for r in records
        ]),
    }

    definitions = [
        (
            "scope",
            "all = all valid examples; "
            "gt_positive = ground-truth is_process=True only.",
        ),
        (
            "OVERALL_FEWSHOT",
            "All few-shot k values pooled. "
            "Zero-shot excluded.",
        ),
        (
            "OVERALL_ALL_AVAILABLE",
            "Run 0 including zero-shot; "
            "not directly comparable to runs 1/2.",
        ),
        (
            "*_mean_3runs",
            "Arithmetic mean of the three run scores; "
            "zero-shot excluded.",
        ),
        (
            "dimension_macro_kappa",
            "Average Kappa across dimensions "
            "with defined Kappa values.",
        ),
        (
            "precision_positive / recall_positive / f1_positive",
            "Boolean evaluation for is_process=True.",
        ),
        (
            "precision_micro / recall_micro / f1_micro",
            "Micro-averaged metrics.",
        ),
        (
            "precision_macro / recall_macro / f1_macro",
            "Unweighted average across classes "
            "or active dimensions.",
        ),
        (
            "exact_set_accuracy",
            "Predicted dimension sets exactly match "
            "ground truth, ignoring order.",
        ),
        (
            "cohens_kappa",
            "Chance-corrected agreement between "
            "ground truth and prediction.",
        ),
        (
            "tp / fp / fn / tn",
            "True positives, false positives, "
            "false negatives and true negatives.",
        ),
        (
            "Undefined metrics",
            "Undefined precision/recall/F1 become zero; "
            "undefined Kappa is left empty.",
        ),
    ]

    tables["definitions"] = pd.DataFrame(
        definitions,
        columns=["column", "definition"],
    )

    # ----------------------------------------------
    # EXPORT CSV FILES
    # ----------------------------------------------

    for name, df in tables.items():
        df.to_csv(
            args.output / f"{name}.csv",
            index=False,
        )

    # ----------------------------------------------
    # EXPORT ONE EXCEL WORKBOOK
    # ----------------------------------------------

    xlsx_path = (
        args.output / "model_evaluation.xlsx"
    )

    with pd.ExcelWriter(
        xlsx_path,
        engine="openpyxl",
    ) as writer:

        for name, df in tables.items():

            df.to_excel(
                writer,
                sheet_name=name,
                index=False,
            )

            ws = writer.sheets[name]

            ws.freeze_panes = "C2"
            ws.auto_filter.ref = ws.dimensions

            for column in ws.columns:
                letter = column[0].column_letter

                sample_length = max(
                    (
                        len(str(cell.value))
                        for cell in column[:100]
                        if cell.value is not None
                    ),
                    default=10,
                )

                ws.column_dimensions[letter].width = min(
                    max(sample_length + 2, 12),
                    42,
                )

    # ----------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------

    print("\nBOOLEAN OVERALL_FEWSHOT:")

    print(
        tables["boolean_per_run"]
        .query('k == "OVERALL_FEWSHOT"')
        .to_string(index=False)
    )

    print(
        f"\nSaved Excel report: "
        f"{xlsx_path.resolve()}"
    )

    print(
        f"Excluded rows: {len(exclusions)}"
    )


if __name__ == "__main__":
    main()
