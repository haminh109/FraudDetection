import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "merged_train_data.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "sample_request.json"
DEFAULT_TRANSACTION_ID = 3577280
DEFAULT_NUM_ROWS = 70


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value


def build_payload(
    df: pd.DataFrame,
    transaction_id: int | None = DEFAULT_TRANSACTION_ID,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> dict[str, list[dict[str, Any]]]:
    if num_rows <= 0:
        raise ValueError("num_rows must be greater than 0.")

    start_index = 0
    if transaction_id is not None:
        if "TransactionID" not in df.columns:
            raise ValueError("TransactionID column is required when transaction_id is provided.")

        matching_indices = df.index[df["TransactionID"] == transaction_id].tolist()
        if matching_indices:
            start_index = matching_indices[0]

    target_rows = df.iloc[start_index:start_index + num_rows]
    if target_rows.empty:
        raise ValueError("No rows available to build a sample request payload.")

    records = sanitize_json_value(target_rows.to_dict(orient="records"))
    return {"records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample_request.json from raw fraud data.")
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the raw merged training CSV.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the generated JSON payload.",
    )
    parser.add_argument(
        "--transaction-id",
        type=int,
        default=DEFAULT_TRANSACTION_ID,
        help="Optional TransactionID used as the starting point for the sample window.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=DEFAULT_NUM_ROWS,
        help="Number of rows to include in the generated payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    df = pd.read_csv(input_path)
    payload = build_payload(
        df,
        transaction_id=args.transaction_id,
        num_rows=args.num_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved RAW samples to {output_path}")


if __name__ == "__main__":
    main()
