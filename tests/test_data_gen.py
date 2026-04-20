import pandas as pd

from data_gen import build_payload, sanitize_json_value


def test_sanitize_json_value_replaces_nan_with_none():
    payload = {"value": float("nan"), "items": [1.0, float("inf")]}

    sanitized = sanitize_json_value(payload)

    assert sanitized == {"value": None, "items": [1.0, None]}


def test_build_payload_uses_transaction_window_when_available():
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        }
    )

    payload = build_payload(frame, transaction_id=2, num_rows=2)

    assert [row["TransactionID"] for row in payload["records"]] == [2, 3]


def test_build_payload_falls_back_to_first_rows_when_transaction_id_missing():
    frame = pd.DataFrame(
        {
            "TransactionID": [10, 11, 12],
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    )

    payload = build_payload(frame, transaction_id=999, num_rows=2)

    assert [row["TransactionID"] for row in payload["records"]] == [10, 11]


def test_build_payload_requires_positive_num_rows():
    frame = pd.DataFrame({"TransactionID": [1], "TransactionAmt": [10.0]})

    try:
        build_payload(frame, transaction_id=1, num_rows=0)
    except ValueError as exc:
        assert "num_rows" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive num_rows.")
