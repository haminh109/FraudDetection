import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote a registered MLflow model version to Candidate."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", ""),
    )
    parser.add_argument("--version", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--candidate-alias",
        default=os.getenv("MLFLOW_CANDIDATE_ALIAS", "candidate"),
    )
    parser.add_argument(
        "--candidate-tag-key",
        default=os.getenv("MLFLOW_CANDIDATE_TAG_KEY", "deployment_status"),
    )
    parser.add_argument(
        "--candidate-tag-value",
        default=os.getenv("MLFLOW_CANDIDATE_TAG_VALUE", "Candidate"),
    )
    parser.add_argument(
        "--stage",
        default=os.getenv("MLFLOW_CANDIDATE_STAGE", ""),
    )
    parser.add_argument("--archive-existing", action="store_true")
    parser.add_argument(
        "--output-path",
        default="metrics/model_registry_promotion.json",
    )
    return parser.parse_args()


def escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def resolve_tracking_uri(tracking_uri: str) -> str:
    return tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:./mlruns"


def fetch_model_versions(client: MlflowClient, model_name: str):
    filter_string = f"name = '{escape_filter_value(model_name)}'"
    return list(client.search_model_versions(filter_string=filter_string))


def select_model_version(model_versions, version: str = "", run_id: str = ""):
    if version:
        version = str(version)
        for model_version in model_versions:
            if str(model_version.version) != version:
                continue
            if run_id and getattr(model_version, "run_id", "") != run_id:
                continue
            return version, getattr(model_version, "run_id", "")
        raise ValueError(
            f"Model version {version} was not found for the requested filters."
        )

    candidates = [
        model_version
        for model_version in model_versions
        if not run_id or getattr(model_version, "run_id", "") == run_id
    ]
    if not candidates:
        raise ValueError(
            "No model versions matched the requested model name and run_id."
        )

    selected = max(
        candidates,
        key=lambda model_version: int(model_version.version),
    )
    return str(selected.version), getattr(selected, "run_id", "")


def promote_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    alias: str,
    tag_key: str,
    tag_value: str,
    stage: str = "",
    archive_existing: bool = False,
):
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key=tag_key,
        value=tag_value,
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="candidate_alias",
        value=alias,
    )
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )

    if stage:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )

    return client.get_model_version(name=model_name, version=version)


def write_promotion_report(output_path: str, payload: dict):
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def main():
    args = parse_args()

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_versions = fetch_model_versions(
        client=client,
        model_name=args.model_name,
    )
    version, source_run_id = select_model_version(
        model_versions=model_versions,
        version=args.version,
        run_id=args.run_id,
    )

    promoted_version = promote_model_version(
        client=client,
        model_name=args.model_name,
        version=version,
        alias=args.candidate_alias,
        tag_key=args.candidate_tag_key,
        tag_value=args.candidate_tag_value,
        stage=args.stage,
        archive_existing=args.archive_existing,
    )

    payload = {
        "tracking_uri": tracking_uri,
        "model_name": args.model_name,
        "model_version": str(promoted_version.version),
        "source_run_id": source_run_id,
        "candidate_alias": args.candidate_alias,
        "candidate_tag": {
            args.candidate_tag_key: args.candidate_tag_value,
        },
        "current_stage": getattr(promoted_version, "current_stage", ""),
    }
    write_promotion_report(args.output_path, payload)

    logging.info(
        "Promoted model %s version %s to alias '%s'",
        args.model_name,
        promoted_version.version,
        args.candidate_alias,
    )


if __name__ == "__main__":
    main()
