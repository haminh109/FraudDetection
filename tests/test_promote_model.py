from types import SimpleNamespace

import pytest

from src.promote_model import promote_model_version, select_model_version


class DummyClient:
    def __init__(self):
        self.calls = []

    def set_model_version_tag(self, name, version, key, value):
        self.calls.append(("set_model_version_tag", name, version, key, value))

    def set_registered_model_alias(self, name, alias, version):
        self.calls.append(("set_registered_model_alias", name, alias, version))

    def transition_model_version_stage(
        self,
        name,
        version,
        stage,
        archive_existing_versions=False,
    ):
        self.calls.append(
            (
                "transition_model_version_stage",
                name,
                version,
                stage,
                archive_existing_versions,
            )
        )

    def get_model_version(self, name, version):
        return SimpleNamespace(
            name=name,
            version=version,
            run_id="run-3",
            current_stage="Staging",
        )


def test_select_model_version_prefers_latest_version_for_run_id():
    versions = [
        SimpleNamespace(version="1", run_id="run-1"),
        SimpleNamespace(version="2", run_id="run-2"),
        SimpleNamespace(version="3", run_id="run-2"),
    ]

    version, run_id = select_model_version(versions, run_id="run-2")

    assert version == "3"
    assert run_id == "run-2"


def test_select_model_version_raises_when_no_version_matches():
    versions = [SimpleNamespace(version="1", run_id="run-1")]

    with pytest.raises(ValueError, match="No model versions matched"):
        select_model_version(versions, run_id="missing-run")


def test_promote_model_version_sets_candidate_alias_tags_and_stage():
    client = DummyClient()

    promoted = promote_model_version(
        client=client,
        model_name="fraud_detection_model",
        version="7",
        alias="candidate",
        tag_key="deployment_status",
        tag_value="Candidate",
        stage="Staging",
        archive_existing=True,
    )

    assert promoted.version == "7"
    assert (
        "set_model_version_tag",
        "fraud_detection_model",
        "7",
        "deployment_status",
        "Candidate",
    ) in client.calls
    assert (
        "set_model_version_tag",
        "fraud_detection_model",
        "7",
        "candidate_alias",
        "candidate",
    ) in client.calls
    assert (
        "set_registered_model_alias",
        "fraud_detection_model",
        "candidate",
        "7",
    ) in client.calls
    assert (
        "transition_model_version_stage",
        "fraud_detection_model",
        "7",
        "Staging",
        True,
    ) in client.calls
