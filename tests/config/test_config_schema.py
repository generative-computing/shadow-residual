# SPDX-License-Identifier: Apache-2.0
"""The committed JSON schema must match the pydantic model that validates configs."""

from __future__ import annotations

import json
from pathlib import Path

from shadow_residual.config import TrainingConfig

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src/shadow_residual/config/schema/training_config.schema.json"


def test_committed_schema_matches_training_config():
    committed = json.loads(SCHEMA_PATH.read_text())
    assert committed == TrainingConfig.model_json_schema(), (
        "training_config.schema.json is stale; regenerate it with: python scripts/generate_config_schema.py"
    )


def test_schema_has_no_removed_options():
    # shared_base_kv (dual-K/V toggle) and invocation_tokens (aLoRA gating) were removed from the model;
    # a schema that still offers them makes editors suggest keys that fail validation.
    defs = json.loads(SCHEMA_PATH.read_text())["$defs"]
    assert "shared_base_kv" not in defs["ModelConfig"]["properties"]
    assert "invocation_tokens" not in defs["AdapterConfig"]["properties"]
