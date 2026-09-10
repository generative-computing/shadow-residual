# SPDX-License-Identifier: Apache-2.0
"""Read-back of SR-specific keys from a saved adapter_config.json.

These keys are structural/forward-path choices train.py records into
adapter_config.json so the serving path can rebuild a matching SR base before
PEFT attaches (see build.build_sr_base). Defaults must be safe for older adapters
that predate the keys.
"""
from __future__ import annotations

import json

from shadow_residual.training.generation_utils import (
    read_cross_stream_taps_from_adapter,
    read_cross_stream_type_from_adapter,
    read_share_moe_routing_from_adapter,
)


def _write_adapter_cfg(tmp_path, **extra):
    cfg = {"peft_type": "LORA", "r": 32}
    cfg.update(extra)
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


# read_cross_stream_type_from_adapter returns (type, dim). linear is
# single-tap-only (mutually exclusive with the multi-tap registry), so one
# scalar dim fully describes it.


def test_cross_stream_type_defaults_lora_when_absent(tmp_path):
    path = _write_adapter_cfg(tmp_path)  # no cross_stream_* keys
    assert read_cross_stream_type_from_adapter(path) == ("lora", None)


def test_cross_stream_type_reads_lora(tmp_path):
    path = _write_adapter_cfg(tmp_path, cross_stream_type="lora")
    assert read_cross_stream_type_from_adapter(path) == ("lora", None)


def test_cross_stream_type_reads_linear_with_dim(tmp_path):
    path = _write_adapter_cfg(
        tmp_path, cross_stream_type="linear", cross_stream_dim=32
    )
    assert read_cross_stream_type_from_adapter(path) == ("linear", 32)


def test_cross_stream_type_missing_file_defaults(tmp_path):
    # An empty dir (no adapter_config.json) → safe default.
    assert read_cross_stream_type_from_adapter(str(tmp_path)) == ("lora", None)


def test_cross_stream_taps_default_when_absent(tmp_path):
    # No cross_stream* entry → the historical single-tap topology.
    path = _write_adapter_cfg(tmp_path, target_modules=["q_proj"])
    assert read_cross_stream_taps_from_adapter(path) == ("cross_stream",)


def test_cross_stream_taps_from_target_modules_lora(tmp_path):
    path = _write_adapter_cfg(
        tmp_path, target_modules=["q_proj", "cross_stream", "cross_stream_post_attn"]
    )
    assert read_cross_stream_taps_from_adapter(path) == (
        "cross_stream", "cross_stream_post_attn",
    )


def test_cross_stream_taps_from_modules_to_save_linear(tmp_path):
    # linear taps live in modules_to_save, not target_modules — still derived.
    path = _write_adapter_cfg(
        tmp_path, target_modules=["q_proj"],
        modules_to_save=["cross_stream", "cross_stream_post_attn"],
    )
    assert read_cross_stream_taps_from_adapter(path) == (
        "cross_stream", "cross_stream_post_attn",
    )


def test_share_moe_routing_default_false(tmp_path):
    path = _write_adapter_cfg(tmp_path)
    assert read_share_moe_routing_from_adapter(path) is False
