# SPDX-License-Identifier: Apache-2.0
"""Hub tokenizer loading for tests: skip when unavailable, or fail when
``SR_REQUIRE_TOKENIZERS=1`` so a test run cannot silently skip the
tokenizer-dependent cases (paste such a run, with its skip count, in a PR)."""

from __future__ import annotations

import os

import pytest

REQUIRE_ENV = "SR_REQUIRE_TOKENIZERS"


def load_tokenizer(model_id: str):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:  # pragma: no cover - environment dependent
        msg = f"{model_id} tokenizer unavailable: {e}"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{msg} ({REQUIRE_ENV}=1)")
        pytest.skip(msg)
