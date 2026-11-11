# SPDX-License-Identifier: Apache-2.0
"""tests._tokenizers.load_tokenizer: skip by default, fail under SR_REQUIRE_TOKENIZERS=1."""

import pytest

from tests._tokenizers import REQUIRE_ENV, load_tokenizer

MISSING = "/nonexistent/sr-test-tokenizer"


def test_missing_tokenizer_skips_by_default(monkeypatch):
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="tokenizer unavailable"):
        load_tokenizer(MISSING)


def test_missing_tokenizer_fails_when_required(monkeypatch):
    monkeypatch.setenv(REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match=f"{REQUIRE_ENV}=1"):
        load_tokenizer(MISSING)
