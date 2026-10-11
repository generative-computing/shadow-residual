# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``training_config.schema.json`` from :class:`TrainingConfig`.

The reference YAMLs point editors at this schema
(``# yaml-language-server: $schema=./schema/training_config.schema.json``), so it
must match the pydantic model that actually validates configs.

    python src/shadow_residual/config/schema/generate_config_schema.py          # rewrite the schema
    python src/shadow_residual/config/schema/generate_config_schema.py --check  # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shadow_residual.config import TrainingConfig

SCHEMA_PATH = Path(__file__).resolve().with_name("training_config.schema.json")


def render() -> str:
    return json.dumps(TrainingConfig.model_json_schema(), indent=2) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Do not write; exit 1 if the committed schema is stale.")
    args = p.parse_args()
    text = render()
    if args.check:
        if SCHEMA_PATH.read_text() != text:
            print(f"{SCHEMA_PATH} is stale; run: python src/shadow_residual/config/schema/generate_config_schema.py", file=sys.stderr)
            return 1
        return 0
    SCHEMA_PATH.write_text(text)
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
