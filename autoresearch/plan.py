"""Plan autoresearch trials without launching DESC."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .sampling import planned_trials_for_generation, write_trial_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to campaign config.")
    parser.add_argument("--generation", type=int, default=0, help="Generation to plan.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned trials only.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    records = planned_trials_for_generation(config, args.generation, previous_records=[])
    if args.dry_run:
        print(json.dumps(records, indent=2, sort_keys=True, default=str))
        return 0

    for record in records:
        write_trial_json(record)
    print(f"Wrote {len(records)} trial JSON files for generation {args.generation}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
