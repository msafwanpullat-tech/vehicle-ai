import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.analysis_history import import_legacy_manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import completed .analysis_sessions manifests into PostgreSQL."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit valid records. Without this option, perform a dry run.",
    )
    arguments = parser.parse_args()
    result = import_legacy_manifests(dry_run=not arguments.apply)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
