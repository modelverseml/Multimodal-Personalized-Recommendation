"""Category-distribution drift (PSI) between training data and recent live events.

    python -m scripts.drift [--days 7]

Exit code 2 when PSI exceeds the configured threshold, so this can gate a
scheduled retraining job. The threshold is a monitoring heuristic, not a
universal statistical cutoff.
"""
import argparse
import json
import sys

from app.config import settings
from app.database import SessionLocal
from app.monitoring import drift_report, engagement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    manifest = json.loads((settings.artifact_dir / "current" / "manifest.json").read_text())
    with SessionLocal() as db:
        report = drift_report(db, manifest, days=args.days)
        report["engagement"] = engagement(db, days=args.days)
    print(json.dumps(report, indent=2))
    if report["status"] == "drift":
        sys.exit(2)


if __name__ == "__main__":
    main()
