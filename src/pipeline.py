"""Plain-Python entry point that runs one stage or the whole pipeline in dependency order.

    python -m src.pipeline                 # all stages
    python -m src.pipeline --stage validate

The Prefect flow in ``dags/recommendation_flow.py`` orchestrates the same stage functions.
"""

from __future__ import annotations

import argparse
import time

from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.utils import get_logger, set_seed
from src.validation import validate

log = get_logger("pipeline")

STAGES = {
    "ingest": ingest.run,
    "validate": validate.run,
    "prepare": prepare.run,
    "transform": transform.run,
    "features": feature_store.run,
    "train": train.run,
    "evaluate": evaluate.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recommendation pipeline")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    args = parser.parse_args()
    set_seed()
    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    t0 = time.time()
    for name, fn in stages.items():
        start = time.time()
        log.info("===== stage %-9s started =====", name)
        fn()
        log.info("===== stage %-9s finished in %.1fs =====", name, time.time() - start)
    log.info("Pipeline finished (%s) in %.1fs", ", ".join(stages), time.time() - t0)


if __name__ == "__main__":
    main()
