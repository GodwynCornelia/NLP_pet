import argparse
import logging
import os

from support_router.common import run_paths, setup_compute
from support_router.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Support Router reproducible CPU pipeline")
    parser.add_argument(
        "command",
        choices=[
            "download",
            "prepare",
            "train",
            "tune-policy",
            "build-index",
            "evaluate",
            "verify-artifacts",
            "bootstrap",
        ],
    )
    parser.add_argument("--config")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--refresh", action="store_true", help="Explicitly refresh source revisions"
    )
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
    config = load_config(args.config)
    _controller = setup_compute(config)
    from support_router.data import download, prepare

    if args.command == "download":
        download(config, args.refresh)
    elif args.command == "prepare":
        prepare(config, args.run_id)
    elif args.command == "bootstrap":
        download(config, args.refresh)
        run_id = prepare(config, args.run_id)
        for command in ["train", "tune-policy", "build-index", "evaluate", "verify-artifacts"]:
            execute(command, config, run_id)
    else:
        execute(args.command, config, run_paths(config, args.run_id)[0])


def execute(command, config, run_id):
    if command in {"train", "tune-policy"}:
        from support_router.training import train, tune_policy

        {"train": train, "tune-policy": tune_policy}[command](config, run_id)
    elif command == "build-index":
        from support_router.retrieval import build_index

        build_index(config, run_id)
    elif command == "evaluate":
        from support_router.evaluation import evaluate

        evaluate(config, run_id)
    elif command == "verify-artifacts":
        from support_router.runtime import Router

        router = Router(config, run_id)
        print(f"Verified {router.run_id}: model, policy, encoder, train_fit-only index")


if __name__ == "__main__":
    main()
