import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic healthcheck")
    parser.add_argument(
        "--only",
        help="Comma-separated list of checks to run (weaviate,postgres,redis)",
    )
    parser.add_argument(
        "--skip",
        help="Comma-separated list of checks to skip",
    )
    return parser.parse_args()