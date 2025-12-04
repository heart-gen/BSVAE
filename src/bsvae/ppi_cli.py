"""CLI utility to prefetch STRING PPI networks into the local cache."""

import argparse
import logging
import sys
from os.path import expanduser

from bsvae.utils.ppi import DEFAULT_CACHE_DIR, download_string


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Download STRING v12.0 protein–protein interaction networks "
            "into the BSVAE cache."
        )
    )
    parser.add_argument(
        "--taxid",
        type=str,
        default="9606",
        help="NCBI taxonomy identifier (default: 9606 for human).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=(
            "Directory to store the downloaded file. Defaults to "
            "BSVAE_PPI_CACHE or ~/.bsvae/ppi."
        ),
    )
    return parser.parse_args(args)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger(__name__)


def cli():
    args = parse_args(sys.argv[1:])
    logger = setup_logging()

    cache_dir = expanduser(args.cache_dir)
    filepath = download_string(taxid=args.taxid, cache_dir=cache_dir)
    logger.info("STRING PPI cached at %s", filepath)


if __name__ == "__main__":
    cli()
