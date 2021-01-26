#!/usr/bin/env python
"""
Update a JSON config from an old format to the latest format

> Tyler Ganter, tganter@sandia.gov
"""
import argparse
import logging

from ctseg.ctutil.utils import read_json, write_json

from ctseg.schema import validate_schema, SCHEMA_UPDATES


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update JSON config to new schema")
    parser.add_argument(
        "inpath", type=str, help="The path to the input JSON config file"
    )
    parser.add_argument(
        "outpath", type=str, help="The path to the output JSON config file"
    )
    parser.add_argument(
        "-v",
        "--input-version",
        default=0,
        help="The integer schema version of the input file. See the docstrings of the "
             "`ctseg.schema.update_schema_*()` functions to check which commit and"
             " config schema your config file is.",
    )
    args = parser.parse_args()

    logger.info("Reading JSON")
    d = read_json(args.inpath)

    logger.info("Updating schema")

    for update_func in SCHEMA_UPDATES[int(args.input_version)::]:
        d = update_func(d)

    logger.info("Validating schema")
    validate_schema(d)

    logger.info(f"Schema updated. Writing to '{args.outpath}'")
    write_json(d, args.outpath, pretty_print=True)

    logger.info("Complete")
