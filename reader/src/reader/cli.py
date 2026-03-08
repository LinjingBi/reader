"""CLI entry point for reader package"""

import argparse
import asyncio
import sys

from reader.pipelines.hf_data.config.config import load_config as load_hf_data_config
from reader.pipelines.report_generation.config.config import load_config as load_report_config
from reader.pipelines.collect_data import run_hf_data
from reader.pipelines.generate_report import generate_report
from reader.logging.logging_setup import setup_logging, get_logger


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Reader: Monthly paper clustering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # hf-data subcommand
    hf_parser = subparsers.add_parser('hf-data', help='Run HF data pipeline (fetch, cluster, chunk)')
    hf_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to HF data YAML config file (e.g., pipelines/hf_data/config/hf-data.yaml)',
    )

    # report subcommand
    report_parser = subparsers.add_parser('report', help='Run report generation pipeline')
    report_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to report YAML config file (e.g., pipelines/report_generation/config/report.yaml)',
    )

    args = parser.parse_args()

    if args.command == 'hf-data':
        try:
            config = load_hf_data_config(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

        setup_logging(config.run.log_config_path, config.run.log_file_path)
        logger = get_logger()

        try:
            asyncio.run(run_hf_data(config))
        except Exception as e:
            logger.error(f"Error running pipeline: {e}", exc_info=True)
            sys.exit(1)

    elif args.command == 'report':
        try:
            config = load_report_config(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

        setup_logging(config.run.log_config_path, config.cache.report_generation_log_path)
        logger = get_logger()

        try:
            asyncio.run(generate_report(config))
        except Exception as e:
            logger.error(f"Error running report pipeline: {e}", exc_info=True)
            sys.exit(1)
