"""CLI entry point for reader package"""

import argparse
import asyncio
import sys

from reader.pipelines.hf_data.config.config import load_config as load_hf_data_config
from reader.pipelines.report_generation.config.config import load_config as load_report_config
from reader.pipelines.report_render.config.config import load_config as load_render_report_config
from reader.pipelines.collect_data import run_hf_data
from reader.pipelines.generate_report import generate_report
from reader.pipelines.render_report import render_report
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

    # render-report subcommand
    render_report_parser = subparsers.add_parser(
        'render-report',
        help='Fetch report from memo, validate, optionally check signature, display in TUI',
    )
    render_report_parser.add_argument(
        '--cluster-pk-hash',
        type=str,
        required=True,
        help='Cluster pk_hash for report lookup',
    )
    render_report_parser.add_argument(
        '--intent',
        type=str,
        default=None,
        help='Intent mode for validation (e.g., quick_background). Optional.',
    )
    render_report_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to render_report YAML config (e.g., pipelines/report_render/config/render_report.yaml)',
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

    elif args.command == 'generate-report':
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

    elif args.command == 'render-report':
        try:
            config = load_render_report_config(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

        setup_logging(config.log_config_path, config.cache.render_report_log_path)
        logger = get_logger()

        try:
            output = asyncio.run(
                render_report(
                    cluster_pk_hash=args.cluster_pk_hash,
                    intent_mode=args.intent,
                    cfg=config,
                )
            )
            if output.status == "error":
                print(output.message, file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error running render-report: {e}", exc_info=True)
            sys.exit(1)
    else:
        print(f"Invalid command: {args.command}", file=sys.stderr)
        sys.exit(1)
