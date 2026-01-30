"""CLI entry point for reader package"""

import argparse
import sys
from pathlib import Path

from reader.config import load_config
from reader.pipelines.monthly import run_monthly
from reader.logging.logging_setup import setup_logging, get_logger


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Reader: Monthly paper clustering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file (e.g., configs/reader.yaml)',
    )
    
    args = parser.parse_args()
    
    # Load config first to get artifacts_dir
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Resolve log_config_path relative to project root (current working directory)
    log_config_path = Path(config.run.log_config_path).resolve()
    
    # Setup logging using paths from config
    setup_logging(log_config_path, config.run.log_file_path)
    logger = get_logger()
    
    # Run pipeline
    try:
        run_monthly(config)
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        sys.exit(1)
