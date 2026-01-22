"""CLI entry point for reader package"""

import argparse
import sys

from reader.config import load_config
from reader.pipelines.monthly import run_monthly


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
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run pipeline
    try:
        run_monthly(config)
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
