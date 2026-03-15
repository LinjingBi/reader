#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: ./run.sh <workflow> [args]
  hf-data                    Run HF data pipeline
  generate-report            Run report generation
  check-report-signature <report-file>   Validate report signature
  render-report <report-file>           Render report in TUI
EOF
}

# uv sync from reader/
cd "$SCRIPT_DIR/reader" && uv sync && cd "$SCRIPT_DIR"

# Build memory_cli
cd "$SCRIPT_DIR/memory_cli" && cargo build && cd "$SCRIPT_DIR"

# Use project venv's python; run workflows from reader/src
PYTHON="${SCRIPT_DIR}/reader/.venv/bin/python"
READER_SRC="${SCRIPT_DIR}/reader/src"
# Config paths relative to reader/src
CONFIG_HF_DATA="reader/pipelines/hf_data/config/hf-data.yaml"
CONFIG_REPORT="reader/pipelines/report_generation/config/report.yaml"
CONFIG_SIGNATURE="reader/pipelines/report_signature_check/config/report_signature_check.yaml"
CONFIG_RENDER="reader/pipelines/render_report/config/render_report.yaml"

case "${1:-}" in
    hf-data)
        cd "$READER_SRC" && "$PYTHON" -m reader hf-data --config "$CONFIG_HF_DATA" && cd - > /dev/null
        ;;
    generate-report)
        cd "$READER_SRC" && "$PYTHON" -m reader generate-report --config "$CONFIG_REPORT" && cd - > /dev/null
        ;;
    check-report-signature)
        if [[ -z "${2:-}" ]]; then
            echo "Error: report-file required for check-report-signature" >&2
            usage
            exit 1
        fi
        REPORT_FILE="$2"
        [[ "$REPORT_FILE" == /* ]] || REPORT_FILE="$SCRIPT_DIR/$REPORT_FILE"
        cd "$READER_SRC" && "$PYTHON" -m reader check-report-signature --config "$CONFIG_SIGNATURE" --report-file "$REPORT_FILE" && cd - > /dev/null
        ;;
    render-report)
        if [[ -z "${2:-}" ]]; then
            echo "Error: report-file required for render-report" >&2
            usage
            exit 1
        fi
        REPORT_FILE="$2"
        [[ "$REPORT_FILE" == /* ]] || REPORT_FILE="$SCRIPT_DIR/$REPORT_FILE"
        cd "$READER_SRC" && "$PYTHON" -m reader render-report --config "$CONFIG_RENDER" --report-file "$REPORT_FILE" && cd - > /dev/null
        ;;
    help|--help|-h|"")
        usage
        ;;
    *)
        echo "Error: unknown workflow '${1}'" >&2
        usage
        exit 1
        ;;
esac
