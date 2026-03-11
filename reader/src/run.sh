uv sync
cd ../../memory_cli && cargo build && cd -
# python -m reader hf-data --config reader/pipelines/hf_data/config/hf-data.yaml
python -m reader generate-report --config reader/pipelines/report_generation/config/report.yaml
# python -m reader hf-data --config reader/pipelines/hf_data/config/hf-data.yaml