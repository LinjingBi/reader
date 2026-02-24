uv sync
cd ../../memory_cli && cargo build && cd -
# python -m reader --config ../configs/reader.yaml
python -m reader --config reader/pipelines/hf_data/config/hf-data.yaml