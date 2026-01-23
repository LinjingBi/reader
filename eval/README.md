# Reader Evaluation Pipeline

Evaluation pipeline for comparing P0 vs P-rich packing strategies for LLM cluster enrichment.

## Setup

1. Install dependencies:
```bash
cd eval
pip install -e .
# or
uv pip install -e .
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
# or for Gemini:
export GEMINI_API_KEY="your-key-here"
```

3. Prepare input data:
   - Place monthly clustering JSON files in `data/monthly_clusters/`
   - Format should match `memory_cli/examples/fresh_paper_payload.json`

## Usage

Run evaluation for a specific month:

```bash
python run_eval.py --config configs/eval.yaml --month 2025-01
```

## Configuration

Edit `configs/eval.yaml` to configure:
- Dataset months and input directory
- Packer variants (P0, P_RICH)
- Prompt templates
- LLM provider (openai/gemini), model, temperature, max_tokens
- Judge schema path

## Output

Results are saved to `results/{run_id}/`:
- `run.log`: Structured log file
- `model_output.json`: Parsed LLM output
- `judge_result.json`: Heuristic judge results

## Components

- **packers/**: Transform cluster data to LLM input format
- **judges/**: Validate LLM outputs (heuristic + future LLM judge)
- **utils/**: LLM client, logger, template loader
- **prompts/**: Prompt templates
- **schemas/**: JSON schemas for input/output validation

