# robinhood — Reasoning Trace Distillation Library

## Background

`robinhood` is a library for creating finetuning datasets from LLM reasoning traces.  It implements the research-grade distillation techniques documented in papers from DeepSeek, MiniMax, OpenThoughts, Light-R1, and REDI — collecting reasoning traces from teacher models and converting them into high-quality training data for student models.

Supports any provider (Anthropic, OpenAI, OpenRouter) when permitted by their terms of service.  Users are responsible for ensuring they have the rights and permissions to use model outputs for training purposes.

---

## Installation

```bash
# Install from the repo
pip install -e robinhood/

# Or just use it directly (zero-install, only needs `anthropic` package)
pip install anthropic
```

---

## Quick Start

### As a library

```python
import robinhood

# Generate prompts from built-in templates
prompts = robinhood.generate_prompts(samples_per_category=10)

# Collect reasoning traces from Claude
collector = robinhood.ClaudeTraceCollector(
    config=robinhood.CollectionConfig(
        model="claude-sonnet-4-20250514",
        thinking=robinhood.ThinkingConfig(budget_tokens=10000),
    )
)
traces = collector.collect_traces_sync(prompts, save_path="traces.json")

# Format into a training dataset
formatter = robinhood.DatasetFormatter(
    config=robinhood.FormatterConfig(format_mode="thinking_and_output")
)
dataset = formatter.format_traces([t.to_dict() for t in traces])
formatter.export_for_platform(dataset, output_dir="./my_dataset")
```

### As a CLI

```bash
# Quick test run
python -m robinhood --model claude-sonnet-4-20250514 --samples-per-category 5

# Full production run
python -m robinhood \
    --model claude-sonnet-4-20250514 \
    --samples-per-category 500 \
    --thinking-budget 16000 \
    --format thinking_and_output \
    --output-dir ./robinhood_output

# Reformat existing traces without re-querying Claude
python -m robinhood \
    --reformat ./robinhood_output/traces/raw_traces.json \
    --format output_only \
    --output-dir ./output_only_dataset
```

### As a pipeline

```python
from robinhood import RobinhoodPipeline, PipelineConfig

config = PipelineConfig(
    model="claude-sonnet-4-20250514",
    samples_per_category=100,
    thinking_budget=16000,
    format_mode="thinking_and_output",
    output_dir="./robinhood_output",
)
pipeline = RobinhoodPipeline(config=config)
stats = pipeline.run()
```

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Prompt Generation│────>│ Trace Collection  │────>│ Quality Filter   │
│                 │     │ (Claude API +     │     │                 │
│ - Templates     │     │  Extended Thinking)│     │ - Min thinking  │
│ - Custom files  │     │                  │     │ - Min output    │
│ - Category mix  │     │ Captures:        │     │ - Dedup         │
└─────────────────┘     │ - Thinking blocks │     └────────┬────────┘
                        │ - Text output    │              │
                        │ - Token counts   │              ▼
                        │ - Latency        │     ┌─────────────────┐
                        └──────────────────┘     │ Dataset Formatter│
                                                 │                 │
                                                 │ Modes:          │
                                                 │ 1. think+output │
                                                 │ 2. output only  │
                                                 │ 3. RGT strategy │
                                                 │ 4. multi-turn   │
                                                 └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ Export           │
                                                 │                 │
                                                 │ - Platform SFT  │
                                                 │ - HuggingFace   │
                                                 │ - Train/Val     │
                                                 └─────────────────┘
```

---

## Prompt Design

Effective distillation datasets require prompts that are:

1. **Systematically diverse** — covering reasoning, code, analysis, creative writing, domain knowledge
2. **High complexity** — targeting tasks where the reasoning trace adds substantial value
3. **Structured** — using templates that vary parameters to create large volumes

The built-in prompt library covers 7 capability categories:

| Category | Focus | Why It Matters |
|----------|-------|----------------|
| `reasoning_math` | Mathematical problem-solving, logic puzzles | Tests formal reasoning that benefits most from chain-of-thought |
| `code_generation` | Algorithms, debugging, system design | Complex multi-step tasks with verifiable outputs |
| `analysis_comprehension` | NER, summarization, classification | Information extraction requiring careful reading |
| `creative_language` | Writing, translation, style transfer | Tests language generation quality |
| `domain_knowledge` | Science, law, medicine, finance | Expert-level knowledge application |
| `instruction_following` | Constrained generation, formatting | Tests ability to follow complex specifications |
| `multi_step_reasoning` | Bayesian inference, combinatorics | Problems requiring extended reasoning chains |

---

## Trace Collection

The collector queries Claude with **extended thinking enabled**, capturing both the internal reasoning trace and the final output.

**What gets captured for each prompt:**

| Field | Description |
|-------|-------------|
| `thinking_text` | Claude's internal reasoning (from `thinking` content blocks) |
| `output_text` | The final answer (from `text` content blocks) |
| `thinking_tokens` | Tokens spent on reasoning |
| `output_tokens` | Tokens in the final answer |
| `latency_seconds` | Wall-clock time for the request |
| `model` | Exact model version used |

**Model selection guidance:**

| Model | Thinking Quality | Cost | Best For |
|-------|-----------------|------|----------|
| `claude-sonnet-4-20250514` | Good | Medium | General distillation, balanced cost/quality |
| `claude-opus-4-20250514` | Excellent | High | Math, code, complex reasoning |
| `claude-haiku-4-5-20250514` | Moderate | Low | High-volume data collection, simpler tasks |

---

## Dataset Format Modes

### Mode 1: `thinking_and_output` (Recommended)

```
Target: <think>
[Claude's full reasoning trace]
</think>
[Claude's final answer]
```

The student model learns to both *reason like Claude* and *produce Claude-quality outputs*.

### Mode 2: `output_only` (Traditional distillation)

```
Target: [Claude's final answer]
```

Discards the reasoning trace. Simpler but less effective for reasoning-heavy tasks.

### Mode 3: `reasoning_augmented` (RGT-compatible)

```
Target: <STRATEGY>[condensed strategy]</STRATEGY>[Claude's final answer]
```

Extracts the key strategic insight from the reasoning trace and prepends it as a compact hint.

### Mode 4: `multi_turn` (Explicit reasoning turn)

```
Input: [..., {role: "assistant", content: "[Internal reasoning]\n..."}, {role: "user", content: "Provide final answer"}]
Target: [Claude's final answer]
```

Makes the reasoning an explicit conversation turn.

| Mode | Training Cost | Reasoning Quality | Output Quality | Complexity |
|------|--------------|-------------------|----------------|------------|
| `thinking_and_output` | Highest (long targets) | Best | Best | Simple |
| `output_only` | Lowest | None transferred | Good | Simplest |
| `reasoning_augmented` | Medium | Good (condensed) | Good | Medium |
| `multi_turn` | Medium-High | Good (explicit) | Good | Complex |

---

## Config File

```json
{
    "output_dir": "./robinhood_output",
    "run_name": "sonnet_full_run",
    "model": "claude-sonnet-4-20250514",
    "thinking_budget": 16000,
    "samples_per_category": 200,
    "format_mode": "thinking_and_output",
    "categories": ["reasoning_math", "code_generation", "multi_step_reasoning"],
    "rate_limit_rpm": 100,
    "max_concurrent": 20,
    "train_split_ratio": 0.9,
    "export_format": "platform"
}
```

```bash
python -m robinhood --config my_config.json
```

---

## Integration with Platform SFT Pipeline

The exported `train.json` and `val.json` files are directly compatible with the platform's SFT training pipeline. Each sample has the format:

```json
{
    "input": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "output": {
        "choices": [{
            "message": {
                "content": "<think>\n...\n</think>\n...",
                "role": "assistant"
            }
        }]
    }
}
```

---

## Scaling Considerations

| Scale | Prompts | Est. Cost (Sonnet) | Est. Time (50 RPM) |
|-------|---------|---------------------|---------------------|
| Test | 50 | ~$5 | ~1 min |
| Small | 1,000 | ~$100 | ~20 min |
| Medium | 10,000 | ~$1,000 | ~3.3 hours |
| Large | 100,000 | ~$10,000 | ~33 hours |

**Cost optimization strategies:**
- Use Haiku for simpler tasks, Sonnet/Opus for complex reasoning
- Lower thinking budget for straightforward tasks
- Use prompt caching to reduce input token costs

---

## Compliance & Responsible Use

Key considerations:

1. **Provider terms of service**: Before collecting traces at scale, confirm that your intended use is permitted by the provider's terms. Some providers restrict using outputs to train competing models.
2. **Safety alignment**: Models trained via distillation may not inherit the teacher's safety training. Evaluate your student model's safety properties independently.
3. **Rate limits**: Always respect provider rate limits. Use ``--compliance`` mode for conservative defaults.

---

## API Reference

| Module | Key Exports |
|--------|-------------|
| `robinhood` | `ClaudeTraceCollector`, `DatasetFormatter`, `RobinhoodPipeline`, `generate_prompts` |
| `robinhood.trace_collector` | `ClaudeTraceCollector`, `CollectionConfig`, `ThinkingConfig`, `CollectedTrace` |
| `robinhood.dataset_formatter` | `DatasetFormatter`, `FormatterConfig` |
| `robinhood.prompt_sources` | `generate_prompts`, `generate_prompts_from_file`, `PROMPT_TEMPLATES` |
| `robinhood.pipeline` | `RobinhoodPipeline`, `PipelineConfig`, `run_from_existing_traces`, `main` |
