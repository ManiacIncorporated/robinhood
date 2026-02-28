"""
robinhood — Reasoning trace distillation library

Toolkit for collecting reasoning traces from LLM teacher models and
converting them into high-quality finetuning datasets, then training
your own student models on them.

Implements research-grade distillation techniques (rejection sampling,
dual verification, curriculum training, contrastive refinement) as a
clean, extensible Python library.

Supports any provider (Anthropic, OpenAI, OpenRouter) when permitted
by their terms of service.  Users are responsible for ensuring they
have the rights and permissions to use model outputs for training.
"""

__version__ = "0.2.0"

from robinhood.providers import (
    LLMClient,
    LLMResponse,
    detect_provider,
    resolve_api_key,
)
from robinhood.trace_collector import (
    TraceCollector,
    ClaudeTraceCollector,
    CollectionConfig,
    ThinkingConfig,
    CollectedTrace,
)
from robinhood.dataset_formatter import (
    DatasetFormatter,
    FormatterConfig,
)
from robinhood.verification import (
    TraceVerifier,
    VerificationConfig,
    VerifiedTrace,
)
from robinhood.pipeline import (
    RobinhoodPipeline,
    PipelineConfig,
    run_from_existing_traces,
)
from robinhood.prompt_sources import (
    generate_prompts,
    generate_prompts_from_file,
    PROMPT_TEMPLATES,
)
from robinhood.skills import (
    Skill,
    SkillSet,
    SkillPromptSynthesizer,
    generate_prompts_from_skills,
)
from robinhood.trainer import (
    DistillTrainer,
    TrainConfig,
    LoraConfig,
)
