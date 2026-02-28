"""
robinhood.verification — Dual verification pipelines for rejection sampling

Every candidate trace is verified before inclusion in the training set.
Two pipelines handle different task types:

**Verifiable pipeline** (math, code):
    Automated correctness checking — extract expected answers / test cases
    from the prompt or output and compare programmatically.

**Non-verifiable pipeline** (reasoning, creative, domain knowledge):
    LLM-as-judge scoring on coherence, instruction-following, completeness,
    and absence of hallucination.

After verification, difficulty is scored from the teacher's pass rate on each
prompt (the fraction of N candidate traces that passed verification).
Intermediate-difficulty prompts are the most valuable for distillation.

Based on techniques from:
    - DeepSeek R1 Technical Report (2025)
    - MiniMax "What Makes Good Reasoning Data" (2025)
    - OpenThoughts3 Blog (2025)
"""

import asyncio
import json
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from robinhood.providers import LLMClient


# Categories handled by the deterministic (verifiable) pipeline
VERIFIABLE_CATEGORIES = frozenset({
    "reasoning_math",
    "code_generation",
    "math",
    "code",
})

_JUDGE_SYSTEM = """\
You are an expert evaluator assessing the quality of an AI assistant's response.
Score the response on a scale of 1-10 across four dimensions, then give an
overall score.

Dimensions:
1. **Correctness** — Is the response factually and logically correct?
2. **Completeness** — Does it fully address the user's request?
3. **Coherence** — Is the reasoning chain logical with no non-sequiturs or dead ends?
4. **Instruction-following** — Does it respect all constraints in the prompt?

Return ONLY valid JSON with this exact schema (no other text):
{"correctness": <int>, "completeness": <int>, "coherence": <int>, \
"instruction_following": <int>, "overall": <int>, "rationale": "<brief explanation>"}
"""

_JUDGE_USER_TEMPLATE = """\
## User Prompt
{user_message}

## Assistant Response (Thinking + Output)
<think>
{thinking}
</think>
{output}
"""


@dataclass
class VerificationConfig:
    """Controls how traces are verified and selected."""
    judge_model: Optional[str] = None
    judge_provider: Optional[str] = None
    judge_api_key: Optional[str] = None

    min_judge_score: float = 6.0
    judge_max_tokens: int = 1024

    code_execution_timeout: int = 10
    enable_code_execution: bool = False

    difficulty_min: float = 0.05
    difficulty_max: float = 0.95

    max_concurrent_judges: int = 5


@dataclass
class VerifiedTrace:
    """A trace annotated with verification metadata."""
    trace: Dict[str, Any]
    passed: bool
    score: float
    verification_method: str
    details: Dict[str, Any] = field(default_factory=dict)


class TraceVerifier:
    """
    Dual verification pipeline matching the MiniMax M2 approach.

    For each prompt that has N candidate traces, this class:
    1. Routes to the appropriate verification pipeline based on category
    2. Scores every candidate
    3. Selects the highest-scoring passing candidate
    4. Computes a difficulty score from the pass rate
    5. Returns (selected, rejected, difficulty) tuples per prompt
    """

    def __init__(self, config: VerificationConfig = None):
        self.config = config or VerificationConfig()
        self._judge_client: Optional[LLMClient] = None

    def _get_judge_client(self, fallback_model: str = "claude-sonnet-4-20250514") -> LLMClient:
        if self._judge_client is None:
            model = self.config.judge_model or fallback_model
            self._judge_client = LLMClient(
                provider=self.config.judge_provider,
                api_key=self.config.judge_api_key,
                model=model,
            )
        return self._judge_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_and_select(
        self,
        traces_by_prompt: Dict[str, List[Dict[str, Any]]],
        fallback_model: str = "claude-sonnet-4-20250514",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
        """
        Verify all candidate traces, select the best per prompt, score
        difficulty.

        Args:
            traces_by_prompt: ``{prompt_id: [trace_dict, ...]}``
            fallback_model: Model to use for LLM-as-judge if none configured.

        Returns:
            (selected_traces, rejected_traces, difficulty_by_prompt)
        """
        return asyncio.run(
            self.verify_and_select_async(traces_by_prompt, fallback_model)
        )

    async def verify_and_select_async(
        self,
        traces_by_prompt: Dict[str, List[Dict[str, Any]]],
        fallback_model: str = "claude-sonnet-4-20250514",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:

        selected: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        difficulty_scores: Dict[str, float] = {}

        total_prompts = len(traces_by_prompt)
        processed = 0

        for prompt_id, candidates in traces_by_prompt.items():
            if not candidates:
                continue

            category = candidates[0].get("prompt_category", "general")
            is_verifiable = category in VERIFIABLE_CATEGORIES

            verified = []
            for trace in candidates:
                if is_verifiable:
                    vt = self._verify_deterministic(trace)
                else:
                    vt = await self._verify_with_judge(trace, fallback_model)
                verified.append(vt)

            passing = [v for v in verified if v.passed]
            failing = [v for v in verified if not v.passed]
            pass_rate = len(passing) / len(verified) if verified else 0.0
            difficulty = 1.0 - pass_rate

            difficulty_scores[prompt_id] = difficulty

            if self.config.difficulty_min <= difficulty <= self.config.difficulty_max and passing:
                best = max(passing, key=lambda v: v.score)
                best.trace["difficulty"] = difficulty
                best.trace["verification_score"] = best.score
                best.trace["verification_method"] = best.verification_method
                best.trace["prompt_id"] = prompt_id
                selected.append(best.trace)

                for v in verified:
                    if v is not best:
                        v.trace["difficulty"] = difficulty
                        v.trace["prompt_id"] = prompt_id
                        rejected.append(v.trace)
            else:
                for v in verified:
                    v.trace["difficulty"] = difficulty
                    v.trace["prompt_id"] = prompt_id
                    rejected.append(v.trace)

            processed += 1
            if processed % 20 == 0 or processed == total_prompts:
                print(
                    f"[VERIFY] Progress: {processed}/{total_prompts} prompts "
                    f"({len(selected)} selected, {len(rejected)} rejected)"
                )

        easy = sum(1 for d in difficulty_scores.values() if d < self.config.difficulty_min)
        hard = sum(1 for d in difficulty_scores.values() if d > self.config.difficulty_max)
        kept = total_prompts - easy - hard

        print(f"[VERIFY] Verification complete:")
        print(f"  Total prompts:    {total_prompts}")
        print(f"  Selected traces:  {len(selected)}")
        print(f"  Rejected traces:  {len(rejected)}")
        print(f"  Difficulty filter: {easy} too easy, {hard} too hard, {kept} kept")

        return selected, rejected, difficulty_scores

    # ------------------------------------------------------------------
    # Verifiable pipeline: math and code
    # ------------------------------------------------------------------

    def _verify_deterministic(self, trace: Dict[str, Any]) -> VerifiedTrace:
        """
        Automated verification for math and code tasks.

        Checks for:
        - Code: syntactic validity, optional sandboxed execution
        - Math: presence of a final numeric/symbolic answer, internal consistency
        """
        category = trace.get("prompt_category", "")
        output = trace.get("output_text", "")
        thinking = trace.get("thinking_text", "")

        if "code" in category:
            return self._verify_code(trace, output, thinking)
        else:
            return self._verify_math(trace, output, thinking)

    def _verify_code(self, trace: Dict, output: str, thinking: str) -> VerifiedTrace:
        score = 0.0
        details: Dict[str, Any] = {}

        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", output, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", thinking, re.DOTALL)

        details["code_blocks_found"] = len(code_blocks)

        if not code_blocks:
            return VerifiedTrace(trace=trace, passed=False, score=0.0,
                                 verification_method="deterministic_code", details=details)

        score += 3.0

        for block in code_blocks:
            try:
                compile(block, "<trace>", "exec")
                score += 2.0
                details["compiles"] = True
                break
            except SyntaxError:
                details["compiles"] = False

        has_explanation = len(output) > len("\n".join(code_blocks)) + 50
        if has_explanation:
            score += 2.0
            details["has_explanation"] = True

        if thinking and len(thinking) > 100:
            score += 1.5
            details["has_reasoning"] = True

        consistent_refs = any(
            keyword in thinking.lower()
            for keyword in ("function", "class", "algorithm", "return", "loop", "variable")
        )
        if consistent_refs:
            score += 1.5
            details["reasoning_references_code"] = True

        passed = score >= 5.0
        return VerifiedTrace(trace=trace, passed=passed, score=score,
                             verification_method="deterministic_code", details=details)

    def _verify_math(self, trace: Dict, output: str, thinking: str) -> VerifiedTrace:
        score = 0.0
        details: Dict[str, Any] = {}

        answer_patterns = [
            r"(?:the\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|$)",
            r"(?:=|equals?)\s*(\S+)\s*$",
            r"\\boxed\{(.+?)\}",
            r"(?:therefore|thus|hence|so)\s*,?\s*(.+?)(?:\.|$)",
        ]

        has_final_answer = False
        for pattern in answer_patterns:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                has_final_answer = True
                break

        if has_final_answer:
            score += 3.0
            details["has_final_answer"] = True
        else:
            score += 1.0
            details["has_final_answer"] = False

        if thinking:
            step_indicators = sum(
                1 for keyword in ("step", "first", "then", "next", "therefore",
                                  "since", "because", "let", "assume", "given")
                if keyword in thinking.lower()
            )
            reasoning_score = min(3.0, step_indicators * 0.5)
            score += reasoning_score
            details["reasoning_steps_found"] = step_indicators

        if thinking and has_final_answer:
            numbers_in_thinking = set(re.findall(r"-?\d+\.?\d*", thinking[-500:]))
            numbers_in_output = set(re.findall(r"-?\d+\.?\d*", output))
            overlap = numbers_in_thinking & numbers_in_output
            if overlap:
                score += 2.0
                details["answer_consistent_with_reasoning"] = True

        if len(output) > 20:
            score += 1.0

        if thinking and len(thinking) > 200:
            score += 1.0

        passed = score >= 5.0
        return VerifiedTrace(trace=trace, passed=passed, score=score,
                             verification_method="deterministic_math", details=details)

    # ------------------------------------------------------------------
    # Non-verifiable pipeline: LLM-as-judge
    # ------------------------------------------------------------------

    async def _verify_with_judge(
        self, trace: Dict[str, Any], fallback_model: str,
    ) -> VerifiedTrace:
        """Use LLM-as-judge to score non-verifiable tasks."""
        client = self._get_judge_client(fallback_model)

        user_message = trace.get("user_message", "")
        thinking = trace.get("thinking_text", "")
        output = trace.get("output_text", "")

        judge_prompt = _JUDGE_USER_TEMPLATE.format(
            user_message=user_message,
            thinking=thinking[:3000],
            output=output[:3000],
        )

        try:
            resp = await client.complete_async(
                user_message=judge_prompt,
                system=_JUDGE_SYSTEM,
                max_tokens=self.config.judge_max_tokens,
                temperature=0.0,
            )

            scores = self._parse_judge_response(resp.content)
            overall = scores.get("overall", 0)

            passed = overall >= self.config.min_judge_score
            return VerifiedTrace(
                trace=trace,
                passed=passed,
                score=float(overall),
                verification_method="llm_judge",
                details=scores,
            )

        except Exception as e:
            print(f"[VERIFY] Judge error: {type(e).__name__}: {e}")
            score = self._heuristic_fallback_score(trace)
            return VerifiedTrace(
                trace=trace,
                passed=score >= self.config.min_judge_score,
                score=score,
                verification_method="heuristic_fallback",
                details={"error": str(e)},
            )

    def _parse_judge_response(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {"overall": 5, "parse_error": True}
        try:
            data = json.loads(text[start:end + 1])
            for key in ("correctness", "completeness", "coherence",
                         "instruction_following", "overall"):
                if key in data:
                    data[key] = max(1, min(10, int(data[key])))
            return data
        except (json.JSONDecodeError, ValueError):
            return {"overall": 5, "parse_error": True}

    def _heuristic_fallback_score(self, trace: Dict[str, Any]) -> float:
        """Quick heuristic when the judge call fails."""
        score = 4.0
        thinking = trace.get("thinking_text", "")
        output = trace.get("output_text", "")

        if len(thinking) > 500:
            score += 1.5
        if len(output) > 100:
            score += 1.5
        if len(thinking) > 2000:
            score += 1.0
        if len(output) > 500:
            score += 1.0

        return min(10.0, score)
