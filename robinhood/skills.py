"""
robinhood.skills — Skill-based prompt synthesis

Lets users define *what they want the model to be good at* in a structured
skill file, then uses Claude to generate diverse, targeted prompts that
exercise those skills.

A skill file is a JSON or YAML document describing one or more skills:

    {
        "name": "Medical Chart Summarizer",
        "description": "Summarize patient charts into structured clinical notes",
        "system_prompt": "You are a clinical documentation specialist...",
        "skills": [
            {
                "name": "extract_diagnoses",
                "description": "Pull all diagnoses from unstructured chart text",
                "difficulty_levels": ["straightforward", "ambiguous", "adversarial"],
                "example_inputs": ["Patient presents with..."],
                "example_outputs": ["Diagnoses: 1) ..."],
                "constraints": ["Only include explicitly stated diagnoses"]
            },
            ...
        ]
    }

The synthesizer reads this file and calls Claude to generate N prompts per
skill, targeting each difficulty level, producing a prompt set that
comprehensively covers the capability surface the user cares about.
"""

import json
import os
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Skill:
    """A single skill the model should be good at."""
    name: str
    description: str
    difficulty_levels: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    example_inputs: List[str] = field(default_factory=list)
    example_outputs: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SkillSet:
    """
    A complete skill definition loaded from a skill file.

    The top-level fields describe the overall task, while the ``skills``
    list describes individual capabilities to target.
    """
    name: str
    description: str
    skills: List[Skill]
    system_prompt: Optional[str] = None
    domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_file(cls, path: str) -> "SkillSet":
        """Load a SkillSet from a JSON or YAML file."""
        with open(path) as f:
            raw = f.read()

        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                data = yaml.safe_load(raw)
            except ImportError:
                raise ImportError("PyYAML is required to load .yaml skill files: pip install pyyaml")
        else:
            data = json.loads(raw)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillSet":
        skills = [Skill.from_dict(s) if isinstance(s, dict) else s for s in d.get("skills", [])]
        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            skills=skills,
            system_prompt=d.get("system_prompt"),
            domain=d.get("domain"),
            metadata=d.get("metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "domain": self.domain,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "difficulty_levels": s.difficulty_levels,
                    "example_inputs": s.example_inputs,
                    "example_outputs": s.example_outputs,
                    "constraints": s.constraints,
                    "tags": s.tags,
                }
                for s in self.skills
            ],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Prompt synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """\
You are generating training prompts for a language model that needs to be \
excellent at a specific skill.

## Task Context

Overall task: {task_name}
Task description: {task_description}
{domain_line}
{system_prompt_line}

## Target Skill

Skill: {skill_name}
Description: {skill_description}
Difficulty level: {difficulty}
{constraints_block}
{examples_block}

## Your Job

Generate {n} diverse, realistic user prompts that would test this skill at \
the "{difficulty}" difficulty level. Each prompt should be something a real \
user might send to this model.

Requirements:
- Each prompt must be self-contained (no references to prior context)
- Vary the phrasing, domain specifics, and complexity within the difficulty level
- Make prompts realistic — they should feel like actual user requests, not synthetic tests
- For harder difficulty levels, include edge cases, ambiguity, or adversarial elements
- Do NOT include the expected answer — only the user prompt

Return ONLY a JSON array of strings, one per prompt. No other text.\
"""


class SkillPromptSynthesizer:
    """
    Generates targeted prompts from a SkillSet using any LLM provider.

    Usage::

        synth = SkillPromptSynthesizer(provider="anthropic")
        prompts = synth.synthesize(skill_set, prompts_per_skill=20)

        synth = SkillPromptSynthesizer(provider="openai", api_key="sk-...")
        synth = SkillPromptSynthesizer(provider="openrouter")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ):
        from robinhood.providers import LLMClient
        self._client = LLMClient(provider=provider, api_key=api_key, model=model)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_synthesis_prompt(
        self,
        skill_set: SkillSet,
        skill: Skill,
        difficulty: str,
        n: int,
    ) -> str:
        domain_line = f"Domain: {skill_set.domain}" if skill_set.domain else ""
        system_prompt_line = (
            f"System prompt the model will see: \"{skill_set.system_prompt}\""
            if skill_set.system_prompt
            else ""
        )

        constraints_block = ""
        if skill.constraints:
            constraints_block = "Constraints the model must follow:\n" + "\n".join(
                f"- {c}" for c in skill.constraints
            )

        examples_block = ""
        if skill.example_inputs:
            examples_block = "Example inputs (for reference, generate NEW ones):\n"
            for i, ex_in in enumerate(skill.example_inputs):
                examples_block += f"  Input {i+1}: {ex_in}\n"
                if i < len(skill.example_outputs):
                    examples_block += f"  Output {i+1}: {skill.example_outputs[i]}\n"

        return SYNTHESIS_PROMPT.format(
            task_name=skill_set.name,
            task_description=skill_set.description,
            domain_line=domain_line,
            system_prompt_line=system_prompt_line,
            skill_name=skill.name,
            skill_description=skill.description,
            difficulty=difficulty,
            constraints_block=constraints_block,
            examples_block=examples_block,
            n=n,
        )

    def _call_llm(self, prompt: str) -> str:
        resp = self._client.complete(
            user_message=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.content

    def _parse_prompt_list(self, raw: str) -> List[str]:
        """Extract a JSON array of strings from Claude's response."""
        text = raw.strip()
        # Find the JSON array in the response
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            print(f"[SKILLS] Warning: Could not find JSON array in response, trying line-based parse")
            return [line.strip().strip('"').strip("'") for line in text.split("\n") if line.strip()]

        try:
            items = json.loads(text[start:end + 1])
            return [str(item) for item in items if item]
        except json.JSONDecodeError as e:
            print(f"[SKILLS] Warning: JSON parse failed ({e}), trying line-based parse")
            return [line.strip().strip('"').strip("'") for line in text.split("\n") if line.strip()]

    def synthesize(
        self,
        skill_set: SkillSet,
        prompts_per_skill: int = 20,
        save_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate prompts targeting every skill at every difficulty level.

        Args:
            skill_set: The SkillSet defining the target capabilities.
            prompts_per_skill: Total prompts per skill (divided across difficulty levels).
            save_path: Optional path to save generated prompts.

        Returns:
            List of prompt dicts ready for the robinhood pipeline:
                {"user_message": str, "system_prompt": str|None, "category": str}
        """
        all_prompts: List[Dict[str, Any]] = []

        for skill in skill_set.skills:
            n_levels = len(skill.difficulty_levels)
            per_level = max(1, prompts_per_skill // n_levels)

            print(
                f"[SKILLS] Synthesizing prompts for skill '{skill.name}' "
                f"({n_levels} levels x {per_level} each)..."
            )

            for difficulty in skill.difficulty_levels:
                synthesis_prompt = self._build_synthesis_prompt(
                    skill_set, skill, difficulty, per_level
                )

                try:
                    raw = self._call_llm(synthesis_prompt)
                    generated = self._parse_prompt_list(raw)

                    for user_msg in generated:
                        all_prompts.append({
                            "user_message": user_msg,
                            "system_prompt": skill_set.system_prompt,
                            "category": f"{skill.name}:{difficulty}",
                            "skill": skill.name,
                            "difficulty": difficulty,
                        })

                    print(
                        f"  [{skill.name}/{difficulty}] "
                        f"Generated {len(generated)} prompts"
                    )

                except Exception as e:
                    print(
                        f"  [{skill.name}/{difficulty}] "
                        f"Error: {type(e).__name__}: {e}"
                    )

        print(
            f"[SKILLS] Synthesis complete: {len(all_prompts)} total prompts "
            f"across {len(skill_set.skills)} skills"
        )

        if save_path:
            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True,
            )
            with open(save_path, "w") as f:
                json.dump(all_prompts, f, indent=2)
            print(f"[SKILLS] Saved prompts to {save_path}")

        return all_prompts

    def synthesize_from_file(
        self,
        skill_file: str,
        prompts_per_skill: int = 20,
        save_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience: load a skill file and synthesize prompts in one call."""
        skill_set = SkillSet.from_file(skill_file)
        return self.synthesize(skill_set, prompts_per_skill, save_path)


def generate_prompts_from_skills(
    skill_file: str,
    prompts_per_skill: int = 20,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    synthesis_model: str = "claude-sonnet-4-20250514",
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Top-level convenience function: load a skill file -> synthesize prompts.

    This is the main entry point when using skill files with the pipeline.
    Works with any provider (anthropic, openai, openrouter).
    """
    synth = SkillPromptSynthesizer(
        provider=provider, api_key=api_key, model=synthesis_model,
    )
    return synth.synthesize_from_file(skill_file, prompts_per_skill, save_path)
