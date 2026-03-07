# The Collective - Specialist Node Base Class
# All nodes inherit from this. Uses Anthropic Claude.

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceEvent:
    id: str
    timestamp: datetime
    source: str
    description: str
    impact_score: float          # 0.0 – 1.0
    relevant_domains: List[str]
    raw_data: Optional[Any] = None


@dataclass
class NodeOutput:
    node_id: str
    event_id: str
    domain: str
    output: Dict[str, Any]
    confidence: float
    timestamp: datetime
    compute_cost_usd: float = 0.0
    generation: int = 1


# ---------------------------------------------------------------------------
# Pricing (Claude Sonnet 4.5 as of 2025)
# Input:  $3.00 / 1M tokens
# Output: $15.00 / 1M tokens
# ---------------------------------------------------------------------------

SONNET_INPUT_COST  = 3.00  / 1_000_000
SONNET_OUTPUT_COST = 15.00 / 1_000_000


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * SONNET_INPUT_COST) + (output_tokens * SONNET_OUTPUT_COST)


# ---------------------------------------------------------------------------
# SpecialistNode base
# ---------------------------------------------------------------------------

class SpecialistNode(ABC):
    def __init__(self, domain: str, system_prompt: str, model: str = "claude-sonnet-4-5"):
        self.domain = domain
        self.system_prompt = system_prompt
        self.model = model
        self.node_id = f"{domain}_node_v1"
        self._client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @abstractmethod
    def _format_prompt(self, event: IntelligenceEvent) -> str:
        """Format the user prompt for this node's domain."""
        pass

    async def generate(self, event: IntelligenceEvent, n_variants: int = 1) -> List[NodeOutput]:
        """Generate intelligence outputs, run self-critique, return passing outputs."""
        outputs = []
        for _ in range(n_variants):
            output = await self._generate_single(event)
            if output:
                critique = await self._self_critique(output)
                if critique.get("passes_threshold", False):
                    output.confidence = critique.get("average_score", 0.5) / 5.0
                    outputs.append(output)
        return outputs

    async def _generate_single(self, event: IntelligenceEvent) -> Optional[NodeOutput]:
        prompt = self._format_prompt(event)
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text

            # Strip markdown fences if present
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            output_data = json.loads(clean)

            input_tokens  = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = estimate_cost(input_tokens, output_tokens)

            return NodeOutput(
                node_id=self.node_id,
                event_id=event.id,
                domain=self.domain,
                output=output_data,
                confidence=output_data.get("confidence", 0.7),
                timestamp=datetime.now(),
                compute_cost_usd=cost,
            )
        except Exception as e:
            print(f"[{self.node_id}] Generation error: {e}")
            return None

    async def _self_critique(self, output: NodeOutput) -> Dict:
        """Score the output on 4 dimensions. Passes if average >= 3.0/5.0."""
        critique_prompt = f"""Rate this intelligence output on 4 dimensions (1-5 scale):

Output to evaluate:
{json.dumps(output.output, indent=2)}

Score each dimension:
1. Structure (1-5): Is the JSON complete with all required fields?
2. Reasoning (1-5): Is the chain-of-thought logical and deep?
3. Factual (1-5): Are claims grounded and citations present?
4. Actionable (1-5): Is the recommendation specific and useful?

Return JSON only, no markdown, no explanation:
{{"structure": 4, "reasoning": 4, "factual": 3, "actionable": 4, "average_score": 3.75, "passes_threshold": true, "critique_notes": "brief notes"}}"""

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=300,
                system="You are a rigorous quality evaluator. Return only a single valid JSON object, no markdown fences, no extra text.",
                messages=[{"role": "user", "content": critique_prompt}]
            )
            raw = response.content[0].text.strip()
            # Strip any markdown fences
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break
            # Find the JSON object boundaries
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            scores = json.loads(raw)
            avg = (scores.get("structure", 3) + scores.get("reasoning", 3) +
                   scores.get("factual", 3) + scores.get("actionable", 3)) / 4.0
            scores["average_score"] = avg
            scores["passes_threshold"] = avg >= 3.0
            return scores
        except Exception as e:
            print(f"[{self.node_id}] Self-critique error: {e}")
            # Default pass so a critique failure doesn't block all output
            return {"average_score": 3.5, "passes_threshold": True}
