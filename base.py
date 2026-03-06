# The Collective - Specialist Node Base Class
# Agent-to-agent training system

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import openai

@dataclass
class IntelligenceEvent:
    """Trigger event for Node analysis"""
    id: str
    timestamp: datetime
    source: str
    description: str
    impact_score: float  # 0.0 - 1.0
    relevant_domains: List[str]
    raw_data: Optional[Dict] = None

@dataclass
class NodeOutput:
    """Specialist Node output package"""
    node_id: str
    domain: str
    event_id: str
    output: Dict[str, Any]
    chain_of_thought: List[str]
    confidence: float
    citations: List[str]
    generation_time_ms: int
    compute_cost_usd: float
    timestamp: datetime

class SpecialistNode(ABC):
    """Base class for all Specialist Nodes in The Collective"""
    
    def __init__(self, domain: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.domain = domain
        self.system_prompt = system_prompt
        self.model = model
        self.node_id = f"{domain}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Metrics tracking
        self.generations_count = 0
        self.total_compute_cost = 0.0
        self.average_quality_score = 0.0
    
    async def generate(
        self, 
        event: IntelligenceEvent, 
        n_variants: int = 3,
        temperature_range: tuple = (0.3, 0.7)
    ) -> List[NodeOutput]:
        """Generate multiple output variants for self-critique"""
        
        variants = []
        min_temp, max_temp = temperature_range
        temp_step = (max_temp - min_temp) / max(n_variants - 1, 1)
        
        for i in range(n_variants):
            temperature = min_temp + (i * temp_step)
            
            try:
                # Generate raw output
                raw_output = await self._llm_call(
                    system=self.system_prompt,
                    user=self._format_prompt(event),
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                
                # Parse and validate
                parsed = json.loads(raw_output)
                
                # Self-critique loop
                critique = await self._self_critique(parsed)
                
                # Accept if score >= 3.0 (was requiring both boolean and score)
                if critique['average_score'] >= 3.0:
                    variants.append(NodeOutput(
                        node_id=f"{self.node_id}_{i}",
                        domain=self.domain,
                        event_id=event.id,
                        output=parsed,
                        chain_of_thought=parsed.get('chain_of_thought', []),
                        confidence=parsed.get('confidence', 0.5),
                        citations=parsed.get('citations', []),
                        generation_time_ms=critique['time_ms'],
                        compute_cost_usd=critique['cost_usd'],
                        timestamp=datetime.now()
                    ))
                    
            except Exception as e:
                print(f"[{self.domain}] Generation failed for variant {i}: {e}")
                continue
        
        self.generations_count += len(variants)
        return variants
    
    async def _llm_call(
        self, 
        system: str, 
        user: str, 
        temperature: float,
        response_format: Optional[Dict] = None,
        max_tokens: int = 4000
    ) -> str:
        """Call OpenAI API with retry logic"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    
    async def _self_critique(self, output: Dict) -> Dict:
        """Node critiques its own output before submission"""
        
        critique_prompt = f"""
        Review this {self.domain} analysis critically:
        
        {json.dumps(output, indent=2)}
        
        Evaluate on:
        1. Structure completeness (all required fields present)
        2. Reasoning depth (chain-of-thought quality)
        3. Factual claims (verifiable, cited)
        4. Actionability (specific recommendations)
        
        Score each 1-5. Return JSON:
        {{
            "structure_score": X,
            "reasoning_score": X,
            "factual_score": X,
            "actionability_score": X,
            "average_score": X.X,
            "passes_threshold": true/false,
            "issues": ["issue1", "issue2"]
        }}
        """
        
        try:
            critique_response = await self._llm_call(
                system="You are a ruthless quality reviewer. Be harsh and critical.",
                user=critique_prompt,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            critique = json.loads(critique_response)
            
            # Estimate compute cost ($0.0006 per 1K tokens for gpt-4o-mini)
            input_tokens = len(critique_prompt) // 4
            output_tokens = len(critique_response) // 4
            cost_usd = (input_tokens + output_tokens) / 1000 * 0.0006
            
            avg_score = critique.get('average_score', 0)
            
            return {
                'structure_score': critique.get('structure_score', 0),
                'reasoning_score': critique.get('reasoning_score', 0),
                'factual_score': critique.get('factual_score', 0),
                'actionability_score': critique.get('actionability_score', 0),
                'average_score': avg_score,
                'passes_threshold': avg_score >= 3.0,  # Lower threshold, use score directly
                'issues': critique.get('issues', []),
                'time_ms': 0,
                'cost_usd': cost_usd
            }
            
        except Exception as e:
            print(f"[{self.domain}] Self-critique failed: {e}")
            return {
                'average_score': 0,
                'passes_threshold': False,
                'issues': [f'Critique error: {str(e)}'],
                'time_ms': 0,
                'cost_usd': 0
            }
    
    @abstractmethod
    def _format_prompt(self, event: IntelligenceEvent) -> str:
        """Domain-specific prompt formatting - implement in subclasses"""
        pass
    
    def get_metrics(self) -> Dict:
        """Return Node performance metrics"""
        return {
            'node_id': self.node_id,
            'domain': self.domain,
            'generations_count': self.generations_count,
            'total_compute_cost_usd': self.total_compute_cost,
            'average_quality_score': self.average_quality_score
        }


# Example usage / testing
if __name__ == "__main__":
    print("SpecialistNode base class loaded successfully")
    print("Implement domain-specific subclasses in individual node files")
