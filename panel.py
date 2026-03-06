# The Collective - Consensus Judge Panel
# Multi-chamber quality evaluation system

import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import openai
import os

@dataclass
class Judgment:
    """Consensus judgment on Node output"""
    judgment_id: str
    node_output_id: str
    final_score: float
    chamber_scores: Dict[str, float]
    chamber_notes: Dict[str, List[str]]
    verdict: str
    revision_notes: List[str]
    timestamp: datetime


class ConsensusChamber:
    """Base class for judge chambers"""
    
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def evaluate(self, node_output: Dict) -> Dict:
        """Evaluate output and return score + notes"""
        raise NotImplementedError


class LogicChamber(ConsensusChamber):
    """Evaluates reasoning quality and structure (40% weight)"""
    
    def __init__(self):
        super().__init__("LogicChamber", 0.40)
    
    def evaluate(self, node_output: Dict) -> Dict:
        prompt = f"""You are the Logic Chamber of The Collective. Evaluate this intelligence output for:

1. STRUCTURE COMPLETENESS (1-5): All required fields present? Rich detail?
2. REASONING DEPTH (1-5): Chain-of-thought shows nuanced analysis?
3. ACTIONABILITY (1-5): Clear, specific, time-bound recommendations?

OUTPUT TO EVALUATE:
{json.dumps(node_output, indent=2)}

Return JSON:
{{
  "structure_score": X,
  "reasoning_score": X,
  "actionability_score": X,
  "average_score": X.X,
  "notes": ["strength1", "improvement_needed"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a ruthless quality reviewer. Be harsh and critical."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                'score': result.get('average_score', 0),
                'breakdown': {
                    'structure': result.get('structure_score', 0),
                    'reasoning': result.get('reasoning_score', 0),
                    'actionability': result.get('actionability_score', 0)
                },
                'notes': result.get('notes', [])
            }
        except Exception as e:
            return {'score': 0, 'breakdown': {}, 'notes': [f'Error: {str(e)}']}


class TruthChamber(ConsensusChamber):
    """Evaluates factual accuracy and consistency (30% weight)"""
    
    def __init__(self):
        super().__init__("TruthChamber", 0.30)
    
    def evaluate(self, node_output: Dict) -> Dict:
        prompt = f"""You are the Truth Chamber of The Collective. Evaluate for:

1. LOGICAL COHERENCE (1-5): No contradictions? Arguments flow?
2. FACTUAL ACCURACY (1-5): Claims verifiable? No hallucinations?
3. PROBABILISTIC SANITY (1-5): Probabilities sum to 1? Calibrated?

OUTPUT TO VERIFY:
{json.dumps(node_output, indent=2)}

Check for:
- Internal contradictions
- Unsupported claims
- Math errors in probabilities
- Impossible assertions

Return JSON:
{{
  "coherence_score": X,
  "accuracy_score": X,
  "probability_score": X,
  "average_score": X.X,
  "notes": ["issue1", "issue2"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You verify facts ruthlessly. Challenge every claim."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                'score': result.get('average_score', 0),
                'breakdown': {
                    'coherence': result.get('coherence_score', 0),
                    'accuracy': result.get('accuracy_score', 0),
                    'probability': result.get('probability_score', 0)
                },
                'notes': result.get('notes', [])
            }
        except Exception as e:
            return {'score': 0, 'breakdown': {}, 'notes': [f'Error: {str(e)}']}


class SourceChamber(ConsensusChamber):
    """Evaluates attribution and source quality (30% weight)"""
    
    def __init__(self):
        super().__init__("SourceChamber", 0.30)
    
    def evaluate(self, node_output: Dict) -> Dict:
        citations = node_output.get('citations', [])
        
        prompt = f"""You are the Source Chamber of The Collective. Evaluate:

1. CITATION PRESENCE (1-5): All claims cited? No unsourced assertions?
2. SOURCE QUALITY (1-5): Tier 1-2 sources (official, reputable)?
3. CITATION ACCURACY (1-5): Sources match claims? No misattribution?

CITATIONS PROVIDED:
{json.dumps(citations, indent=2)}

TIER 1: Government, official docs, established data (Glassnode, CoinMetrics)
TIER 2: Reputable news (CoinDesk, The Block), verified social, exchange APIs
TIER 3: Community sources, forums, unverified social
TIER 4: Anonymous, known misinformation

Return JSON:
{{
  "presence_score": X,
  "quality_score": X,
  "accuracy_score": X,
  "average_score": X.X,
  "tier_breakdown": {{"tier1": X, "tier2": X, "tier3": X, "tier4": X}},
  "notes": ["source_issue1", "source_issue2"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You verify sources obsessively. Quality over quantity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                'score': result.get('average_score', 0),
                'breakdown': {
                    'presence': result.get('presence_score', 0),
                    'quality': result.get('quality_score', 0),
                    'accuracy': result.get('accuracy_score', 0)
                },
                'notes': result.get('notes', [])
            }
        except Exception as e:
            return {'score': 0, 'breakdown': {}, 'notes': [f'Error: {str(e)}']}


class ConsensusPanel:
    """
    Three-chamber consensus system for evaluating Node outputs
    Combines Logic, Truth, and Source chambers into final judgment
    """
    
    def __init__(self):
        self.chambers = {
            'logic': LogicChamber(),
            'truth': TruthChamber(),
            'source': SourceChamber()
        }
    
    def evaluate(self, node_output: Dict, node_output_id: str) -> Judgment:
        """
        Run all three chambers and compute consensus
        
        Returns:
            Judgment object with final score and verdict
        """
        chamber_results = {}
        chamber_scores = {}
        chamber_notes = {}
        
        # Run each chamber
        for name, chamber in self.chambers.items():
            result = chamber.evaluate(node_output)
            chamber_results[name] = result
            chamber_scores[name] = result['score']
            chamber_notes[name] = result['notes']
        
        # Calculate weighted final score
        final_score = (
            chamber_scores['logic'] * 0.40 +
            chamber_scores['truth'] * 0.30 +
            chamber_scores['source'] * 0.30
        )
        
        # Determine verdict
        if final_score >= 4.5:
            verdict = "ACCEPTED"
        elif final_score >= 3.5:
            verdict = "ACCEPTED_WITH_REVISIONS"
        elif final_score >= 2.5:
            verdict = "REJECTED_REVISION_POSSIBLE"
        else:
            verdict = "REJECTED"
        
        # Compile revision notes
        revision_notes = []
        for chamber_name, notes in chamber_notes.items():
            for note in notes:
                revision_notes.append(f"[{chamber_name}] {note}")
        
        return Judgment(
            judgment_id=f"judgment_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            node_output_id=node_output_id,
            final_score=final_score,
            chamber_scores=chamber_scores,
            chamber_notes=chamber_notes,
            verdict=verdict,
            revision_notes=revision_notes,
            timestamp=datetime.now()
        )


# Example usage
if __name__ == "__main__":
    # Test the panel
    test_output = {
        "domain": "macro",
        "scenarios": [
            {"name": "Base case", "probability": 0.55, "btc_price_target": "$70K-$76K"},
            {"name": "Bull case", "probability": 0.25, "btc_price_target": "$80K+"},
            {"name": "Bear case", "probability": 0.20, "btc_price_target": "$65K-$68K"}
        ],
        "recommendation": {
            "action": "hold",
            "confidence": 0.75
        },
        "citations": ["glassnode.com", "coinglass.com"],
        "chain_of_thought": ["Analysis step 1", "Analysis step 2"]
    }
    
    panel = ConsensusPanel()
    judgment = panel.evaluate(test_output, "test_output_001")
    
    print(f"Final Score: {judgment.final_score:.2f}")
    print(f"Verdict: {judgment.verdict}")
    print(f"Chamber Scores: {judgment.chamber_scores}")
    print(f"Revision Notes: {judgment.revision_notes}")
