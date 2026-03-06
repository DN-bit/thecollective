# MacroNode: Market Intelligence & Geopolitical Analysis
# Specialist Node for The Collective

from typing import Dict
from agents.base import SpecialistNode, IntelligenceEvent

MACRO_NODE_PROMPT = """You are a MacroNode in The Collective — a specialized intelligence unit focused on cryptocurrency markets, geopolitical events, and cross-asset correlations.

Your purpose is to generate institutional-grade macro analysis that will be used to train the next generation of Collective intelligence.

CAPABILITIES:
- Analyze geopolitical events and their impact on crypto markets
- Track cross-asset correlations (BTC vs gold, BTC vs equities, BTC vs DXY)
- Identify regime changes in market structure  
- Generate actionable investment signals with risk assessments
- Provide time-horizon specific recommendations (24h, 1w, 1m)

OUTPUT FORMAT (JSON):
{
  "domain": "macro",
  "timestamp": "ISO8601",
  "input_event": "Brief description of triggering event",
  "market_regime": "bull|bear|chop|risk-on|risk-off",
  "chain_of_thought": [
    "Step 1: Contextualize the event historically",
    "Step 2: Analyze immediate market reaction", 
    "Step 3: Identify key price levels and correlations",
    "Step 4: Assess institutional flow signals",
    "Step 5: Formulate probabilistic scenarios"
  ],
  "scenarios": [
    {
      "name": "Base case",
      "probability": 0.55,
      "description": "Most likely outcome based on current data",
      "btc_price_target": "$X - $Y",
      "time_horizon": "24h-48h",
      "key_levels": {
        "support": ["$68,000", "$65,000"],
        "resistance": ["$75,000", "$78,000"]
      }
    },
    {
      "name": "Bull case",
      "probability": 0.25,
      "description": "Upside surprise scenario",
      "btc_price_target": "$X+",
      "catalyst": "What would trigger this outcome"
    },
    {
      "name": "Bear case", 
      "probability": 0.20,
      "description": "Downside risk scenario",
      "btc_price_target": "$X - $Y",
      "catalyst": "What would trigger this outcome"
    }
  ],
  "recommendation": {
    "action": "hold|accumulate|reduce|hedge",
    "btc_allocation": "X%",
    "rationale": "1-2 sentence evidence-based justification",
    "confidence": 0.0-1.0
  },
  "citations": [
    "source1.com",
    "source2.com/data",
    "source3.com/analysis"
  ],
  "key_metrics": {
    "btc_dominance": "X%",
    "fear_greed_index": "X",
    "etf_flows_24h": "+$XM",
    "funding_rate": "X%"
  }
}

QUALITY STANDARDS:
- Every quantitative claim must have a citation
- Probabilities must sum to exactly 1.0
- Price levels must be specific numbers (not "higher" or "lower")
- Recommendations must include specific time horizons
- Confidence score reflects certainty, not optimism
- Chain-of-thought must show reasoning, not just conclusions

HISTORICAL PRECEDENT EXAMPLES:
- 2020 COVID crash: BTC -50% then +1000% over 12 months
- 2022 FTX collapse: BTC -25% in 3 days, took 6 months to recover
- 2024 ETF approval: BTC +70% in 2 months
- Iran conflicts 2019-2020: Oil +20%, BTC initially correlated then decoupled

Use these precedents to contextualize current events."""


class MacroNode(SpecialistNode):
    """
    Specialist Node for macro market and geopolitical analysis
    Generates institutional-grade intelligence on crypto markets
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(
            domain="macro",
            system_prompt=MACRO_NODE_PROMPT,
            model=model
        )
    
    def _format_prompt(self, event: IntelligenceEvent) -> str:
        """Format intelligence event for macro analysis"""
        
        prompt = f"""Analyze the following macro event and generate comprehensive intelligence:

EVENT: {event.description}
TIMESTAMP: {event.timestamp.isoformat()}
IMPACT SCORE: {event.impact_score}/1.0

CURRENT MARKET CONTEXT (as of {event.timestamp.strftime('%Y-%m-%d %H:%M')}):
- Check latest BTC price, dominance, and key levels
- Review recent ETF flows and funding rates
- Assess Fear & Greed Index trend
- Identify relevant geopolitical precedents

Your analysis must include:
1. Historical precedent comparison
2. Three probabilistic scenarios (base/bull/bear) with specific price targets
3. Cross-asset correlation assessment (BTC vs gold, equities, DXY)
4. Institutional flow interpretation
5. Actionable portfolio recommendation with confidence score

Generate output in the exact JSON format specified in your system prompt."""
        
        # If raw data available, include it
        if event.raw_data:
            prompt += f"\n\nADDITIONAL DATA:\n{event.raw_data}"
        
        return prompt


# Example usage
if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    async def test_macro_node():
        # Create test event
        test_event = IntelligenceEvent(
            id="test_001",
            timestamp=datetime.now(),
            source="news_api",
            description="US submarine sinks Iranian warship off Sri Lanka, Hormuz shipping lanes threatened",
            impact_score=0.85,
            relevant_domains=["macro", "policy"],
            raw_data={
                "btc_price": "$73,000",
                "oil_price": "$76.75",
                "gold_price": "$5,160"
            }
        )
        
        # Initialize and run
        node = MacroNode()
        outputs = await node.generate(test_event, n_variants=2)
        
        print(f"Generated {len(outputs)} macro analysis variants")
        for i, output in enumerate(outputs):
            print(f"\nVariant {i+1}:")
            print(f"  Confidence: {output.confidence}")
            print(f"  Citations: {len(output.citations)} sources")
            print(f"  Compute cost: ${output.compute_cost_usd:.4f}")
    
    # Run test
    asyncio.run(test_macro_node())
