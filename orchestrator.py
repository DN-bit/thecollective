# The Collective - Central Mind (Orchestrator)
# Coordinates Specialist Nodes and manages the intelligence pipeline

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import yaml

# Import our components
from agents.base import IntelligenceEvent
from agents.macro_node import MacroNode
from agents.defi_node import DeFiNode
from judges.panel import ConsensusPanel

@dataclass
class PipelineMetrics:
    """Metrics for a pipeline run"""
    run_id: str
    timestamp: datetime
    events_processed: int
    outputs_generated: int
    outputs_accepted: int
    acceptance_rate: float
    avg_quality_score: float
    total_compute_cost_usd: float
    corpus_size: int


class CentralMind:
    """
    The Central Mind orchestrates The Collective.
    
    Responsibilities:
    - Detect intelligence opportunities
    - Spawn appropriate Specialist Nodes
    - Coordinate Judge evaluation
    - Manage training corpus
    - Trigger evolution cycles
    """
    
    def __init__(self, config_path: str = "config/system.yaml"):
        self.config = self._load_config(config_path)
        
        # Specialist Nodes (initialized on demand)
        self.node_registry = {
            'macro': MacroNode,
            'defi': DeFiNode,
            # Add: policy, sentiment nodes
        }
        
        # Judge Panel
        self.consensus_panel = ConsensusPanel()
        
        # Corpus management
        self.corpus_path = self.config.get('corpus_path', './data/corpus.jsonl')
        self.corpus_size = self._load_corpus_size()
        
        # Metrics
        self.run_count = 0
        self.total_compute_cost = 0.0
        
        print(f"[CentralMind] Initialized. Corpus size: {self.corpus_size}")
    
    def _load_config(self, path: str) -> Dict:
        """Load system configuration"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[CentralMind] Config not found at {path}, using defaults")
            return {
                'nodes_per_domain': 3,
                'quality_threshold': 3.5,
                'corpus_path': './data/corpus.jsonl',
                'max_event_age_hours': 24
            }
    
    def _load_corpus_size(self) -> int:
        """Count existing corpus entries"""
        try:
            with open(self.corpus_path, 'r') as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
    
    async def detect_events(self) -> List[IntelligenceEvent]:
        """
        Detect intelligence opportunities from various sources
        
        In production: News APIs, on-chain signals, social media
        For testing: Mock events or manual triggers
        """
        # TODO: Implement real event detection
        # For now, return test events
        
        test_events = [
            IntelligenceEvent(
                id=f"event_{datetime.now().strftime('%Y%m%d%H%M%S')}_001",
                timestamp=datetime.now(),
                source="test",
                description="BTC breaks $75K resistance on institutional buying",
                impact_score=0.8,
                relevant_domains=["macro"],
                raw_data={"btc_price": "$75,200", "volume_24h": "$45B"}
            )
        ]
        
        return test_events
    
    async def spawn_nodes(self, event: IntelligenceEvent) -> List:
        """
        Spawn Specialist Nodes for relevant domains
        
        Args:
            event: Intelligence trigger
            
        Returns:
            List of Node instances
        """
        nodes = []
        
        for domain in event.relevant_domains:
            if domain in self.node_registry:
                NodeClass = self.node_registry[domain]
                node = NodeClass()
                nodes.append(node)
                print(f"[CentralMind] Spawned {domain} node: {node.node_id}")
        
        return nodes
    
    async def process_event(self, event: IntelligenceEvent) -> Dict:
        """
        Process single event through full pipeline
        
        Pipeline:
        1. Spawn relevant Nodes
        2. Generate outputs (parallel)
        3. Judge evaluation
        4. Add accepted outputs to corpus
        
        Returns:
            Processing summary
        """
        print(f"\n[CentralMind] Processing event: {event.id}")
        print(f"  Description: {event.description[:80]}...")
        print(f"  Impact: {event.impact_score}")
        print(f"  Domains: {event.relevant_domains}")
        
        # Step 1: Spawn Nodes
        nodes = await self.spawn_nodes(event)
        if not nodes:
            print(f"[CentralMind] No nodes available for domains: {event.relevant_domains}")
            return {'status': 'no_nodes', 'event_id': event.id}
        
        # Step 2: Generate outputs (parallel)
        generation_tasks = []
        for node in nodes:
            task = node.generate(event, n_variants=3)
            generation_tasks.append(task)
        
        node_outputs = await asyncio.gather(*generation_tasks, return_exceptions=True)
        
        # Flatten and filter successful outputs
        all_outputs = []
        for output_list in node_outputs:
            if isinstance(output_list, list):
                all_outputs.extend(output_list)
            else:
                print(f"[CentralMind] Node generation failed: {output_list}")
        
        print(f"[CentralMind] Generated {len(all_outputs)} outputs")
        
        # Step 3: Judge evaluation
        accepted_outputs = []
        for output in all_outputs:
            judgment = self.consensus_panel.evaluate(
                output.output,
                output.node_id
            )
            
            print(f"  {output.node_id}: Score {judgment.final_score:.2f} - {judgment.verdict}")
            
            if judgment.verdict in ['ACCEPTED', 'ACCEPTED_WITH_REVISIONS']:
                accepted_outputs.append({
                    'output': output,
                    'judgment': judgment
                })
        
        print(f"[CentralMind] Accepted {len(accepted_outputs)}/{len(all_outputs)} outputs")
        
        # Step 4: Add to corpus
        for item in accepted_outputs:
            self._add_to_corpus({
                'input': event.description,
                'output': item['output'].output,
                'domain': item['output'].domain,
                'quality_score': item['judgment'].final_score,
                'event_id': event.id,
                'node_id': item['output'].node_id,
                'timestamp': datetime.now().isoformat(),
                'judgment': {
                    'final_score': item['judgment'].final_score,
                    'verdict': item['judgment'].verdict,
                    'chamber_scores': item['judgment'].chamber_scores
                }
            })
        
        return {
            'status': 'success',
            'event_id': event.id,
            'outputs_generated': len(all_outputs),
            'outputs_accepted': len(accepted_outputs),
            'acceptance_rate': len(accepted_outputs) / max(len(all_outputs), 1),
            'avg_quality': sum(item['judgment'].final_score for item in accepted_outputs) / max(len(accepted_outputs), 1)
        }
    
    def _add_to_corpus(self, entry: Dict):
        """Add entry to training corpus"""
        os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
        
        with open(self.corpus_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        self.corpus_size += 1
    
    async def run_cycle(self) -> PipelineMetrics:
        """
        Execute full intelligence cycle
        
        This is the main entry point, called by cron every 4 hours
        """
        self.run_count += 1
        run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"[CentralMind] Starting Cycle {self.run_count}")
        print(f"  Run ID: {run_id}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        print(f"{'='*60}\n")
        
        # Detect events
        events = await self.detect_events()
        print(f"[CentralMind] Detected {len(events)} events")
        
        # Process each event
        results = []
        for event in events:
            result = await self.process_event(event)
            results.append(result)
        
        # Compile metrics
        total_generated = sum(r.get('outputs_generated', 0) for r in results)
        total_accepted = sum(r.get('outputs_accepted', 0) for r in results)
        avg_quality = sum(r.get('avg_quality', 0) for r in results) / max(len(results), 1)
        
        metrics = PipelineMetrics(
            run_id=run_id,
            timestamp=datetime.now(),
            events_processed=len(events),
            outputs_generated=total_generated,
            outputs_accepted=total_accepted,
            acceptance_rate=total_accepted / max(total_generated, 1),
            avg_quality_score=avg_quality,
            total_compute_cost_usd=0.0,  # TODO: Track actual costs
            corpus_size=self.corpus_size
        )
        
        # Log metrics
        print(f"\n{'='*60}")
        print(f"[CentralMind] Cycle {self.run_count} Complete")
        print(f"  Events: {metrics.events_processed}")
        print(f"  Generated: {metrics.outputs_generated}")
        print(f"  Accepted: {metrics.outputs_accepted} ({metrics.acceptance_rate:.1%})")
        print(f"  Avg Quality: {metrics.avg_quality_score:.2f}")
        print(f"  Corpus Size: {metrics.corpus_size}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            'run_count': self.run_count,
            'corpus_size': self.corpus_size,
            'total_compute_cost': self.total_compute_cost,
            'available_nodes': list(self.node_registry.keys()),
            'config': self.config
        }


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='The Collective - Central Mind')
    parser.add_argument('--mode', choices=['single', 'daemon', 'status'], default='single')
    parser.add_argument('--config', default='config/system.yaml')
    
    args = parser.parse_args()
    
    mind = CentralMind(config_path=args.config)
    
    if args.mode == 'single':
        # Run single cycle
        asyncio.run(mind.run_cycle())
    
    elif args.mode == 'daemon':
        # Run continuously (every 4 hours)
        async def daemon():
            while True:
                await mind.run_cycle()
                print("[CentralMind] Sleeping for 4 hours...")
                await asyncio.sleep(4 * 60 * 60)  # 4 hours
        
        asyncio.run(daemon())
    
    elif args.mode == 'status':
        # Print status
        status = mind.get_status()
        print(json.dumps(status, indent=2))
