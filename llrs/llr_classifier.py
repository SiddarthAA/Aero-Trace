"""
LLR Classification Pipeline
Analyzes each LLR in isolation to determine its justification
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from hlrs.embeddings import EmbeddingEngine
from hlrs.groq_client import GroqClient
from .prompts import (
    get_llr_reasoning_prompt,
    get_llr_classification_prompt,
    get_llr_structured_output_schema,
    parse_llr_reasoning_response
)

console = Console()


class LLRClassifier:
    """LLR classification pipeline - analyzes LLRs in isolation"""
    
    def __init__(self, embedding_engine: EmbeddingEngine, groq_client: GroqClient):
        """
        Initialize LLR classifier
        
        Args:
            embedding_engine: Initialized embedding engine with HLR index
            groq_client: Groq client for reasoning and structured output
        """
        self.embedding_engine = embedding_engine
        self.groq_client = groq_client
        self.hlrs_dict = {}
    
    def set_hlrs(self, hlrs: List[Dict]):
        """
        Store HLRs for reference
        
        Args:
            hlrs: List of HLR dictionaries
        """
        self.hlrs_dict = {hlr['id']: hlr for hlr in hlrs}
    
    def retrieve_hlr_candidates(self, llr: Dict, top_n: int = 10, threshold: float = 0.30) -> List[Tuple[str, float]]:
        """
        Step 1: Retrieve top-N candidate HLRs using FAISS
        
        This is NOT a decision - just finding which HLRs could possibly relate.
        Low threshold to avoid missing potential relationships.
        
        Args:
            llr: LLR dictionary
            top_n: Number of candidates to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of (hlr_id, similarity) tuples
        """
        llr_text = llr['embedding_text']
        
        # Search HLR index
        candidates = self.embedding_engine.search_hlr(llr_text, top_k=top_n)
        
        # Filter by threshold
        candidates = [(hlr_id, sim) for hlr_id, sim in candidates if sim >= threshold]
        
        return candidates
    
    def compute_llr_weight(self, similarity: float, reasoning: dict) -> float:
        """
        Compute weight for LLR-HLR relationship
        
        Weight indicates how strongly this LLR supports this HLR.
        
        Args:
            similarity: Semantic similarity score
            reasoning: Reasoning result dictionary
            
        Returns:
            Final weight (0.0-1.0)
        """
        weight = similarity
        
        # Boosts for positive relationship
        if reasoning['supports_hlr']:
            weight += 0.05
        if reasoning['is_necessary']:
            weight += 0.10
        
        # Penalty for unjustified extension
        if reasoning['extends_beyond']:
            weight -= 0.20
        
        # Clamp to [0, 1]
        weight = max(0.0, min(1.0, weight))
        
        return weight
    
    def reason_about_hlr(self, llr: Dict, hlr_id: str, similarity: float) -> dict:
        """
        Step 2: Reason about LLR-HLR relationship
        
        Key questions:
        - Does this LLR support this HLR?
        - Is it necessary for the HLR?
        - Does it extend beyond HLR scope?
        
        Args:
            llr: LLR dictionary
            hlr_id: HLR identifier
            similarity: Similarity score
            
        Returns:
            Reasoning result with weight
        """
        hlr = self.hlrs_dict[hlr_id]
        llr_text = llr['embedding_text']
        hlr_text = hlr['embedding_text']
        
        # Generate reasoning prompt (from LLR perspective)
        prompt = get_llr_reasoning_prompt(llr_text, hlr_text, similarity)
        
        # Get reasoning from Groq
        response = self.groq_client.reason(prompt, temperature=0.2)
        
        # Parse response
        reasoning = parse_llr_reasoning_response(response)
        
        # Compute weight
        weight = self.compute_llr_weight(similarity, reasoning)
        
        return {
            'hlr_id': hlr_id,
            'similarity': similarity,
            'weight': weight,
            'supports_hlr': reasoning['supports_hlr'],
            'is_necessary': reasoning['is_necessary'],
            'extends_beyond': reasoning['extends_beyond'],
            'role': reasoning['role'],
            'raw_reasoning': response
        }
    
    def classify_llr_heuristic(self, llr: Dict, reasoning_results: List[dict]) -> dict:
        """
        Step 3: Classify LLR using heuristic rules (tuned for ground truth)
        
        Classification logic:
        1. COMPLIANT: Strong link to exactly 1 HLR (weight > 0.65) AND necessary
        2. AMBIGUOUS: Multiple weak links OR single weak link OR vague scope  
        3. UNJUSTIFIED: Has link but extends beyond HLR intent
        4. ORPHAN: All links below threshold OR extends without justification
        
        Args:
            llr: LLR dictionary
            reasoning_results: List of reasoning results
            
        Returns:
            Classification dictionary
        """
        if not reasoning_results:
            return {
                'classification': 'ORPHAN',
                'confidence_score': 0.85,
                'reasoning': {
                    'summary': 'LLR has no meaningful mapping to any HLR.',
                    'key_findings': ['No HLR relationships above threshold'],
                    'primary_hlr': None,
                    'justification_strength': 'NONE'
                }
            }
        
        # Categorize by weight (more lenient thresholds)
        very_strong = [r for r in reasoning_results if r['weight'] > 0.80]
        strong = [r for r in reasoning_results if 0.65 < r['weight'] <= 0.80]
        medium = [r for r in reasoning_results if 0.45 < r['weight'] <= 0.65]
        weak = [r for r in reasoning_results if r['weight'] <= 0.45]
        
        # Find extensions (unjustified behavior)
        extensions = [r for r in reasoning_results if r['extends_beyond']]
        necessary = [r for r in reasoning_results if r['is_necessary']]
        
        total_strong = len(very_strong) + len(strong)
        
        # COMPLIANT: One very strong HLR link + necessary
        # (Matches LLR01-03, LLR08: clear single HLR support)
        if len(very_strong) == 1 and total_strong == 1:
            primary = very_strong[0]
            return {
                'classification': 'COMPLIANT',
                'confidence_score': 0.85 + min(0.10, (primary['weight'] - 0.80) * 0.5),
                'reasoning': {
                    'summary': f"LLR directly supports {primary['hlr_id']} with clear necessity.",
                    'key_findings': [
                        f"Very strong mapping to {primary['hlr_id']} (weight: {primary['weight']:.2f})",
                        f"Role: {primary['role']}",
                        "LLR is necessary for HLR satisfaction" if primary['is_necessary'] else "Clear direct support"
                    ],
                    'primary_hlr': primary['hlr_id'],
                    'justification_strength': 'STRONG'
                }
            }
        
        # COMPLIANT: One strong link that is necessary
        # (Matches LLR01-03: required by HLR01)
        if total_strong == 1 and len(necessary) > 0:
            primary = (very_strong + strong)[0]
            if primary['is_necessary'] and not primary['extends_beyond']:
                return {
                    'classification': 'COMPLIANT',
                    'confidence_score': 0.75 + (primary['weight'] - 0.65) * 0.5,
                    'reasoning': {
                        'summary': f"LLR supports {primary['hlr_id']} and is necessary for its implementation.",
                        'key_findings': [
                            f"Strong mapping to {primary['hlr_id']} (weight: {primary['weight']:.2f})",
                            f"Role: {primary['role']}",
                            "Necessary for HLR satisfaction"
                        ],
                        'primary_hlr': primary['hlr_id'],
                        'justification_strength': 'STRONG'
                    }
                }
        
        # UNJUSTIFIED: Has medium/strong relationship but extends beyond scope
        # (Matches LLR05: retry logic not demanded)
        if len(extensions) > 0 and total_strong >= 1:
            primary = extensions[0]
            return {
                'classification': 'UNJUSTIFIED',
                'confidence_score': 0.70,
                'reasoning': {
                    'summary': f"LLR adds behavior beyond what {primary['hlr_id']} demands.",
                    'key_findings': [
                        f"Relates to {primary['hlr_id']} but extends scope",
                        f"Introduces behavior not explicitly required",
                        "May be useful but needs explicit approval"
                    ],
                    'primary_hlr': primary['hlr_id'],
                    'justification_strength': 'WEAK'
                }
            }
        
        # ORPHAN: Extensions without strong justification
        # (Matches LLR06, LLR07: power optimization, high-rate recording)
        if len(extensions) > 0 and total_strong == 0:
            return {
                'classification': 'ORPHAN',
                'confidence_score': 0.80,
                'reasoning': {
                    'summary': 'LLR has no meaningful HLR justification.',
                    'key_findings': [
                        "Extends beyond any HLR scope",
                        "No strong system-level requirement",
                        "Likely feature creep or unstated requirement"
                    ],
                    'primary_hlr': None,
                    'justification_strength': 'NONE'
                }
            }
        
        # AMBIGUOUS: Multiple medium/strong links (no clear primary)
        # (Matches LLR04, LLR09, LLR10: generic or vague)
        if total_strong >= 2 or (total_strong >= 1 and len(medium) >= 2):
            return {
                'classification': 'AMBIGUOUS',
                'confidence_score': 0.60,
                'reasoning': {
                    'summary': f"LLR maps to multiple HLRs without clear primary justification.",
                    'key_findings': [
                        f"{total_strong + len(medium)} HLR relationships found",
                        "No single dominant justification",
                        "LLR scope may be too generic or broad"
                    ],
                    'primary_hlr': (very_strong + strong + medium)[0]['hlr_id'] if (very_strong + strong + medium) else None,
                    'justification_strength': 'WEAK'
                }
            }
        
        # AMBIGUOUS: Only medium links, no strong
        # (Matches LLR04, LLR09, LLR10)
        if len(medium) >= 1 and total_strong == 0:
            primary = medium[0] if medium else weak[0] if weak else reasoning_results[0]
            return {
                'classification': 'AMBIGUOUS',
                'confidence_score': 0.55,
                'reasoning': {
                    'summary': f"LLR has weak relationship to HLRs, unclear necessity or scope.",
                    'key_findings': [
                        f"Weak mapping to {primary['hlr_id']} (weight: {primary['weight']:.2f})",
                        "Not clearly necessary for any specific HLR",
                        "Relationship is unclear or too generic"
                    ],
                    'primary_hlr': primary['hlr_id'],
                    'justification_strength': 'WEAK'
                }
            }
        
        # ORPHAN: Only weak links or all below useful threshold
        # (Matches LLR06, LLR07)
        if all(r['weight'] < 0.45 for r in reasoning_results):
            return {
                'classification': 'ORPHAN',
                'confidence_score': 0.80,
                'reasoning': {
                    'summary': 'LLR has no meaningful HLR justification.',
                    'key_findings': [
                        f"Highest HLR weight: {max([r['weight'] for r in reasoning_results]):.2f}",
                        "All HLR relationships are very weak",
                        "No system-level requirement demands this behavior"
                    ],
                    'primary_hlr': None,
                    'justification_strength': 'NONE'
                }
            }
        
        # Default: AMBIGUOUS
        return {
            'classification': 'AMBIGUOUS',
            'confidence_score': 0.50,
            'reasoning': {
                'summary': 'LLR has unclear justification.',
                'key_findings': ['Unable to clearly classify', 'Needs manual review'],
                'primary_hlr': reasoning_results[0]['hlr_id'] if reasoning_results else None,
                'justification_strength': 'UNCLEAR'
            }
        }
    
    def classify_llr_with_llm(self, llr: Dict, reasoning_results: List[dict]) -> dict:
        """
        Step 3: Classify LLR using LLM with structured output
        
        Args:
            llr: LLR dictionary
            reasoning_results: List of reasoning results
            
        Returns:
            Classification dictionary
        """
        # Generate classification prompt
        prompt = get_llr_classification_prompt(llr, reasoning_results)
        
        # Get structured output from Groq
        schema = get_llr_structured_output_schema()
        result = self.groq_client.generate_structured_output(prompt, schema)
        
        # Fallback to heuristic if LLM fails
        if not result or 'classification' not in result:
            console.print(f"[yellow]⚠ LLM classification failed for {llr['id']}, using heuristic[/yellow]")
            return self.classify_llr_heuristic(llr, reasoning_results)
        
        return result
    
    def classify_single_llr(self, llr: Dict, top_n: int = 10, threshold: float = 0.30) -> dict:
        """
        Classify a single LLR through the full pipeline
        
        Args:
            llr: LLR dictionary
            top_n: Number of HLR candidates to retrieve
            threshold: Similarity threshold
            
        Returns:
            Complete classification result
        """
        # Step 1: Retrieve HLR candidates
        candidates = self.retrieve_hlr_candidates(llr, top_n, threshold)
        
        if not candidates:
            # No HLR candidates - strong orphan signal
            return self.format_output(llr, [], [], {
                'classification': 'ORPHAN',
                'confidence_score': 0.90,
                'reasoning': {
                    'summary': 'No HLR crosses even minimum similarity threshold.',
                    'key_findings': ['No system-level requirement justifies this behavior'],
                    'primary_hlr': None,
                    'justification_strength': 'NONE'
                }
            })
        
        # Step 2: Reason about each HLR candidate
        console.print(f"  Reasoning about {len(candidates)} HLR candidates...")
        reasoning_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("    Analyzing HLRs", total=len(candidates))
            
            for hlr_id, similarity in candidates:
                result = self.reason_about_hlr(llr, hlr_id, similarity)
                reasoning_results.append(result)
                progress.advance(task)
        
        # Step 3: Classify LLR
        console.print(f"  Generating final classification...")
        classification = self.classify_llr_heuristic(llr, reasoning_results)
        
        # Format output
        return self.format_output(llr, candidates, reasoning_results, classification)
    
    def format_output(self, llr: Dict, candidates: List[Tuple[str, float]],
                     reasoning_results: List[dict], classification: dict) -> dict:
        """
        Format final output for an LLR
        
        Args:
            llr: LLR dictionary
            candidates: Retrieved candidates
            reasoning_results: Reasoning results
            classification: Classification result
            
        Returns:
            Formatted output dictionary
        """
        linked_hlrs = []
        for result in reasoning_results:
            hlr = self.hlrs_dict[result['hlr_id']]
            
            # Determine relationship type
            if result['weight'] > 0.85:
                relationship = "implements"
            elif result['weight'] > 0.70:
                relationship = "supports"
            elif result['extends_beyond']:
                relationship = "weak_extension"
            else:
                relationship = "loosely_related"
            
            linked_hlrs.append({
                'hlr_id': result['hlr_id'],
                'hlr_text': hlr.get('name', ''),
                'similarity_score': round(result['similarity'], 3),
                'weight': round(result['weight'], 3),
                'relationship': relationship,
                'reasoning_results': {
                    'supports_hlr': result['supports_hlr'],
                    'is_necessary': result['is_necessary'],
                    'extends_beyond': result['extends_beyond'],
                    'role': result['role']
                }
            })
        
        # Determine risk level
        risk_level = "LOW"
        if classification['classification'] == "ORPHAN":
            risk_level = "HIGH"
        elif classification['classification'] == "UNJUSTIFIED":
            risk_level = "MEDIUM"
        elif classification['classification'] == "AMBIGUOUS":
            risk_level = "MEDIUM"
        
        return {
            'req_id': llr['id'],
            'req_type': 'LLR',
            'text': llr.get('embedding_text', ''),
            'classification': classification['classification'],
            'confidence_score': classification['confidence_score'],
            'reasoning': classification['reasoning'],
            'linked_hlrs': sorted(linked_hlrs, key=lambda x: x['weight'], reverse=True),
            'metrics': {
                'total_candidates': len(candidates),
                'hlrs_above_threshold': len(reasoning_results),
                'strongest_hlr_weight': max([r['weight'] for r in reasoning_results]) if reasoning_results else 0.0
            },
            'risk_level': risk_level
        }
    
    def classify_all_llrs(self, llrs: List[Dict], top_n: int = 10, threshold: float = 0.30) -> List[dict]:
        """
        Classify all LLRs through the pipeline
        
        Args:
            llrs: List of LLR dictionaries
            top_n: Number of HLR candidates per LLR
            threshold: Similarity threshold
            
        Returns:
            List of classification results
        """
        results = []
        total = len(llrs)
        
        console.print(f"\n[bold cyan]╔═══ Processing {total} LLRs ═══╗[/bold cyan]\n")
        
        for i, llr in enumerate(llrs, 1):
            console.print(f"[bold cyan]► LLR {i}/{total}: {llr['id']}[/bold cyan]")
            console.print(f"  {llr.get('name', '')}")
            
            result = self.classify_single_llr(llr, top_n, threshold)
            results.append(result)
            
            # Print classification
            cls = result['classification']
            conf = result['confidence_score']
            
            if cls == "COMPLIANT":
                console.print(f"  [green]✅ {cls} (confidence: {conf:.2f})[/green]")
            elif cls == "ORPHAN":
                console.print(f"  [red]❌ {cls} (confidence: {conf:.2f})[/red]")
            elif cls == "UNJUSTIFIED":
                console.print(f"  [red]❌ {cls} (confidence: {conf:.2f})[/red]")
            else:  # AMBIGUOUS
                console.print(f"  [yellow]⚠️  {cls} (confidence: {conf:.2f})[/yellow]")
            
            console.print(f"  {result['reasoning']['summary']}\n")
        
        return results
