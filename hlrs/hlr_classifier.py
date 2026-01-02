"""
HLR Classification Pipeline
"""
import json
from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .embeddings import EmbeddingEngine
from .groq_client import GroqClient
from .prompts import (
    get_reasoning_prompt,
    get_classification_prompt,
    get_structured_output_schema,
    parse_reasoning_response
)

console = Console()


class HLRClassifier:
    """Main HLR classification pipeline"""
    
    def __init__(self, embedding_engine: EmbeddingEngine, groq_client: GroqClient):
        """
        Initialize classifier
        
        Args:
            embedding_engine: Initialized embedding engine with LLR index
            groq_client: Groq client for reasoning and structured output
        """
        self.embedding_engine = embedding_engine
        self.groq_client = groq_client
        self.llrs_dict = {}
    
    def set_llrs(self, llrs: List[Dict]):
        """
        Store LLRs for reference
        
        Args:
            llrs: List of LLR dictionaries
        """
        self.llrs_dict = {llr['id']: llr for llr in llrs}
    
    def retrieve_candidates(self, hlr: Dict, top_k: int = 10, threshold: float = 0.35) -> List[Tuple[str, float]]:
        """
        Step 1: Retrieve top-K candidate LLRs using FAISS
        
        Args:
            hlr: HLR dictionary
            top_k: Number of candidates to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of (llr_id, similarity) tuples
        """
        hlr_text = hlr['embedding_text']
        candidates = self.embedding_engine.search_top_k(hlr_text, k=top_k, threshold=threshold)
        return candidates
    
    def compute_llr_weight(self, similarity: float, reasoning: dict) -> float:
        """
        Compute weight for an LLR based on similarity and reasoning
        
        Enhanced scoring system (tuned for better classification):
        - Base: similarity score (0.30-1.0)
        - Boost: +0.10 if required (makes it necessary)
        - Boost: +0.10 if prevents violation (critical for safety)
        - Boost: +0.08 if constrains unsafe behavior (safety constraint)
        - Penalty: -0.25 if extends beyond intent (unjustified)
        
        Weight categories:
        - 0.85-1.0: Very strong (essential for HLR)
        - 0.70-0.85: Strong (important for HLR)
        - 0.45-0.70: Medium (supports HLR but not critical)
        - 0.30-0.45: Weak (loosely related)
        
        Args:
            similarity: Semantic similarity score
            reasoning: Reasoning result dictionary
            
        Returns:
            Final weight (0.0-1.0)
        """
        # Start with similarity as base
        weight = similarity
        
        # Apply boosts for positive reasoning
        if reasoning['is_required']:
            weight += 0.10
        if reasoning['prevents_violation']:
            weight += 0.10
        if reasoning['constrains_unsafe_behavior']:
            weight += 0.08
        
        # Apply penalty for unjustified extensions
        if reasoning['extends_beyond_intent']:
            weight -= 0.25
        
        # Clamp to [0, 1]
        weight = max(0.0, min(1.0, weight))
        
        return weight
    
    def reason_about_llr(self, hlr: Dict, llr_id: str, similarity: float) -> dict:
        """
        Step 2: Reason about a single HLR-LLR relationship
        
        Args:
            hlr: HLR dictionary
            llr_id: LLR identifier
            similarity: Similarity score
            
        Returns:
            Reasoning result with weight
        """
        llr = self.llrs_dict[llr_id]
        hlr_text = hlr['embedding_text']
        llr_text = llr['embedding_text']
        
        # Generate reasoning prompt
        prompt = get_reasoning_prompt(hlr_text, llr_text, similarity)
        
        # Get reasoning from Groq
        response = self.groq_client.reason(prompt, temperature=0.2)
        
        # Parse response
        reasoning = parse_reasoning_response(response)
        
        # Compute weight
        weight = self.compute_llr_weight(similarity, reasoning)
        
        return {
            'llr_id': llr_id,
            'similarity': similarity,
            'weight': weight,
            'is_required': reasoning['is_required'],
            'prevents_violation': reasoning['prevents_violation'],
            'constrains_unsafe_behavior': reasoning['constrains_unsafe_behavior'],
            'extends_beyond_intent': reasoning['extends_beyond_intent'],
            'raw_reasoning': response
        }
    
    def classify_hlr_heuristic(self, reasoning_results: List[dict]) -> dict:
        """
        Step 3: Classify HLR based on reasoning results (heuristic fallback)
        Tuned to match ground truth patterns with detailed gap analysis
        
        Args:
            reasoning_results: List of reasoning results
            
        Returns:
            Classification dictionary with detailed gap analysis
        """
        # Count by weight categories (stricter thresholds)
        very_strong = [r for r in reasoning_results if r['weight'] > 0.85]
        strong = [r for r in reasoning_results if 0.70 < r['weight'] <= 0.85]
        medium = [r for r in reasoning_results if 0.50 < r['weight'] <= 0.70]
        weak = [r for r in reasoning_results if 0.35 < r['weight'] <= 0.50]
        required = [r for r in reasoning_results if r['is_required']]
        prevents_violation = [r for r in reasoning_results if r['prevents_violation']]
        constrains_unsafe = [r for r in reasoning_results if r['constrains_unsafe_behavior']]
        
        total_strong = len(very_strong) + len(strong)
        
        # Initialize reasoning structure
        covered_aspects = []
        missing_aspects = []
        recommendations = []
        gaps_identified = []
        
        # Analyze what's covered
        if required:
            covered_aspects.append(f"Core functionality addressed by {len(required)} LLR(s)")
        if prevents_violation:
            covered_aspects.append(f"Violation prevention covered by {len(prevents_violation)} LLR(s)")
        if constrains_unsafe:
            covered_aspects.append(f"Safety constraints defined by {len(constrains_unsafe)} LLR(s)")
        
        # FULLY_TRACED: Needs 3+ strong LLRs with high weights
        # (Matches HLR01: has LLR01-03 with high similarity)
        if len(very_strong) >= 3 or (len(very_strong) >= 2 and total_strong >= 3):
            classification = "FULLY_TRACED"
            confidence = 0.85 + min(0.10, len(very_strong) * 0.03)
            summary = f"HLR is well-decomposed with {len(very_strong)} very strong and {len(strong)} strong supporting LLRs."
            key_findings = [
                f"Total of {total_strong} strong LLR relationships identified",
                f"{len(required)} LLRs are required for HLR satisfaction",
                "All critical aspects appear to be covered"
            ]
        
        # PARTIAL_TRACE: Has 1-2 LLRs but incomplete or missing critical aspects
        # (Matches HLR02, HLR05, HLR06: some support but gaps)
        elif total_strong >= 1 and total_strong < 3:
            classification = "PARTIAL_TRACE"
            confidence = 0.50 + (total_strong * 0.10)
            
            # Detailed gap analysis for PARTIAL_TRACE
            if len(very_strong) == 0:
                missing_aspects.append("No very strong LLR decomposition found")
                gaps_identified.append("Lack of comprehensive implementation - only moderate-strength LLRs exist")
                recommendations.append("Add LLRs with stronger direct implementation of HLR intent")
            
            if total_strong < 2:
                missing_aspects.append("Insufficient number of strong supporting LLRs")
                gaps_identified.append(f"Only {total_strong} strong LLR(s) found, expected at least 2-3 for full coverage")
                recommendations.append("Identify and add missing LLRs to cover all HLR aspects")
            
            if len(required) == 0:
                missing_aspects.append("No LLRs marked as strictly required for HLR")
                gaps_identified.append("Current LLRs may be supportive but not essential")
                recommendations.append("Define LLRs that are necessary for HLR fulfillment")
            
            if len(prevents_violation) == 0:
                missing_aspects.append("No violation prevention mechanisms defined")
                gaps_identified.append("Missing LLRs that prevent HLR violation scenarios")
                recommendations.append("Add LLRs specifying error detection and prevention logic")
            
            if len(constrains_unsafe) == 0:
                missing_aspects.append("No safety constraints specified")
                gaps_identified.append("Missing LLRs that constrain unsafe behaviors")
                recommendations.append("Add LLRs defining safety bounds and constraints")
            
            if len(medium) > len(strong):
                missing_aspects.append("Too many medium-strength links relative to strong links")
                gaps_identified.append("LLRs are tangentially related but lack direct implementation")
                recommendations.append("Strengthen LLR specificity to directly address HLR requirements")
            
            summary = f"HLR has partial support with {total_strong} strong supporter(s), but critical gaps remain."
            key_findings = [
                f"Found {total_strong} strong LLR(s) and {len(medium)} medium-strength LLR(s)",
                f"Only {len(required)} LLR(s) marked as required",
                f"Coverage is incomplete - {len(gaps_identified)} gap(s) identified"
            ]
        
        # PARTIAL_TRACE: Only medium supporters, no strong ones
        elif len(medium) >= 1 and total_strong == 0:
            classification = "PARTIAL_TRACE"
            confidence = 0.45
            
            # Detailed gap analysis
            missing_aspects = [
                "No strong LLR decomposition exists",
                "Current LLRs are only tangentially related",
                "Missing direct implementation of HLR intent"
            ]
            gaps_identified = [
                f"Found {len(medium)} medium-strength LLR(s) but zero strong LLRs",
                "Weak semantic relationship indicates insufficient decomposition",
                "HLR aspects are not explicitly addressed by any LLR"
            ]
            recommendations = [
                "Create LLRs that directly implement HLR requirements",
                "Refine existing medium-strength LLRs to be more specific",
                "Add explicit traceability markers in LLR descriptions"
            ]
            
            summary = f"HLR has only medium-strength support ({len(medium)} LLRs). No strong decomposition found."
            key_findings = [
                f"{len(medium)} medium-strength relationships detected",
                "No strong or required LLR relationships",
                "Significant implementation gaps exist"
            ]
        
        # TRACE_HOLE: No meaningful support
        # (Matches HLR03, HLR04: no relevant LLRs)
        else:
            classification = "TRACE_HOLE"
            confidence = 0.70
            
            missing_aspects = [
                "Complete lack of LLR implementation",
                "No detectable semantic relationships",
                "All HLR aspects are unaddressed"
            ]
            gaps_identified = [
                "Zero meaningful LLR support found",
                "HLR appears to have no decomposition in current LLR set"
            ]
            recommendations = [
                "Perform complete decomposition analysis of this HLR",
                "Create comprehensive LLR set covering all HLR aspects",
                "Review if HLR should be split into more specific requirements"
            ]
            
            if len(weak) > 0:
                summary = f"HLR has no meaningful LLR implementation. Only {len(weak)} weak link(s) found."
                key_findings = [f"Only {len(weak)} weak relationship(s) detected", "No substantial implementation exists"]
            else:
                summary = "HLR has no meaningful LLR implementation. No supporting LLRs found."
                key_findings = ["Zero LLR relationships detected", "Complete implementation gap"]
        
        return {
            'classification': classification,
            'confidence_score': min(0.95, confidence),
            'reasoning': {
                'summary': summary,
                'key_findings': key_findings,
                'gaps_identified': gaps_identified,
                'covered_aspects': covered_aspects,
                'missing_aspects': missing_aspects,
                'recommendations': recommendations
            },
            'metrics': {
                'total_candidates': len(reasoning_results),
                'very_strong_supporters': len(very_strong),
                'strong_supporters': len(strong),
                'medium_supporters': len(medium),
                'weak_supporters': len(weak),
                'required_llrs': len(required)
            }
        }
    
    def classify_hlr_with_llm(self, hlr: Dict, reasoning_results: List[dict]) -> dict:
        """
        Step 3: Classify HLR using LLM with structured output
        
        Args:
            hlr: HLR dictionary
            reasoning_results: List of reasoning results
            
        Returns:
            Classification dictionary with gap analysis
        """
        # Generate classification prompt
        prompt = get_classification_prompt(hlr, reasoning_results)
        
        # Get structured output from Groq (using fast model)
        schema = get_structured_output_schema()
        result = self.groq_client.generate_structured_output(prompt, schema)
        
        # Fallback to heuristic if LLM fails
        if not result or 'classification' not in result:
            console.print(f"[yellow]⚠ LLM classification failed, using heuristic fallback[/yellow]")
            return self.classify_hlr_heuristic(reasoning_results)
        
        # Ensure all required fields exist (LLM might not provide all fields)
        if 'reasoning' not in result:
            result['reasoning'] = {}
        
        reasoning = result['reasoning']
        
        # Ensure all reasoning sub-fields exist
        if 'summary' not in reasoning:
            reasoning['summary'] = "Classification completed"
        if 'key_findings' not in reasoning:
            reasoning['key_findings'] = []
        if 'gaps_identified' not in reasoning:
            reasoning['gaps_identified'] = []
        if 'covered_aspects' not in reasoning:
            reasoning['covered_aspects'] = []
        if 'missing_aspects' not in reasoning:
            reasoning['missing_aspects'] = []
        if 'recommendations' not in reasoning:
            reasoning['recommendations'] = []
        
        return result
    
    def format_output(self, hlr: Dict, candidates: List[Tuple[str, float]], 
                     reasoning_results: List[dict], classification: dict) -> dict:
        """
        Format final output for an HLR
        
        Args:
            hlr: HLR dictionary
            candidates: Retrieved candidates
            reasoning_results: Reasoning results
            classification: Classification result
            
        Returns:
            Formatted output dictionary
        """
        linked_llrs = []
        for result in reasoning_results:
            llr = self.llrs_dict[result['llr_id']]
            
            # Determine relationship type
            if result['weight'] > 0.85:
                relationship = "implements"
            elif result['weight'] > 0.70:
                relationship = "supports"
            elif result['constrains_unsafe_behavior']:
                relationship = "constrains_unsafe_behavior"
            elif result['extends_beyond_intent']:
                relationship = "weak_extension"
            else:
                relationship = "partially_supports"
            
            linked_llrs.append({
                'llr_id': result['llr_id'],
                'llr_text': llr.get('name', ''),
                'similarity_score': round(result['similarity'], 3),
                'weight': round(result['weight'], 3),
                'relationship': relationship,
                'reasoning_results': {
                    'is_required': result['is_required'],
                    'prevents_violation': result['prevents_violation'],
                    'constrains_unsafe_behavior': result['constrains_unsafe_behavior'],
                    'extends_beyond_intent': result['extends_beyond_intent']
                }
            })
        
        # Determine risk level
        if classification['classification'] == 'TRACE_HOLE':
            risk_level = "HIGH"
        elif classification['classification'] == 'PARTIAL_TRACE':
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'req_id': hlr['id'],
            'req_type': 'HLR',
            'text': hlr.get('embedding_text', ''),
            'classification': classification['classification'],
            'confidence_score': round(classification['confidence_score'], 3),
            'reasoning': classification['reasoning'],
            'linked_llrs': linked_llrs,
            'metrics': classification['metrics'],
            'risk_level': risk_level
        }
    
    def classify_single_hlr(self, hlr: Dict, top_k: int = 10, threshold: float = 0.35) -> dict:
        """
        Complete pipeline for classifying a single HLR
        
        Args:
            hlr: HLR dictionary
            top_k: Number of candidates to retrieve
            threshold: Similarity threshold
            
        Returns:
            Complete classification result
        """
        # Step 1: Retrieve candidates
        candidates = self.retrieve_candidates(hlr, top_k, threshold)
        
        if not candidates:
            # No candidates found
            return self.format_output(
                hlr, [], [],
                {
                    'classification': 'TRACE_HOLE',
                    'confidence_score': 0.90,
                    'reasoning': {
                        'summary': 'No LLRs found above similarity threshold.',
                        'key_findings': ['Zero semantic matches detected'],
                        'gaps_identified': ['Complete lack of implementation']
                    },
                    'metrics': {
                        'total_candidates': 0,
                        'strong_supporters': 0,
                        'medium_supporters': 0,
                        'weak_supporters': 0,
                        'required_llrs': 0
                    }
                }
            )
        
        # Step 2: Reason about each candidate
        console.print(f"  [cyan]Reasoning about {len(candidates)} candidates...[/cyan]")
        reasoning_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("  Analyzing LLRs", total=len(candidates))
            
            for llr_id, similarity in candidates:
                result = self.reason_about_llr(hlr, llr_id, similarity)
                reasoning_results.append(result)
                progress.advance(task)
        
        # Step 3: Classify HLR
        console.print(f"  [cyan]Generating final classification...[/cyan]")
        classification = self.classify_hlr_with_llm(hlr, reasoning_results)
        
        # Format output
        output = self.format_output(hlr, candidates, reasoning_results, classification)
        
        return output
    
    def classify_all_hlrs(self, hlrs: List[Dict], top_k: int = 10, threshold: float = 0.35) -> List[dict]:
        """
        Classify all HLRs
        
        Args:
            hlrs: List of HLR dictionaries
            top_k: Number of candidates to retrieve per HLR
            threshold: Similarity threshold
            
        Returns:
            List of classification results
        """
        results = []
        
        console.print(f"\n[bold cyan]╔═══ Processing {len(hlrs)} HLRs ═══╗[/bold cyan]\n")
        
        for i, hlr in enumerate(hlrs, 1):
            console.print(f"[bold blue]► HLR {i}/{len(hlrs)}: {hlr['id']}[/bold blue]")
            console.print(f"  [dim]{hlr.get('name', 'No name')}[/dim]")
            
            result = self.classify_single_hlr(hlr, top_k, threshold)
            results.append(result)
            
            # Display result
            classification = result['classification']
            confidence = result['confidence_score']
            
            if classification == 'FULLY_TRACED':
                emoji = "✅"
                color = "green"
            elif classification == 'PARTIAL_TRACE':
                emoji = "⚠️"
                color = "yellow"
            else:
                emoji = "❌"
                color = "red"
            
            console.print(f"  [{color}]{emoji} {classification} (confidence: {confidence:.2f})[/{color}]")
            console.print(f"  [dim]{result['reasoning']['summary']}[/dim]")
            
            # Show gap information for PARTIAL_TRACE
            if classification == 'PARTIAL_TRACE':
                gaps = result['reasoning'].get('gaps_identified', [])
                if gaps:
                    console.print(f"  [yellow]⚠ Gaps: {len(gaps)} identified[/yellow]")
                    for gap in gaps[:2]:  # Show first 2 gaps inline
                        console.print(f"    [dim]• {gap[:80]}{'...' if len(gap) > 80 else ''}[/dim]")
            
            # Show critical info for TRACE_HOLE
            elif classification == 'TRACE_HOLE':
                recommendations = result['reasoning'].get('recommendations', [])
                if recommendations:
                    console.print(f"  [red]❗ Action needed: {recommendations[0][:80]}{'...' if len(recommendations[0]) > 80 else ''}[/red]")
            
            console.print("")
        
        console.print(f"[bold cyan]╚════════════════════════╝[/bold cyan]\n")
        
        return results
