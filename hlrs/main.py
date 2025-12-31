#!/usr/bin/env python3
"""
HLR Classification Pipeline - Main CLI
Beautiful, feature-rich requirements traceability analysis
"""
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

# Load environment variables
load_dotenv()

# Import pipeline modules
from hlrs.embeddings import EmbeddingEngine
from hlrs.groq_client import GroqClient
from hlrs.hlr_classifier import HLRClassifier
from hlrs.visualizer import RequirementsGraph
from hlrs.output import save_results, print_summary_table, print_statistics

console = Console()


def print_banner():
    """Print beautiful ASCII banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🎯 HLR CLASSIFICATION PIPELINE                                     ║
║   Semantic Traceability Analysis for Safety-Critical Requirements   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def load_requirements(hlrs_path: str, llrs_path: str):
    """
    Load HLRs and LLRs from JSON files
    
    Args:
        hlrs_path: Path to HLRs JSON file
        llrs_path: Path to LLRs JSON file
        
    Returns:
        Tuple of (hlrs, llrs)
    """
    console.print("\n[bold cyan]═══ Loading Requirements ═══[/bold cyan]\n")
    
    with console.status("[cyan]Loading HLRs...", spinner="dots"):
        with open(hlrs_path, 'r', encoding='utf-8') as f:
            hlrs = json.load(f)
    console.print(f"[green]✓ Loaded {len(hlrs)} HLRs from {hlrs_path}[/green]")
    
    with console.status("[cyan]Loading LLRs...", spinner="dots"):
        with open(llrs_path, 'r', encoding='utf-8') as f:
            llrs = json.load(f)
    console.print(f"[green]✓ Loaded {len(llrs)} LLRs from {llrs_path}[/green]")
    
    return hlrs, llrs


def initialize_pipeline():
    """
    Initialize all pipeline components
    
    Returns:
        Tuple of (embedding_engine, groq_client)
    """
    console.print("\n[bold cyan]═══ Initializing Pipeline ═══[/bold cyan]\n")
    
    # Initialize embedding engine
    embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    # Initialize Groq client (handles both reasoning and structured output)
    with console.status("[cyan]Connecting to Groq API...", spinner="dots"):
        groq_client = GroqClient()
    console.print("[green]✓ Groq client initialized[/green]")
    console.print("[dim]  - Reasoning model: openai/gpt-oss-120b[/dim]")
    console.print("[dim]  - Structured output: llama-3.1-8b-instant[/dim]")
    
    return embedding_engine, groq_client


def main():
    """Main execution flow"""
    try:
        # Print banner
        print_banner()
        
        # Configuration
        HLRS_PATH = "src/hlrs.json"
        LLRS_PATH = "src/llrs.json"
        OUTPUT_PATH = "dest/hlrs-summary.json"
        GRAPH_PATH = "dest/requirements_graph.png"
        
        # Step 1: Load requirements
        hlrs, llrs = load_requirements(HLRS_PATH, LLRS_PATH)
        
        # Step 2: Initialize pipeline
        embedding_engine, groq_client = initialize_pipeline()
        
        # Step 3: Build FAISS index
        console.print("\n[bold cyan]═══ Building Vector Index ═══[/bold cyan]\n")
        embedding_engine.build_llr_index(llrs)
        
        # Step 4: Initialize classifier
        console.print("\n[bold cyan]═══ Initializing Classifier ═══[/bold cyan]\n")
        classifier = HLRClassifier(embedding_engine, groq_client)
        classifier.set_llrs(llrs)
        console.print("[green]✓ Classifier ready[/green]")
        
        # Step 5: Classify all HLRs
        console.print("\n[bold cyan]═══ Running Classification Pipeline ═══[/bold cyan]")
        results = classifier.classify_all_hlrs(
            hlrs,
            top_k=10,
            threshold=0.30  # Lower threshold to catch more potential relationships
        )
        
        # Step 6: Save results
        console.print("\n[bold cyan]═══ Saving Results ═══[/bold cyan]\n")
        save_results(results, OUTPUT_PATH)
        
        # Step 7: Generate visualization
        console.print("\n[bold cyan]═══ Generating Visualization ═══[/bold cyan]\n")
        graph = RequirementsGraph()
        graph.build_graph(results)
        graph.visualize(GRAPH_PATH)
        
        # Step 8: Print summary
        console.print("\n[bold cyan]═══ Analysis Complete ═══[/bold cyan]")
        print_statistics(results)
        print_summary_table(results)
        
        # Final message
        console.print(Panel.fit(
            f"""[bold green]✓ Pipeline execution complete![/bold green]

📊 Results saved to: [cyan]{OUTPUT_PATH}[/cyan]
📈 Graph saved to: [cyan]{GRAPH_PATH}[/cyan]

[dim]Total HLRs analyzed: {len(results)}
Total reasoning calls: {sum(len(r.get('linked_llrs', [])) for r in results)}[/dim]
            """,
            title="✨ Success",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Pipeline interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
