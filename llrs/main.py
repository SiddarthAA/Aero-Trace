#!/usr/bin/env python3
"""
LLR Classification Pipeline - Main CLI
Analyzes LLRs in isolation to determine their justification
"""
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Load environment variables
load_dotenv()

# Import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from hlrs.embeddings import EmbeddingEngine
from hlrs.groq_client import GroqClient
from llrs.llr_classifier import LLRClassifier

console = Console()


def print_banner():
    """Print beautiful ASCII banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🔍 LLR CLASSIFICATION PIPELINE                                     ║
║   Analyzing Low-Level Requirements for System Justification         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold magenta")


def load_requirements(hlrs_path: str, llrs_path: str):
    """
    Load HLRs and LLRs from JSON files
    
    Args:
        hlrs_path: Path to HLRs JSON file
        llrs_path: Path to LLRs JSON file
        
    Returns:
        Tuple of (hlrs, llrs)
    """
    console.print("\n[bold magenta]═══ Loading Requirements ═══[/bold magenta]\n")
    
    with console.status("[magenta]Loading HLRs...", spinner="dots"):
        with open(hlrs_path, 'r', encoding='utf-8') as f:
            hlrs = json.load(f)
    console.print(f"[green]✓ Loaded {len(hlrs)} HLRs from {hlrs_path}[/green]")
    
    with console.status("[magenta]Loading LLRs...", spinner="dots"):
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
    console.print("\n[bold magenta]═══ Initializing Pipeline ═══[/bold magenta]\n")
    
    # Initialize embedding engine
    embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    # Initialize Groq client
    with console.status("[magenta]Connecting to Groq API...", spinner="dots"):
        groq_client = GroqClient()
    console.print("[green]✓ Groq client initialized[/green]")
    console.print("[dim]  - Reasoning model: openai/gpt-oss-120b[/dim]")
    console.print("[dim]  - Structured output: llama-3.1-8b-instant[/dim]")
    
    return embedding_engine, groq_client


def print_summary_table(results: list):
    """Print summary table of LLR classifications"""
    
    # Count by classification
    counts = {
        'COMPLIANT': 0,
        'AMBIGUOUS': 0,
        'UNJUSTIFIED': 0,
        'ORPHAN': 0
    }
    
    for result in results:
        cls = result['classification']
        counts[cls] = counts.get(cls, 0) + 1
    
    # Create table
    table = Table(title="\n📊 LLR Classification Summary", show_header=True, header_style="bold magenta")
    table.add_column("LLR ID", style="cyan", width=12)
    table.add_column("Classification", width=20)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Risk", justify="center", width=10)
    table.add_column("Primary HLR", width=12)
    table.add_column("Summary", width=50)
    
    for result in results:
        llr_id = result['req_id']
        cls = result['classification']
        conf = result['confidence_score']
        risk = result['risk_level']
        primary = result['reasoning'].get('primary_hlr', '-')
        summary = result['reasoning']['summary'][:47] + "..." if len(result['reasoning']['summary']) > 50 else result['reasoning']['summary']
        
        # Color based on classification
        if cls == "COMPLIANT":
            cls_display = f"[green]✅ {cls}[/green]"
            risk_display = f"[green]{risk}[/green]"
        elif cls == "ORPHAN":
            cls_display = f"[red]❌ {cls}[/red]"
            risk_display = f"[red]{risk}[/red]"
        elif cls == "UNJUSTIFIED":
            cls_display = f"[red]❌ {cls}[/red]"
            risk_display = f"[yellow]{risk}[/yellow]"
        else:  # AMBIGUOUS
            cls_display = f"[yellow]⚠️  {cls}[/yellow]"
            risk_display = f"[yellow]{risk}[/yellow]"
        
        table.add_row(
            llr_id,
            cls_display,
            f"{conf:.2f}",
            risk_display,
            primary or "-",
            summary
        )
    
    console.print(table)
    
    # Print statistics
    console.print(f"\n[bold magenta]═══ Statistics ═══[/bold magenta]\n")
    console.print(f"[green]✅ Compliant:[/green]       {counts['COMPLIANT']}/{len(results)}")
    console.print(f"[yellow]⚠️  Ambiguous:[/yellow]      {counts['AMBIGUOUS']}/{len(results)}")
    console.print(f"[red]❌ Unjustified:[/red]    {counts['UNJUSTIFIED']}/{len(results)}")
    console.print(f"[red]❌ Orphan:[/red]         {counts['ORPHAN']}/{len(results)}")


def save_results(results: list, output_path: str):
    """Save results to JSON file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ Results saved to {output_path}[/green]")


def main():
    """Main execution flow"""
    try:
        # Print banner
        print_banner()
        
        # Configuration
        HLRS_PATH = "../src/hlrs.json"
        LLRS_PATH = "../src/llrs.json"
        OUTPUT_PATH = "../dist/llrs-summary.json"
        
        # Step 1: Load requirements
        hlrs, llrs = load_requirements(HLRS_PATH, LLRS_PATH)
        
        # Step 2: Initialize pipeline
        embedding_engine, groq_client = initialize_pipeline()
        
        # Step 3: Build HLR index (LLRs search against HLRs)
        console.print("\n[bold magenta]═══ Building Vector Index ═══[/bold magenta]\n")
        embedding_engine.build_hlr_index(hlrs)
        
        # Step 4: Initialize LLR classifier
        console.print("\n[bold magenta]═══ Initializing Classifier ═══[/bold magenta]\n")
        classifier = LLRClassifier(embedding_engine, groq_client)
        classifier.set_hlrs(hlrs)
        console.print("[green]✓ LLR Classifier ready[/green]")
        
        # Step 5: Classify all LLRs
        console.print("\n[bold magenta]═══ Running Classification Pipeline ═══[/bold magenta]")
        results = classifier.classify_all_llrs(
            llrs,
            top_n=10,
            threshold=0.30
        )
        
        # Step 6: Save results
        console.print("\n[bold magenta]═══ Saving Results ═══[/bold magenta]\n")
        save_results(results, OUTPUT_PATH)
        
        # Step 7: Print summary
        print_summary_table(results)
        
        # Final message
        console.print("\n[bold green]✅ LLR Classification Complete![/bold green]\n")
        console.print(f"[dim]Results saved to: {OUTPUT_PATH}[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠ Pipeline interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
