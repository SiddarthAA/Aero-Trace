#!/usr/bin/env python3
"""
Complete Requirements Traceability Analysis Pipeline
Runs both HLR and LLR classification with beautiful output
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint
from rich.layout import Layout
from rich.text import Text

# Load environment variables
load_dotenv()

# Import pipeline modules
from hlrs.embeddings import EmbeddingEngine
from hlrs.groq_client import GroqClient
from hlrs.hlr_classifier import HLRClassifier
from hlrs.visualizer import RequirementsGraph
from hlrs.output import save_results, print_summary_table, print_statistics

from llrs.llr_classifier import LLRClassifier
from llrs.output import save_results as save_llr_results
from llrs.output import print_summary_table as print_llr_summary_table

console = Console()


def print_custom_banner():
    """Print custom ASCII art banner"""
    
    # Custom ASCII art
    ascii_art = """
    
    ███        ▄████████    ▄████████  ▄████████    ▄████████ ▀████    ▐████▀ 
▀█████████▄   ███    ███   ███    ███ ███    ███   ███    ███   ███▌   ████▀  
   ▀███▀▀██   ███    ███   ███    ███ ███    █▀    ███    █▀     ███  ▐███    
    ███   ▀  ▄███▄▄▄▄██▀   ███    ███ ███         ▄███▄▄▄        ▀███▄███▀    
    ███     ▀▀███▀▀▀▀▀   ▀███████████ ███        ▀▀███▀▀▀        ████▀██▄     
    ███     ▀███████████   ███    ███ ███    █▄    ███    █▄    ▐███  ▀███    
    ███       ███    ███   ███    ███ ███    ███   ███    ███  ▄███     ███▄  
   ▄████▀     ███    ███   ███    █▀  ████████▀    ██████████ ████       ███▄ 
              ███    ███                                                      
                                                                                                               
      Semantic Analysis • LLM Reasoning • Bidirectional Classification             
                   For Safety-Critical System Requirements                          
                                                                                    
    """
    
    console.print(ascii_art, style="bold cyan")
    
    # System info
    info_text = Text()
    info_text.append("    🚀 ", style="bold yellow")
    info_text.append("System Ready", style="bold white")
    info_text.append(" • ", style="dim")
    info_text.append("HLR Pipeline", style="bold green")
    info_text.append(" + ", style="dim")
    info_text.append("LLR Pipeline", style="bold blue")
    info_text.append(" • ", style="dim")
    info_text.append(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    
    console.print(info_text)
    console.print()


def print_phase_header(phase_num: int, total: int, title: str, emoji: str):
    """Print beautiful phase header"""
    console.print()
    console.print(f"[bold cyan]{'═' * 80}[/bold cyan]")
    console.print(f"[bold cyan]║[/bold cyan] {emoji} [bold white]PHASE {phase_num}/{total}: {title}[/bold white]")
    console.print(f"[bold cyan]{'═' * 80}[/bold cyan]")
    console.print()


def load_requirements(hlrs_path: str, llrs_path: str):
    """Load HLRs and LLRs from JSON files"""
    console.print("[cyan]📥 Loading requirements files...[/cyan]")
    
    with open(hlrs_path, 'r', encoding='utf-8') as f:
        hlrs = json.load(f)
    console.print(f"  [green]✓[/green] Loaded [bold]{len(hlrs)} HLRs[/bold] from {hlrs_path}")
    
    with open(llrs_path, 'r', encoding='utf-8') as f:
        llrs = json.load(f)
    console.print(f"  [green]✓[/green] Loaded [bold]{len(llrs)} LLRs[/bold] from {llrs_path}")
    
    return hlrs, llrs


def initialize_clients():
    """Initialize embedding engine and LLM clients"""
    console.print("\n[cyan]🔧 Initializing AI models and clients...[/cyan]")
    
    # Initialize embedding engine
    with console.status("[cyan]Loading embedding model...", spinner="dots"):
        embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    console.print("  [green]✓[/green] Embedding model loaded: [bold]all-MiniLM-L6-v2[/bold]")
    
    # Initialize Groq client
    with console.status("[cyan]Connecting to Groq API...", spinner="dots"):
        groq_client = GroqClient()
    console.print("  [green]✓[/green] Groq client connected")
    console.print("    [dim]→ Reasoning: openai/gpt-oss-120b[/dim]")
    console.print("    [dim]→ Structured: llama-3.1-8b-instant[/dim]")
    
    return embedding_engine, groq_client


def run_hlr_pipeline(hlrs, llrs, embedding_engine, groq_client):
    """Run HLR classification pipeline"""
    print_phase_header(1, 2, "HLR CLASSIFICATION", "🎯")
    
    console.print("[bold]Building vector index for LLRs...[/bold]")
    embedding_engine.build_llr_index(llrs)
    console.print(f"  [green]✓[/green] FAISS index built with [bold]{len(llrs)}[/bold] LLR vectors\n")
    
    console.print("[bold]Initializing HLR classifier...[/bold]")
    hlr_classifier = HLRClassifier(embedding_engine, groq_client)
    hlr_classifier.set_llrs(llrs)
    console.print("  [green]✓[/green] HLR classifier ready\n")
    
    console.print(f"[bold yellow]▶ Processing {len(hlrs)} HLRs...[/bold yellow]\n")
    hlr_results = hlr_classifier.classify_all_hlrs(hlrs, top_k=10, threshold=0.30)
    
    console.print("\n[green]✅ HLR classification complete![/green]\n")
    
    return hlr_results


def run_llr_pipeline(hlrs, llrs, embedding_engine, groq_client):
    """Run LLR classification pipeline"""
    print_phase_header(2, 2, "LLR CLASSIFICATION", "🔍")
    
    console.print("[bold]Building vector index for HLRs...[/bold]")
    embedding_engine.build_hlr_index(hlrs)
    console.print(f"  [green]✓[/green] FAISS index built with [bold]{len(hlrs)}[/bold] HLR vectors\n")
    
    console.print("[bold]Initializing LLR classifier...[/bold]")
    llr_classifier = LLRClassifier(embedding_engine, groq_client)
    llr_classifier.set_hlrs(hlrs)
    console.print("  [green]✓[/green] LLR classifier ready\n")
    
    console.print(f"[bold yellow]▶ Processing {len(llrs)} LLRs...[/bold yellow]\n")
    llr_results = llr_classifier.classify_all_llrs(llrs, top_n=5, threshold=0.30)
    
    console.print("\n[green]✅ LLR classification complete![/green]\n")
    
    return llr_results


def save_all_results(hlr_results, llr_results, output_dir: str):
    """Save all results to JSON files"""
    console.print("\n[bold cyan]💾 Saving Results[/bold cyan]\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save HLR results
    hlr_output = output_path / "hlrs-summary.json"
    save_results(hlr_results, str(hlr_output))
    console.print(f"  [green]✓[/green] HLR results → [bold]{hlr_output}[/bold]")
    
    # Save LLR results
    llr_output = output_path / "llrs-summary.json"
    save_llr_results(llr_results, str(llr_output))
    console.print(f"  [green]✓[/green] LLR results → [bold]{llr_output}[/bold]")
    
    # Save combined summary
    combined = {
        "generated_at": datetime.now().isoformat(),
        "hlr_analysis": {
            "total_hlrs": len(hlr_results),
            "fully_traced": len([r for r in hlr_results if r['classification'] == 'FULLY_TRACED']),
            "partial_trace": len([r for r in hlr_results if r['classification'] == 'PARTIAL_TRACE']),
            "trace_hole": len([r for r in hlr_results if r['classification'] == 'TRACE_HOLE'])
        },
        "llr_analysis": {
            "total_llrs": len(llr_results),
            "compliant": len([r for r in llr_results if r['classification'] == 'COMPLIANT']),
            "ambiguous": len([r for r in llr_results if r['classification'] == 'AMBIGUOUS']),
            "unjustified": len([r for r in llr_results if r['classification'] == 'UNJUSTIFIED']),
            "orphan": len([r for r in llr_results if r['classification'] == 'ORPHAN'])
        },
        "hlr_results": hlr_results,
        "llr_results": llr_results
    }
    
    combined_output = output_path / "complete-traceability-analysis.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    console.print(f"  [green]✓[/green] Combined results → [bold]{combined_output}[/bold]")


def generate_visualizations(hlr_results, llr_results, output_dir: str):
    """Generate all visualizations"""
    console.print("\n[bold cyan]📊 Generating Visualizations[/bold cyan]\n")
    
    output_path = Path(output_dir)
    
    try:
        # Generate requirements graph using build_graph method
        console.print("  [cyan]Creating requirements traceability graph...[/cyan]")
        graph = RequirementsGraph()
        graph.build_graph(hlr_results)
        
        # Save main graph
        graph_output = output_path / "requirements-traceability-graph.png"
        graph.visualize(str(graph_output))
        console.print(f"  [green]✓[/green] Traceability graph → [bold]{graph_output}[/bold]")
        
        # Generate classification-specific visualizations
        console.print("  [cyan]Creating classification summary charts...[/cyan]")
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # HLR classifications pie chart
        hlr_compliant = len([r for r in hlr_results if r['classification'] == 'FULLY_TRACED'])
        hlr_weakly = len([r for r in hlr_results if r['classification'] == 'PARTIAL_TRACE'])
        hlr_non = len([r for r in hlr_results if r['classification'] == 'TRACE_HOLE'])
        
        ax1.pie(
            [hlr_compliant, hlr_weakly, hlr_non],
            labels=['Fully Traced', 'Partial Trace', 'Trace Hole'],
            colors=['#00ff00', '#ffaa00', '#ff0000'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('HLR Classifications', fontsize=14, fontweight='bold')
        
        # LLR classifications pie chart
        llr_compliant = len([r for r in llr_results if r['classification'] == 'COMPLIANT'])
        llr_ambiguous = len([r for r in llr_results if r['classification'] == 'AMBIGUOUS'])
        llr_unjustified = len([r for r in llr_results if r['classification'] == 'UNJUSTIFIED'])
        llr_orphan = len([r for r in llr_results if r['classification'] == 'ORPHAN'])
        
        ax2.pie(
            [llr_compliant, llr_ambiguous, llr_unjustified, llr_orphan],
            labels=['Compliant', 'Ambiguous', 'Unjustified', 'Orphan'],
            colors=['#00ff00', '#ffaa00', '#ff6600', '#ff0000'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('LLR Classifications', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        chart_output = output_path / "classification-summary.png"
        plt.savefig(chart_output, dpi=150, bbox_inches='tight')
        plt.close()
        
        console.print(f"  [green]✓[/green] Classification charts → [bold]{chart_output}[/bold]")
        
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow] Could not generate visualizations: {e}")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")


def print_final_summary(hlr_results, llr_results):
    """Print beautiful final summary"""
    console.print("\n" + "═" * 80)
    console.print("[bold cyan]📈 FINAL ANALYSIS SUMMARY[/bold cyan]")
    console.print("═" * 80 + "\n")
    
    # Create summary table
    table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("HLRs", justify="center", style="green")
    table.add_column("LLRs", justify="center", style="blue")
    
    # HLR stats
    hlr_fully_traced = len([r for r in hlr_results if r['classification'] == 'FULLY_TRACED'])
    hlr_partial = len([r for r in hlr_results if r['classification'] == 'PARTIAL_TRACE'])
    hlr_hole = len([r for r in hlr_results if r['classification'] == 'TRACE_HOLE'])
    
    # LLR stats
    llr_compliant = len([r for r in llr_results if r['classification'] == 'COMPLIANT'])
    llr_ambiguous = len([r for r in llr_results if r['classification'] == 'AMBIGUOUS'])
    llr_unjustified = len([r for r in llr_results if r['classification'] == 'UNJUSTIFIED'])
    llr_orphan = len([r for r in llr_results if r['classification'] == 'ORPHAN'])
    
    table.add_row("Total Requirements", str(len(hlr_results)), str(len(llr_results)))
    table.add_row("✅ Fully Traced / Compliant", f"{hlr_fully_traced} ({hlr_fully_traced/len(hlr_results)*100:.1f}%)", 
                  f"{llr_compliant} ({llr_compliant/len(llr_results)*100:.1f}%)")
    table.add_row("⚠️  Partial Trace / Ambiguous", f"{hlr_partial} ({hlr_partial/len(hlr_results)*100:.1f}%)", 
                  f"{llr_ambiguous} ({llr_ambiguous/len(llr_results)*100:.1f}%)")
    table.add_row("❌ Trace Hole / Unjustified", f"{hlr_hole} ({hlr_hole/len(hlr_results)*100:.1f}%)", 
                  f"{llr_unjustified} ({llr_unjustified/len(llr_results)*100:.1f}%)")
    table.add_row("🔴 Orphan", "-", f"{llr_orphan} ({llr_orphan/len(llr_results)*100:.1f}%)")
    
    console.print(table)
    console.print()


def print_completion_message(output_dir: str):
    """Print final completion message"""
    console.print("\n" + "═" * 80)
    completion_panel = Panel(
        f"""[bold green]✨ Analysis Complete! ✨[/bold green]

[cyan]📁 Results Location:[/cyan]
  → {output_dir}/

[cyan]📄 Generated Files:[/cyan]
  → hlrs-summary.json                    (HLR classifications)
  → llrs-summary.json                    (LLR classifications)
  → complete-traceability-analysis.json  (Combined results)
  → requirements-traceability-graph.png  (Visual graph)

[cyan]📖 Next Steps:[/cyan]
  1. Review classification results in JSON files
  2. Examine the traceability graph visualization
  3. Address non-compliant and orphan requirements
  4. Update requirements based on findings

[yellow]💡 Tip:[/yellow] Use 'jq' to explore JSON files:
  jq '.hlr_analysis' {output_dir}/complete-traceability-analysis.json
""",
        title="[bold green]SUCCESS[/bold green]",
        border_style="green",
        padding=(1, 2)
    )
    console.print(completion_panel)
    console.print()


def main():
    """Main execution flow"""
    start_time = datetime.now()
    
    try:
        # Print custom banner
        print_custom_banner()
        
        # Configuration
        HLRS_PATH = "src/hlrs.json"
        LLRS_PATH = "src/llrs.json"
        OUTPUT_DIR = "dest"
        
        # Phase 0: Load requirements
        print_phase_header(0, 2, "INITIALIZATION", "🚀")
        hlrs, llrs = load_requirements(HLRS_PATH, LLRS_PATH)
        embedding_engine, groq_client = initialize_clients()
        
        # Phase 1: Run HLR pipeline
        hlr_results = run_hlr_pipeline(hlrs, llrs, embedding_engine, groq_client)
        
        # Phase 2: Run LLR pipeline
        llr_results = run_llr_pipeline(hlrs, llrs, embedding_engine, groq_client)
        
        # Save all results
        save_all_results(hlr_results, llr_results, OUTPUT_DIR)
        
        # Generate visualizations
        generate_visualizations(hlr_results, llr_results, OUTPUT_DIR)
        
        # Print detailed summaries
        console.print("\n[bold cyan]═══ HLR CLASSIFICATION RESULTS ═══[/bold cyan]\n")
        print_summary_table(hlr_results)
        
        console.print("\n[bold cyan]═══ LLR CLASSIFICATION RESULTS ═══[/bold cyan]\n")
        print_llr_summary_table(llr_results)
        
        # Print final summary
        print_final_summary(hlr_results, llr_results)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        console.print(f"[dim]⏱️  Total execution time: {duration:.1f} seconds[/dim]\n")
        
        # Print completion message
        print_completion_message(OUTPUT_DIR)
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Pipeline interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
