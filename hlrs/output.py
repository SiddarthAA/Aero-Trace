"""
Output generation module
"""
import json
from pathlib import Path
from typing import List, Dict
from rich.console import Console

console = Console()


def save_results(results: List[Dict], output_path: str = "dest/hlrs-summary.json"):
    """
    Save classification results to JSON file
    
    Args:
        results: List of HLR classification results
        output_path: Path to save JSON file
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output structure
    output = {
        "metadata": {
            "total_hlrs": len(results),
            "compliant": sum(1 for r in results if r['classification'] == 'COMPLIANT'),
            "weakly_covered": sum(1 for r in results if r['classification'] == 'WEAKLY_COVERED'),
            "non_compliant": sum(1 for r in results if r['classification'] == 'NON_COMPLIANT'),
        },
        "results": results
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ Results saved to {output_path}[/green]")


def print_summary_table(results: List[Dict]):
    """
    Print summary table to console
    
    Args:
        results: List of HLR classification results
    """
    from rich.table import Table
    from rich.panel import Panel
    
    # Create table
    table = Table(title="HLR Classification Summary", show_header=True, header_style="bold cyan")
    table.add_column("HLR ID", style="bold", width=10)
    table.add_column("Classification", width=18)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Risk", justify="center", width=8)
    table.add_column("Linked LLRs", justify="center", width=12)
    table.add_column("Summary", width=50)
    
    for result in results:
        hlr_id = result['req_id']
        classification = result['classification']
        confidence = f"{result['confidence_score']:.2f}"
        risk = result['risk_level']
        llr_count = len(result.get('linked_llrs', []))
        summary = result['reasoning']['summary'][:50] + "..." if len(result['reasoning']['summary']) > 50 else result['reasoning']['summary']
        
        # Color coding
        if classification == 'COMPLIANT':
            classification_text = f"[green]✅ {classification}[/green]"
            risk_text = f"[green]{risk}[/green]"
        elif classification == 'WEAKLY_COVERED':
            classification_text = f"[yellow]⚠️  {classification}[/yellow]"
            risk_text = f"[yellow]{risk}[/yellow]"
        else:
            classification_text = f"[red]❌ {classification}[/red]"
            risk_text = f"[red]{risk}[/red]"
        
        table.add_row(
            hlr_id,
            classification_text,
            confidence,
            risk_text,
            str(llr_count),
            f"[dim]{summary}[/dim]"
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n")


def print_statistics(results: List[Dict]):
    """
    Print overall statistics
    
    Args:
        results: List of HLR classification results
    """
    from rich.panel import Panel
    from rich.columns import Columns
    
    total = len(results)
    compliant = sum(1 for r in results if r['classification'] == 'COMPLIANT')
    weakly = sum(1 for r in results if r['classification'] == 'WEAKLY_COVERED')
    non_compliant = sum(1 for r in results if r['classification'] == 'NON_COMPLIANT')
    
    avg_confidence = sum(r['confidence_score'] for r in results) / total if total > 0 else 0
    
    total_llrs = sum(len(r.get('linked_llrs', [])) for r in results)
    avg_llrs = total_llrs / total if total > 0 else 0
    
    # Create panels
    panels = [
        Panel(
            f"[bold green]{compliant}[/bold green] / {total}\n({compliant/total*100:.1f}%)",
            title="✅ Compliant",
            border_style="green"
        ),
        Panel(
            f"[bold yellow]{weakly}[/bold yellow] / {total}\n({weakly/total*100:.1f}%)",
            title="⚠️  Weakly Covered",
            border_style="yellow"
        ),
        Panel(
            f"[bold red]{non_compliant}[/bold red] / {total}\n({non_compliant/total*100:.1f}%)",
            title="❌ Non-Compliant",
            border_style="red"
        ),
        Panel(
            f"[bold cyan]{avg_confidence:.2f}[/bold cyan]\n(0.0-1.0)",
            title="📊 Avg Confidence",
            border_style="cyan"
        ),
        Panel(
            f"[bold blue]{avg_llrs:.1f}[/bold blue]\n({total_llrs} total)",
            title="🔗 Avg LLRs/HLR",
            border_style="blue"
        ),
    ]
    
    console.print("\n")
    console.print(Columns(panels, equal=True, expand=True))
    console.print("\n")
