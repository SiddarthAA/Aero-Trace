"""
Output formatting and saving for LLR classification results
"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def save_results(results: list, output_path: str):
    """
    Save LLR classification results to JSON file
    
    Args:
        results: List of LLR classification results
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ Results saved to {output_path}[/green]")


def print_summary_table(results: list):
    """
    Print a beautiful summary table of LLR classification results
    
    Args:
        results: List of LLR classification results
    """
    table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
    
    table.add_column("LLR ID", style="cyan", width=12)
    table.add_column("Classification", style="bold", width=20)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Risk", justify="center", width=10)
    table.add_column("Linked HLRs", justify="center", width=14)
    table.add_column("Summary", width=50)
    
    for result in results:
        # Format classification with emoji
        classification = result.get('classification', 'UNKNOWN')
        if classification == 'COMPLIANT':
            class_display = "✅ COMPLIANT"
            style = "green"
        elif classification == 'AMBIGUOUS':
            class_display = "⚠️  AMBIGUOUS"
            style = "yellow"
        elif classification == 'UNJUSTIFIED':
            class_display = "❌ UNJUSTIFIED"
            style = "orange1"
        elif classification == 'ORPHAN':
            class_display = "🔴 ORPHAN"
            style = "red"
        else:
            class_display = classification
            style = "white"
        
        # Format confidence
        confidence = result.get('confidence_score', 0.0)
        conf_display = f"{confidence:.2f}"
        
        # Format risk
        risk = result.get('risk_level', 'UNKNOWN')
        
        # Count linked HLRs
        linked_hlrs = len(result.get('linked_hlrs', []))
        
        # Get summary (truncate if too long)
        summary = result.get('reasoning', {}).get('summary', 'No summary')
        if len(summary) > 50:
            summary = summary[:47] + "..."
        
        table.add_row(
            result['req_id'],
            f"[{style}]{class_display}[/{style}]",
            conf_display,
            risk,
            str(linked_hlrs),
            summary
        )
    
    console.print(table)


def print_statistics(results: list):
    """
    Print statistics about LLR classifications
    
    Args:
        results: List of LLR classification results
    """
    total = len(results)
    compliant = len([r for r in results if r['classification'] == 'COMPLIANT'])
    ambiguous = len([r for r in results if r['classification'] == 'AMBIGUOUS'])
    unjustified = len([r for r in results if r['classification'] == 'UNJUSTIFIED'])
    orphan = len([r for r in results if r['classification'] == 'ORPHAN'])
    
    console.print("\n[bold cyan]📊 LLR Classification Statistics[/bold cyan]\n")
    console.print(f"  Total LLRs: [bold]{total}[/bold]")
    console.print(f"  ✅ Compliant: [green]{compliant}[/green] ({compliant/total*100:.1f}%)")
    console.print(f"  ⚠️  Ambiguous: [yellow]{ambiguous}[/yellow] ({ambiguous/total*100:.1f}%)")
    console.print(f"  ❌ Unjustified: [orange1]{unjustified}[/orange1] ({unjustified/total*100:.1f}%)")
    console.print(f"  🔴 Orphan: [red]{orphan}[/red] ({orphan/total*100:.1f}%)")
    console.print()
