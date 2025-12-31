"""
Visualization module for HLR-LLR relationships
Creates Neo4j-style weighted graph
"""
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict
from rich.console import Console

console = Console()


class RequirementsGraph:
    """Visualize HLR-LLR relationships as a weighted graph"""
    
    def __init__(self):
        """Initialize graph"""
        self.G = nx.DiGraph()
    
    def build_graph(self, results: List[Dict]):
        """
        Build graph from classification results
        
        Args:
            results: List of HLR classification results
        """
        console.print("[cyan]Building requirements graph...[/cyan]")
        
        for hlr_result in results:
            hlr_id = hlr_result['req_id']
            classification = hlr_result['classification']
            
            # Add HLR node with classification
            self.G.add_node(
                hlr_id,
                node_type='HLR',
                classification=classification,
                risk=hlr_result.get('risk_level', 'UNKNOWN')
            )
            
            # Add LLR edges
            for llr_link in hlr_result.get('linked_llrs', []):
                llr_id = llr_link['llr_id']
                weight = llr_link.get('weight', llr_link.get('similarity_score', 0))
                relationship = llr_link.get('relationship', 'relates_to')
                
                # Add LLR node
                if llr_id not in self.G:
                    self.G.add_node(llr_id, node_type='LLR')
                
                # Add weighted edge
                self.G.add_edge(
                    hlr_id,
                    llr_id,
                    weight=weight,
                    relationship=relationship
                )
        
        console.print(f"[green]✓ Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges[/green]")
    
    def visualize(self, output_path: str = "dest/requirements_graph.png"):
        """
        Create and save graph visualization
        
        Args:
            output_path: Path to save PNG file
        """
        console.print("[cyan]Generating graph visualization...[/cyan]")
        
        plt.figure(figsize=(16, 12))
        
        # Layout
        pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        
        # Separate HLR and LLR nodes
        hlr_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'HLR']
        llr_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'LLR']
        
        # Color HLRs by classification
        hlr_colors = []
        for node in hlr_nodes:
            classification = self.G.nodes[node].get('classification', 'UNKNOWN')
            if classification == 'FULLY_TRACED':
                hlr_colors.append('#2ecc71')  # Green
            elif classification == 'PARTIAL_TRACE':
                hlr_colors.append('#f39c12')  # Orange
            elif classification == 'TRACE_HOLE':
                hlr_colors.append('#e74c3c')  # Red
            else:
                hlr_colors.append('#95a5a6')  # Gray
        
        # Draw HLR nodes (larger, squares)
        nx.draw_networkx_nodes(
            self.G, pos,
            nodelist=hlr_nodes,
            node_color=hlr_colors,
            node_size=2000,
            node_shape='s',
            alpha=0.9,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw LLR nodes (smaller, circles)
        nx.draw_networkx_nodes(
            self.G, pos,
            nodelist=llr_nodes,
            node_color='#3498db',  # Blue
            node_size=1000,
            node_shape='o',
            alpha=0.8,
            edgecolors='black',
            linewidths=1.5
        )
        
        # Draw edges with varying thickness based on weight
        edges = self.G.edges(data=True)
        edge_widths = [d['weight'] * 5 for _, _, d in edges]
        edge_colors = []
        
        for _, _, d in edges:
            weight = d['weight']
            if weight > 0.75:
                edge_colors.append('#2ecc71')  # Green - strong
            elif weight > 0.50:
                edge_colors.append('#f39c12')  # Orange - medium
            else:
                edge_colors.append('#e74c3c')  # Red - weak
        
        nx.draw_networkx_edges(
            self.G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.G, pos,
            font_size=9,
            font_weight='bold',
            font_color='white'
        )
        
        # Title and legend
        plt.title("HLR-LLR Traceability Graph", fontsize=18, fontweight='bold', pad=20)
        
        # Create legend
        from matplotlib.patches import Patch, Rectangle, Circle
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc='#2ecc71', ec='black', label='HLR: Fully Traced'),
            Rectangle((0, 0), 1, 1, fc='#f39c12', ec='black', label='HLR: Partial Trace'),
            Rectangle((0, 0), 1, 1, fc='#e74c3c', ec='black', label='HLR: Trace Hole'),
            Circle((0, 0), 1, fc='#3498db', ec='black', label='LLR'),
            Patch(facecolor='#2ecc71', alpha=0.6, label='Strong Link (>0.75)'),
            Patch(facecolor='#f39c12', alpha=0.6, label='Medium Link (0.50-0.75)'),
            Patch(facecolor='#e74c3c', alpha=0.6, label='Weak Link (<0.50)'),
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        console.print(f"[green]✓ Graph saved to {output_path}[/green]")
    
    def get_statistics(self) -> dict:
        """
        Get graph statistics
        
        Returns:
            Dictionary of statistics
        """
        hlr_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'HLR']
        llr_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'LLR']
        
        # Count by classification
        fully_traced = sum(1 for n in hlr_nodes if self.G.nodes[n].get('classification') == 'FULLY_TRACED')
        partial_trace = sum(1 for n in hlr_nodes if self.G.nodes[n].get('classification') == 'PARTIAL_TRACE')
        trace_hole = sum(1 for n in hlr_nodes if self.G.nodes[n].get('classification') == 'TRACE_HOLE')
        
        # Edge statistics
        edges = list(self.G.edges(data=True))
        strong_edges = sum(1 for _, _, d in edges if d['weight'] > 0.75)
        medium_edges = sum(1 for _, _, d in edges if 0.50 < d['weight'] <= 0.75)
        weak_edges = sum(1 for _, _, d in edges if d['weight'] <= 0.50)
        
        return {
            'total_hlrs': len(hlr_nodes),
            'total_llrs': len(llr_nodes),
            'fully_traced_hlrs': fully_traced,
            'partial_trace_hlrs': partial_trace,
            'trace_hole_hlrs': trace_hole,
            'total_edges': len(edges),
            'strong_edges': strong_edges,
            'medium_edges': medium_edges,
            'weak_edges': weak_edges
        }
