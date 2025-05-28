"""
Citation network analysis module for research papers.
"""

import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class CitationAnalysis:
    """
    Class for analyzing citation networks and relationships between papers.
    """
    
    def __init__(self):
        """Initialize citation analysis."""
        self.papers = {}  # Paper ID -> paper data
        self.citation_graph = nx.DiGraph()
        self.reference_patterns = [
            r'\[\s*(\d+)\s*\]',  # [1], [2], etc.
            r'\(\s*(\w+)\s*,\s*(\d{4})\s*\)',  # (Author, 2020), (Smith, 2019), etc.
            r'(\w+)\s+et\s+al\.\s*\(\s*(\d{4})\s*\)'  # Smith et al. (2020), etc.
        ]
    
    def add_paper(self, paper_id, title, authors, year=None, abstract=None, references=None):
        """
        Add a paper to the citation network.
        
        Args:
            paper_id (str): Unique identifier for the paper
            title (str): Paper title
            authors (list): List of authors
            year (int, optional): Publication year
            abstract (str, optional): Paper abstract
            references (list, optional): List of reference strings or paper IDs
        """
        # Store paper data
        self.papers[paper_id] = {
            'id': paper_id,
            'title': title,
            'authors': authors,
            'year': year,
            'abstract': abstract,
            'references': references or []
        }
        
        # Add node to citation graph
        self.citation_graph.add_node(
            paper_id,
            title=title,
            authors=authors,
            year=year
        )
        
        # Process references if provided
        if references:
            for ref in references:
                # Add edge from paper to reference
                if ref in self.papers:
                    self.citation_graph.add_edge(paper_id, ref)
    
    def extract_references_from_text(self, text):
        """
        Extract references from text using patterns.
        
        Args:
            text (str): Text containing references
            
        Returns:
            list: Extracted reference strings
        """
        references = []
        
        # Apply each pattern
        for pattern in self.reference_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle tuple matches (e.g., author, year)
                    for match in matches:
                        ref_str = ' '.join(match)
                        references.append(ref_str)
                else:
                    # Handle simple matches
                    references.extend(matches)
        
        return references
    
    def build_citation_network(self):
        """
        Build the citation network from papers and their references.
        
        Returns:
            networkx.DiGraph: Citation network graph
        """
        # Clear existing graph
        self.citation_graph = nx.DiGraph()
        
        # Add nodes for all papers
        for paper_id, paper_data in self.papers.items():
            self.citation_graph.add_node(
                paper_id,
                title=paper_data['title'],
                authors=paper_data['authors'],
                year=paper_data['year']
            )
        
        # Add edges for citations
        for paper_id, paper_data in self.papers.items():
            for ref in paper_data['references']:
                # Check if reference is a valid paper ID
                if ref in self.papers:
                    self.citation_graph.add_edge(paper_id, ref)
                else:
                    # Try to match reference with existing papers
                    # This is a simplified approach; in practice, more sophisticated matching would be needed
                    for candidate_id, candidate_data in self.papers.items():
                        # Skip self-comparisons
                        if candidate_id == paper_id:
                            continue
                            
                        # Check if candidate title contains the reference
                        if ref.lower() in candidate_data['title'].lower():
                            self.citation_graph.add_edge(paper_id, candidate_id)
                            break
        
        return self.citation_graph
    
    def get_citation_stats(self):
        """
        Calculate citation statistics.
        
        Returns:
            dict: Citation statistics
        """
        if not self.citation_graph.nodes():
            return {
                'num_papers': 0,
                'num_citations': 0,
                'avg_citations': 0,
                'most_cited': None,
                'citation_counts': {}
            }
            
        # Calculate in-degree (number of citations) for each paper
        citation_counts = dict(self.citation_graph.in_degree())
        
        # Get most cited paper
        most_cited = max(citation_counts.items(), key=lambda x: x[1]) if citation_counts else (None, 0)
        
        # Calculate average citations
        total_citations = sum(citation_counts.values())
        avg_citations = total_citations / len(citation_counts) if citation_counts else 0
        
        return {
            'num_papers': self.citation_graph.number_of_nodes(),
            'num_citations': self.citation_graph.number_of_edges(),
            'avg_citations': avg_citations,
            'most_cited': most_cited,
            'citation_counts': citation_counts
        }
    
    def identify_key_papers(self, top_n=5, algorithm='pagerank'):
        """
        Identify key papers in the citation network using various centrality measures.
        
        Args:
            top_n (int): Number of top papers to return
            algorithm (str): Centrality algorithm to use ('pagerank', 'eigenvector', 'betweenness')
            
        Returns:
            list: Top papers according to the selected centrality measure
        """
        if not self.citation_graph.nodes():
            return []
            
        if algorithm == 'pagerank':
            centrality = nx.pagerank(self.citation_graph)
        elif algorithm == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality(self.citation_graph)
            except nx.PowerIterationFailedConvergence:
                # Fall back to pagerank if eigenvector centrality fails
                centrality = nx.pagerank(self.citation_graph)
        elif algorithm == 'betweenness':
            centrality = nx.betweenness_centrality(self.citation_graph)
        else:
            centrality = nx.pagerank(self.citation_graph)
        
        # Sort papers by centrality score
        sorted_papers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N papers
        top_papers = []
        for paper_id, score in sorted_papers[:top_n]:
            if paper_id in self.papers:
                paper_data = self.papers[paper_id].copy()
                paper_data['centrality_score'] = score
                top_papers.append(paper_data)
        
        return top_papers
    
    def visualize_citation_network(self, highlight_nodes=None, node_size_factor=100, figsize=(12, 10)):
        """
        Visualize the citation network.
        
        Args:
            highlight_nodes (list, optional): List of node IDs to highlight
            node_size_factor (int): Factor to scale node sizes by citation count
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Citation network visualization
        """
        if not self.citation_graph.nodes():
            plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, "No citation data available", ha='center', va='center')
            plt.axis('off')
            return plt.gcf()
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Calculate node sizes based on citation counts (in-degree)
        in_degrees = dict(self.citation_graph.in_degree())
        node_sizes = [max(10, in_degrees.get(node, 0) * node_size_factor) for node in self.citation_graph.nodes()]
        
        # Determine node colors
        if highlight_nodes:
            node_colors = [
                'red' if node in highlight_nodes else 'skyblue' for node in self.citation_graph.nodes()
            ]
        else:
            # Color nodes by year if available
            node_years = [self.papers.get(node, {}).get('year', 0) for node in self.citation_graph.nodes()]
            if any(node_years):
                # Normalize years to a color scale
                min_year = min([y for y in node_years if y > 0]) if any(y > 0 for y in node_years) else 0
                max_year = max(node_years) if max(node_years) > min_year else min_year + 10
                
                # Create color map
                node_colors = [
                    plt.cm.viridis((year - min_year) / max(1, max_year - min_year)) if year > 0 else 'lightgray'
                    for year in node_years
                ]
            else:
                node_colors = 'skyblue'
        
        # Create layout
        pos = nx.spring_layout(self.citation_graph, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            self.citation_graph,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.7
        )
        
        nx.draw_networkx_edges(
            self.citation_graph,
            pos,
            alpha=0.3,
            arrows=True,
            arrowsize=10,
            width=0.5
        )
        
        # Draw labels for larger nodes (more cited papers)
        significant_nodes = [node for node, size in zip(self.citation_graph.nodes(), node_sizes) if size > node_size_factor * 2]
        labels = {node: self.papers.get(node, {}).get('title', node)[:20] + '...' for node in significant_nodes}
        nx.draw_networkx_labels(self.citation_graph, pos, labels=labels, font_size=8)
        
        # Add legend if colored by year
        if highlight_nodes is None and any(node_years):
            # Create a legend for the year colormap
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_year, vmax=max_year))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Publication Year')
        
        plt.title('Citation Network')
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    def find_research_communities(self, resolution=1.0):
        """
        Identify research communities/clusters in the citation network.
        
        Args:
            resolution (float): Resolution parameter for community detection
            
        Returns:
            dict: Community assignments for each paper
        """
        if not self.citation_graph.nodes():
            return {}
            
        # Convert directed graph to undirected for community detection
        undirected_graph = self.citation_graph.to_undirected()
        
        # Find communities using Louvain algorithm
        try:
            from community import community_louvain
            communities = community_louvain.best_partition(undirected_graph, resolution=resolution)
        except ImportError:
            # Fall back to connected components if community detection library not available
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected_graph)):
                for node in component:
                    communities[node] = i
        
        return communities
    
    def find_citation_path(self, source_id, target_id, max_length=5):
        """
        Find citation paths between two papers.
        
        Args:
            source_id (str): Source paper ID
            target_id (str): Target paper ID
            max_length (int): Maximum path length to consider
            
        Returns:
            list: List of citation paths
        """
        if not self.citation_graph.has_node(source_id) or not self.citation_graph.has_node(target_id):
            return []
            
        # Find all simple paths from source to target
        try:
            paths = list(nx.all_simple_paths(self.citation_graph, source_id, target_id, cutoff=max_length))
        except nx.NetworkXNoPath:
            paths = []
        
        # Format paths with paper details
        formatted_paths = []
        for path in paths:
            formatted_path = []
            for paper_id in path:
                if paper_id in self.papers:
                    formatted_path.append({
                        'id': paper_id,
                        'title': self.papers[paper_id]['title'],
                        'authors': self.papers[paper_id]['authors'],
                        'year': self.papers[paper_id]['year']
                    })
                else:
                    formatted_path.append({'id': paper_id})
            
            formatted_paths.append(formatted_path)
        
        return formatted_paths