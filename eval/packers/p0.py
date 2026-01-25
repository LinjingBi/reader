"""P0 packer: transforms monthly clustering data to LLM input format"""


def build_input(cluster_data: dict, config: dict = None) -> dict:
    """
    Transform monthly clustering payload to input_schema.json format.
    
    Args:
        cluster_data: Input payload matching fresh_paper_payload.json format:
            - clusters: Array with cluster_index, size, cohesion, members
            - papers: Array with paper_id, title, summary, keywords, url, published_at
            - members: Array with paper_id, rank_in_cluster, sim_to_centroid
        config: Optional config dict (for future extensions)
    
    Returns:
        Dict matching input_schema.json format:
            - clusters: Array of cluster objects, each with papers array
            - Each paper has: paper_id, title, summary, keywords, url, sim_to_centroid, rank_in_cluster
    """
    if config is None:
        config = {}
    
    # Build paper lookup dictionary
    papers_dict = {paper["paper_id"]: paper for paper in cluster_data.get("papers", [])}
    
    # Process each cluster
    output_clusters = []
    warnings = []
    
    for cluster in cluster_data.get("clusters", []):
        cluster_papers = []
        
        # Get members sorted by rank_in_cluster (ascending, 0 = most representative)
        members = sorted(
            cluster.get("members", []),
            key=lambda m: m.get("rank_in_cluster", 999)
        )
        
        for member in members:
            paper_id = member.get("paper_id")
            if not paper_id:
                warnings.append(f"Missing paper_id in cluster {cluster.get('cluster_index')}")
                continue
            
            # Look up paper
            paper = papers_dict.get(paper_id)
            if not paper:
                warnings.append(f"Paper {paper_id} not found in papers array")
                continue
            
            # Build paper object matching input_schema.json format
            paper_obj = {
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "summary": paper.get("summary", ""),
                "keywords": paper.get("keywords", []),
                "url": paper.get("url", ""),
                "sim_to_centroid": member.get("sim_to_centroid", 0.0),
                "rank_in_cluster": member.get("rank_in_cluster", 0),
            }
            
            cluster_papers.append(paper_obj)
        
        # Add cluster with papers array and preserve cluster_index for mapping
        output_clusters.append({
            "papers": cluster_papers,
            "cluster_index": cluster.get("cluster_index")  # Preserve for later use
        })
    
    result = {
        "clusters": output_clusters
    }
    return result, warnings


def get_packer_stats(packed_data: dict) -> dict:
    """
    Get statistics about packed data for logging.
    
    Args:
        packed_data: Output from build_input()
    
    Returns:
        Dict with counts: clusters, papers
    """
    clusters = packed_data.get("clusters", [])
    total_papers = sum(len(c.get("papers", [])) for c in clusters)
    
    return {
        "clusters": len(clusters),
        "papers": total_papers
    }

