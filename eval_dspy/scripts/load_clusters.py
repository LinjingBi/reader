#!/usr/bin/env python3
"""
Load raw JSON cluster files and transform them into a unified clusters.json format.

Reads all JSON files from eval_dspy/data/raw/ and creates eval_dspy/data/ready/clusters.json
with the structure:
[
  {
    "cluster_id": "source|period_start|period_end|cluster_index",
    "papers": [
      {"paper_id": "...", "title": "...", "summary": "...", "keywords": [...], "rank_in_cluster": 0}
    ]
  }
]
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_raw_files(raw_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON files from the raw directory."""
    raw_files = sorted(raw_dir.glob("best_clustering_json_reports_*.json"))
    if not raw_files:
        raise ValueError(f"No JSON files found in {raw_dir}")
    
    loaded_data = []
    for file_path in raw_files:
        print(f"Loading {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            loaded_data.append(data)
    
    return loaded_data


def transform_clusters(raw_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform raw cluster data into the unified format."""
    output_clusters = []
    
    for raw_data in raw_data_list:
        # Extract metadata
        source = raw_data.get("source", "")
        period_start = raw_data.get("period_start", "")
        period_end = raw_data.get("period_end", "")
        
        # Create paper lookup dictionary for efficient matching
        papers = raw_data.get("papers", [])
        paper_lookup: Dict[str, Dict[str, Any]] = {
            paper["paper_id"]: paper
            for paper in papers
            if "paper_id" in paper
        }
        
        # Process clusters
        clusters = raw_data.get("clusters", [])
        for cluster in clusters:
            cluster_index = cluster.get("cluster_index")
            members = cluster.get("members", [])
            
            # Construct cluster_id
            cluster_id = f"{source}|{period_start}|{period_end}|{cluster_index}"
            
            # Match papers from members
            cluster_papers = []
            for member in members:
                paper_id = member.get("paper_id")
                rank_in_cluster = member.get("rank_in_cluster")
                if paper_id and paper_id in paper_lookup:
                    paper = paper_lookup[paper_id]
                    # Extract only required fields
                    cluster_papers.append({
                        "paper_id": paper["paper_id"],
                        "title": paper.get("title", ""),
                        "summary": paper.get("summary", ""),
                        "keywords": paper.get("keywords", []),
                        "rank_in_cluster": rank_in_cluster
                    })
            
            # Only add cluster if it has papers
            if cluster_papers:
                output_clusters.append({
                    "cluster_id": cluster_id,
                    "papers": cluster_papers
                })
    
    return output_clusters


def main():
    """Main function to load and transform cluster data."""
    # Set up paths
    script_dir = Path(__file__).parent
    eval_dspy_dir = script_dir.parent
    raw_dir = eval_dspy_dir / "data" / "raw"
    ready_dir = eval_dspy_dir / "data" / "ready"
    output_file = ready_dir / "clusters.json"
    
    # Ensure ready directory exists
    ready_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw files
    print(f"Loading raw files from {raw_dir}...")
    raw_data_list = load_raw_files(raw_dir)
    
    # Transform clusters
    print("Transforming clusters...")
    output_clusters = transform_clusters(raw_data_list)
    
    # Write output
    print(f"Writing {len(output_clusters)} clusters to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_clusters, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Created {output_file} with {len(output_clusters)} clusters.")


if __name__ == "__main__":
    main()

