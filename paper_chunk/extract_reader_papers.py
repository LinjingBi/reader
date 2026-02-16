#!/usr/bin/env python3
"""
Extract papers from cluster JSON files and generate reader_papers.json.

This script:
1. Loads all clusters_data/*.json files
2. Extracts papers from cluster members
3. Removes "hf:" prefix from paper_ids
4. Deduplicates papers across all files
5. Generates reader_papers.json with the same structure as papers.json
6. Prints statistics comparing deduplicated count vs result file count
"""

import json
import glob
from pathlib import Path
from collections import defaultdict


def load_cluster_files(cluster_dir):
    """Load all cluster JSON files from the specified directory."""
    pattern = str(Path(cluster_dir) / "*.json")
    files = glob.glob(pattern)
    return sorted(files)


def extract_papers_from_clusters(cluster_files):
    """
    Extract papers from all cluster files.
    
    Returns:
        tuple: (dict of papers, int count before deduplication)
        dict: Mapping of paper_id (without "hf:" prefix) to paper info with url
    """
    all_papers = {}
    total_before_dedup = 0
    
    for cluster_file in cluster_files:
        print(f"Processing {Path(cluster_file).name}...")
        
        with open(cluster_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a mapping of paper_id to paper info for quick lookup
        papers_map = {}
        for paper in data.get('papers', []):
            paper_id = paper.get('paper_id', '')
            if paper_id:
                papers_map[paper_id] = paper
        
        # Extract papers from clusters
        clusters = data.get('clusters', [])
        for cluster in clusters:
            members = cluster.get('members', [])
            for member in members:
                paper_id_with_prefix = member.get('paper_id', '')
                if paper_id_with_prefix and paper_id_with_prefix.startswith('hf:'):
                    # Count before deduplication
                    total_before_dedup += 1
                    
                    # Remove "hf:" prefix
                    paper_id = paper_id_with_prefix[3:]
                    
                    # Get full paper info from papers array
                    if paper_id_with_prefix in papers_map:
                        paper_info = papers_map[paper_id_with_prefix]
                        url = paper_info.get('url', '')
                        
                        # Store paper (will deduplicate by paper_id)
                        if paper_id not in all_papers:
                            all_papers[paper_id] = {
                                'id': paper_id,
                                'url': url
                            }
    
    return all_papers, total_before_dedup


def write_reader_papers(output_file, papers_dict):
    """Write papers to reader_papers.json in the same format as papers.json."""
    # Convert dict to list sorted by id for consistency
    papers_list = sorted(papers_dict.values(), key=lambda x: x['id'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers_list, f, indent=2, ensure_ascii=False)
    
    return len(papers_list)


def main():
    """Main execution function."""
    # Paths
    script_dir = Path(__file__).parent
    cluster_dir = script_dir / "clusters_data"
    output_file = script_dir / "papers.json"
    
    # Load all cluster files
    cluster_files = load_cluster_files(cluster_dir)
    
    if not cluster_files:
        print(f"No cluster JSON files found in {cluster_dir}")
        return
    
    print(f"Found {len(cluster_files)} cluster file(s)\n")
    
    # Extract papers from clusters
    papers_dict, total_before_dedup = extract_papers_from_clusters(cluster_files)
    
    # Get deduplicated count
    total_deduplicated = len(papers_dict)
    
    # Write output file
    print(f"\nWriting output to {output_file.name}...")
    result_count = write_reader_papers(output_file, papers_dict)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"  Total papers (before deduplication): {total_before_dedup}")
    print(f"  Total papers (after deduplication): {total_deduplicated}")
    print(f"  Papers in result JSON: {result_count}")
    print(f"{'='*60}")
    
    if total_deduplicated == result_count:
        print("✓ Deduplicated count matches result file count!")
    else:
        print("⚠ Warning: Deduplicated count doesn't match result file count!")
    
    duplicates_removed = total_before_dedup - total_deduplicated
    if duplicates_removed > 0:
        print(f"  Duplicates removed: {duplicates_removed}")


if __name__ == "__main__":
    main()

