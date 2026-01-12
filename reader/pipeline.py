import json
from fetch import Paper
from embed import Embedder, build_embedding_text
from cluster import KmeanClusterer


def run_pipeline():
    # papers = fetch_papers()
    with open("../papers_report.json", "r") as f:
        data = json.load(f)
    papers_data = data['papers']
    embedder = Embedder()

    for month, papers_list in papers_data.items():
        print("looking at month: ", month)

        # Create Paper objects from JSON data
        papers = []
        for paper_data in papers_list:
            paper = paper_data['paper']
            paper_obj = Paper(
                pid=paper['id'],
                title=paper['title'],
                summary=paper['summary'],
                keywords=paper.get('ai_keywords', []),
                url=paper.get('projectPage', '')
            )
            papers.append(paper_obj)

        # Build embedding text and encode
        embedding_texts = [build_embedding_text(p) for p in papers]
        embeddings = embedder.encode(embedding_texts)
        print("embeddings shape: ", embeddings.shape)

        # Create clusterer with papers and embeddings
        clusterer = KmeanClusterer(papers, embeddings, random_state=42, n_clusters=5)
        clusterer.fit_predict()

        # Get grouped papers by cluster
        grouped = clusterer.group_by_cluster()
        for cluster_id, paper_titles in grouped.items():
            print(f"Cluster {cluster_id}: {len(paper_titles)} papers")
            for title in paper_titles[:3]:  # Show first 3 titles
                print(f"  - {title}")
            if len(paper_titles) > 3:
                print(f"  ... and {len(paper_titles) - 3} more")

        # Plot clusters
        clusterer.plot_umap()
        break
        # cluster = Cluster(embeddings)

if __name__ == "__main__":
    run_pipeline()