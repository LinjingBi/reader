"""Download PDFs from arXiv for benchmarking."""
import asyncio
import json
import random
from pathlib import Path
from typing import List

import aiohttp


def load_all_paper_ids(papers_report_path: Path) -> List[str]:
    """Extract all paper IDs from papers_report.json."""
    with open(papers_report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    paper_ids = []
    for month_key, papers in data.get("papers", {}).items():
        for paper_entry in papers:
            if "paper" in paper_entry and "id" in paper_entry["paper"]:
                paper_ids.append(paper_entry["paper"]["id"])
    
    return paper_ids


def save_paper_ids(paper_ids: List[str], output_path: Path):
    """Save paper IDs to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"paper_ids": paper_ids}, f, indent=2)
    print(f"Saved {len(paper_ids)} paper IDs to {output_path}")


def load_paper_ids(paper_ids_path: Path) -> List[str]:
    """Load paper IDs from JSON file."""
    with open(paper_ids_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("paper_ids", [])


async def download_pdf(session: aiohttp.ClientSession, paper_id: str, output_dir: Path, semaphore: asyncio.Semaphore):
    """Download a single PDF from arXiv."""
    async with semaphore:
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        output_path = output_dir / f"arxiv_{paper_id}.pdf"
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"✓ Already exists: {output_path.name}")
            return True
        
        try:
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    content = await response.read()
                    output_path.write_bytes(content)
                    print(f"✓ Downloaded: {output_path.name}")
                    return True
                else:
                    print(f"✗ Failed to download {paper_id}: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"✗ Error downloading {paper_id}: {e}")
            return False


async def download_all_pdfs(paper_ids: List[str], output_dir: Path, max_concurrent: int = 10):
    """Download all PDFs concurrently."""
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_pdf(session, paper_id, output_dir, semaphore)
            for paper_id in paper_ids
        ]
        results = await asyncio.gather(*tasks)
    
    successful = sum(results)
    print(f"\nDownloaded {successful}/{len(paper_ids)} PDFs successfully")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent
    papers_report_path = repo_root / "reader" / "src" / "papers_report.json"
    data_dir = script_dir / "data"
    paper_ids_path = data_dir / "paper_ids.json"
    
    # Step 1: Load all paper IDs and randomly select 50
    if not paper_ids_path.exists():
        print("Loading all paper IDs from papers_report.json...")
        all_ids = load_all_paper_ids(papers_report_path)
        print(f"Found {len(all_ids)} total paper IDs")
        
        selected_ids = random.sample(all_ids, min(50, len(all_ids)))
        print(f"Randomly selected {len(selected_ids)} paper IDs")
        
        save_paper_ids(selected_ids, paper_ids_path)
    else:
        print(f"Using existing paper IDs from {paper_ids_path}")
    
    # Step 2: Load IDs and download PDFs
    paper_ids = load_paper_ids(paper_ids_path)
    print(f"\nDownloading {len(paper_ids)} PDFs...")
    
    asyncio.run(download_all_pdfs(paper_ids, data_dir, max_concurrent=10))
    
    print(f"\nAll PDFs saved to: {data_dir}")


if __name__ == "__main__":
    main()

