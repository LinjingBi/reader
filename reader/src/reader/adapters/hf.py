"""HuggingFace API adapter"""

import httpx
import asyncio
import json
from typing import List, Dict
from datetime import datetime

from reader.models import Paper
from reader.config import ReaderConfig


async def fetch_papers(client: httpx.AsyncClient, url: str, params: str) -> List[Dict]:
    """
    Fetch papers for a given month asynchronously.
    
    Args:
        client: httpx AsyncClient instance
        url: Base API URL
        params: Query parameters (e.g., "month=2025-01")
        
    Returns:
        List of paper dictionaries
    """
    url_with_params = f'{url}?{params}'
    try:
        r = await client.get(url_with_params, headers={"User-Agent":"daily-ai-briefing/0.1"}, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        # Handle both dict response {"paper": [...]} and list response [...]
        if isinstance(data, dict):
            papers = data.get('paper', [])
        elif isinstance(data, list):
            papers = data
        else:
            papers = []
        return papers
    except Exception as e:
        print(f"Error fetching papers for {params}: {e}. hf err: {r.text if 'r' in locals() else 'N/A'}")
        return []


async def get_monthly_report(config: ReaderConfig) -> Dict:
    """
    Fetch papers for all 12 months concurrently.
    
    Args:
        config: ReaderConfig instance
        
    Returns:
        Dictionary with 'papers' and 'metadata' keys
    """
    results = {
        'papers': {},
        'metadata': {
            'total num': {},
        },
    }
    
    async with httpx.AsyncClient() as client:
        # Create tasks for all 12 months
        tasks = []
        months = []
        # Extract year from month_key (e.g., "month=2025-01" -> 2025)
        year = int(config.run.month_key.split('=')[1].split('-')[0])
        for m in range(1, 13):
            month = f'month={year}-{m:02d}'
            months.append(month)
            task = fetch_papers(client, config.sources.hf.daily_papers_api, month)
            tasks.append(task)
        
        # Fetch all months concurrently
        papers_list = await asyncio.gather(*tasks)
        
        # Map results to months
        for month, papers in zip(months, papers_list):
            results['papers'][month] = papers
            results['metadata']['total num'][month] = len(papers)
    
    return results


def parse_papers(papers: List[Dict], config: ReaderConfig) -> List[Paper]:
    """
    Parse paper dictionaries into Paper objects.
    
    Args:
        papers: List of paper dictionaries from API
        config: ReaderConfig instance
        
    Returns:
        List of Paper objects
    """
    result = []
    for paper in papers:
        # Extract published_at and convert to YYYY-MM-DD format
        published_at = ""
        pub_date_str = paper['paper'].get('publishedAt', "")
        if pub_date_str:
            try:
                # Parse ISO format: "2025-01-22T15:19:35.000Z"
                dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                published_at = dt.strftime('%Y-%m-%d')
            except (ValueError, AttributeError):
                pass
        
        result.append(Paper(
            pid=paper['paper']['id'],
            title=paper['paper']['title'],
            summary=paper['paper']['summary'],
            keywords=paper['paper'].get('ai_keywords', []),
            url=f"{config.sources.hf.paper_page_base_url}{paper['paper']['id']}",
            published_at=published_at
        ))
    return result


def save_papers_to_file(results: Dict, config: ReaderConfig, output_txt: str = 'papers_report.txt') -> None:
    """
    Save papers to file in the specified format.
    
    Args:
        results: Results dictionary with 'papers' key
        config: ReaderConfig instance
        output_txt: Optional path for text output file
    """
    output_json = config.sources.hf.output_json
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_txt, 'w', encoding='utf-8') as f:
        for month in sorted(results['papers'].keys()):
            papers = results['papers'][month]
            f.write(f"{month} total {len(papers)} papers:\n")
            for paper_data in papers:
                # Handle both {"paper": {...}} and direct paper object
                if isinstance(paper_data, dict):
                    paper = paper_data.get('paper', paper_data)
                else:
                    paper = {}
                
                paper_id = paper.get('id', 'N/A')
                title = paper.get('title', 'N/A')
                summary = paper.get('summary', 'N/A')
                ai_summary = paper.get('ai_summary', 'N/A')

                
                f.write(f"   title: {title}\n")
                f.write(f"   summary: {summary}\n")
                f.write(f"   id: {paper_id}\n")
                f.write(f"   ai_summary: {ai_summary}\n")
                f.write("\n")
