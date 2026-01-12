import httpx
import asyncio
from bs4 import BeautifulSoup
import trafilatura
import re, json, time
from urllib.parse import urlparse

from dataclasses import dataclass

from typing import List, Dict, Tuple, Iterable, Optional

url = 'https://huggingface.co/api/daily_papers'

@dataclass
class Paper:
    pid: str
    title: str
    summary: str
    keywords: List[str]
    url: str = ""

async def fetch_papers(client, url, params):
    """Fetch papers for a given month asynchronously"""
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
        print(f"Error fetching papers for {params}: {e}")
        return []


async def get_monthly_report():
    """Fetch papers for all 12 months concurrently"""
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
        for m in range(1, 13):
            month = f'2025/{m:02d}'
            months.append(month)
            task = fetch_papers(client, url, month)
            tasks.append(task)
        
        # Fetch all months concurrently
        papers_list = await asyncio.gather(*tasks)
        
        # Map results to months
        for month, papers in zip(months, papers_list):
            results['papers'][month] = papers
            results['metadata']['total num'][month] = len(papers)
    
    return results


def save_papers_to_file(results, output_json='papers_report.json', output_txt='papers_report.txt'):
    """Save papers to file in the specified format"""
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


async def main():
    """Main async function"""
    print("Fetching papers for past 12 months...")
    results = await get_monthly_report()
    print(f"Fetched papers for {len(results)} months")
    
    save_papers_to_file(results)
    print("Papers saved to papers_report.(txt,json)")


if __name__ == "__main__":
    asyncio.run(main())
