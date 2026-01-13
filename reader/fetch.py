import httpx
import asyncio
import json
from dataclasses import dataclass
from typing import List, Dict

"""
| Parameter | Format           | Meaning                                |                       |
| --------- | ---------------- | -------------------------------------- | --------------------- |
| **date**  | `YYYY-MM-DD`     | Fetch papers for a *specific day*      |                       |
| **month** | `YYYY-MM`        | Fetch papers for the *entire month*    |                       |
| **week**  | e.g., `2025-W22` | Fetch papers for the *entire ISO week* | ([Ismena website][1]) |

[1]: https://www.ismena.com/custom-connectors/hugging-face-connector/?utm_source=chatgpt.com "Hugging Face Connector - Ismena website"

"""
fetch_url = 'https://huggingface.co/api/daily_papers'
hf_paper_url = 'https://huggingface.co/papers/'

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
        print(f"Error fetching papers for {params}: {e}. hf err: {r.text}")
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
            month = f'month=2025-{m:02d}'
            months.append(month)
            task = fetch_papers(client, fetch_url, month)
            tasks.append(task)
        
        # Fetch all months concurrently
        papers_list = await asyncio.gather(*tasks)
        
        # Map results to months
        for month, papers in zip(months, papers_list):
            results['papers'][month] = papers
            results['metadata']['total num'][month] = len(papers)
    
    return results

def serialize_to_paper_objects(papers: List[Dict]) -> List[Paper]:
    return [Paper(
        pid=paper['paper']['id'],
        title=paper['paper']['title'],
        summary=paper['paper']['summary'],
        keywords=paper['paper'].get('ai_keywords', []),
        url=f"{hf_paper_url}{paper['paper']['id']}"
    ) for paper in papers]

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


def _brief():
    with open('papers_report.json') as f:
        d = json.load(f)
    for month, ps in d['papers'].items():
        ak = []
        for p in ps:
            ak.append(len(p['paper'].get('ai_keywords', [])))
            kw_str = ",".join(p['paper'].get('ai_keywords', []))
            kw_len = len(kw_str)
            ts_len =len(p['paper']['title']+p['paper']['summary'])
            p_kw_ratio = kw_len/ts_len
            print(f"keywords/summary ratio: {p_kw_ratio} k: {kw_len} s: {ts_len}")
        print(f"{month} average keywords: {sum(ak)/len(ak)}")

