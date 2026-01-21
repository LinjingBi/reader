"""Memo CLI adapter"""

import json
import subprocess
from typing import Dict, Optional

from reader.config import ReaderConfig


def fresh_paper(payload: dict, config: ReaderConfig) -> dict:
    """
    Call memo CLI fresh-paper command to ingest papers and clustering.
    
    Args:
        payload: Fresh paper payload dictionary matching fresh_paper_payload.json format
        config: ReaderConfig instance
        
    Returns:
        Dictionary with 'snapshot_id' and 'cluster_run_id' keys, or empty dict if disabled/error
    """
    if not config.memo.enabled:
        return {}
    
    try:
        # Convert payload to JSON string
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        
        # Build command (use '-' to read from stdin)
        cmd = [
            config.memo.bin,
            'fresh-paper',
            '--input', '-',
            '--db', config.memo.db_path,
        ]
        
        # Run memo CLI with stdin input
        result = subprocess.run(
            cmd,
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=config.memo.timeout_sec,
            check=True,
        )
        
        # Parse JSON output
        output = json.loads(result.stdout)
        return output
            
    except subprocess.TimeoutExpired:
        print(f"Warning: memo fresh-paper timed out after {config.memo.timeout_sec}s")
        return {}
    except subprocess.CalledProcessError as e:
        print(f"Error calling memo fresh-paper: {e.stderr}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing memo output: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error in memo fresh-paper: {e}")
        return {}


def get_best_clustering(
    source: str,
    period_start: str,
    period_end: str,
    config: ReaderConfig,
    top_n: int = 10,
) -> dict:
    """
    Call memo CLI get-best-run command to retrieve best clustering.
    
    Args:
        source: Snapshot source (e.g., 'hf_monthly')
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)
        config: ReaderConfig instance
        top_n: Maximum papers per cluster to include (default: 10)
        
    Returns:
        Dictionary with clustering data, or empty dict if disabled/error
    """
    if not config.memo.enabled:
        return {}
    
    try:
        # Build command
        cmd = [
            config.memo.bin,
            'get-best-run',
            '--source', source,
            '--period-start', period_start,
            '--period-end', period_end,
            '--top-n', str(top_n),
            '--db', config.memo.db_path,
        ]
        
        # Run memo CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.memo.timeout_sec,
            check=True,
        )
        
        # Parse JSON output
        output = json.loads(result.stdout)
        return output
        
    except subprocess.TimeoutExpired:
        print(f"Warning: memo get-best-run timed out after {config.memo.timeout_sec}s")
        return {}
    except subprocess.CalledProcessError as e:
        print(f"Error calling memo get-best-run: {e.stderr}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing memo output: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error in memo get-best-run: {e}")
        return {}


def fresh_topic(payload: dict, config: ReaderConfig) -> dict:
    """
    Stub for future topic creation.
    
    Args:
        payload: Topic payload dictionary
        config: ReaderConfig instance
        
    Returns:
        Empty dict (not implemented)
    """
    if not config.memo.enabled:
        return {}
    # TODO: Implement when topic functionality is added
    return {}


def get_topic_metadata(topic_id: str, config: ReaderConfig) -> dict:
    """
    Stub for future topic metadata retrieval.
    
    Args:
        topic_id: Topic identifier
        config: ReaderConfig instance
        
    Returns:
        Empty dict (not implemented)
    """
    if not config.memo.enabled:
        return {}
    # TODO: Implement when topic functionality is added
    return {}


def fresh_report(payload: dict, config: ReaderConfig) -> dict:
    """
    Stub for future report creation.
    
    Args:
        payload: Report payload dictionary
        config: ReaderConfig instance
        
    Returns:
        Empty dict (not implemented)
    """
    if not config.memo.enabled:
        return {}
    # TODO: Implement when report functionality is added
    return {}
