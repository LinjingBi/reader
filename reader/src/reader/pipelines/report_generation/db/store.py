"""Database store for report job management."""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from reader.pipelines.report_generation.config.config import ReportGenerationConfig


class ReportJobStatus:
    """Report job status constants."""
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class InitReportJobResponse:
    """Response from init_report_job function."""
    next_status: str  # 'running' | 'resuming' | 'waiting' | 'done'
    meta: 'InitReportJobResponseMeta'


@dataclass
class InitReportJobResponseMeta:
    """Metadata for init_report_job response."""
    report_url: Optional[str] = None
    last_update_utc: Optional[str] = None
    message: str = ""


class ReportJobStore:
    """Database store for report job operations."""

    def __init__(self, db_path: Path, migrations_path: Path):
        """
        Initialize the store.

        Args:
            db_path: Path to SQLite database file
            migrations_path: Path to migrations directory
        """
        self.db_path = db_path
        self.migrations_path = migrations_path
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            # Apply PRAGMAs
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            # Apply migrations
            self._apply_migrations()
        return self._conn

    def _apply_migrations(self):
        """Apply migration files in order."""
        if not self.migrations_path.exists():
            return

        migration_files = sorted(self.migrations_path.glob("*.sql"))
        conn = self._get_connection()
        cursor = conn.cursor()

        for migration_file in migration_files:
            sql = migration_file.read_text(encoding="utf-8")
            cursor.executescript(sql)

        conn.commit()

    def get_report_job(self, cluster_pk_hash: str) -> Optional[Tuple[str, str, str]]:
        """
        Get report job by cluster_pk_hash.

        Returns:
            Tuple of (status, created_at, updated_at) or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, created_at, updated_at FROM report_job WHERE cluster_pk_hash = ?",
            (cluster_pk_hash,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return (row[0], row[1], row[2])

    def create_report_job(self, cluster_pk_hash: str, status: str, now: str):
        """Create a new report job."""
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO report_job(cluster_pk_hash, status, created_at, updated_at) VALUES(?, ?, ?, ?)",
            (cluster_pk_hash, status, now, now)
        )
        conn.commit()

    def update_report_job_status(self, cluster_pk_hash: str, status: str, now: str):
        """Update report job status."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE report_job SET status = ?, updated_at = ? WHERE cluster_pk_hash = ?",
            (status, now, cluster_pk_hash)
        )
        conn.commit()

    def delete_report_job(self, cluster_pk_hash: str):
        """Delete a report job."""
        conn = self._get_connection()
        conn.execute("DELETE FROM report_job WHERE cluster_pk_hash = ?", (cluster_pk_hash,))
        conn.commit()

    def get_avg_runtime_of_latest_done_jobs(self) -> Optional[timedelta]:
        """
        Get average runtime of latest 3 done jobs.

        Returns:
            Average timedelta or None if fewer than 1 done job
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT created_at, updated_at FROM report_job
               WHERE status = 'done'
               ORDER BY updated_at DESC
               LIMIT 3"""
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        runtimes = []
        for created_str, updated_str in rows:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
            runtimes.append(updated - created)

        if not runtimes:
            return None

        total = sum(runtimes, timedelta())
        return total / len(runtimes)

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def format_remaining_time(remaining: timedelta) -> str:
    """Format timedelta as human-readable string."""
    total_seconds = int(remaining.total_seconds())
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    if minutes > 0 and seconds > 0:
        return f"{minutes} minutes {seconds} seconds"
    elif minutes > 0:
        return f"{minutes} minutes"
    else:
        return f"{seconds} seconds"


def init_report_job(cluster_pk_hash: str, cfg: ReportGenerationConfig) -> InitReportJobResponse:
    """
    Initialize or check report job status.

    Args:
        cluster_pk_hash: Cluster pk_hash
        cfg: ReportGenerationConfig instance

    Returns:
        InitReportJobResponse with next_status and metadata
    """
    # Load paths from config
    db_path = cfg.cache.report_generation_db_path
    migrations_path = cfg.cache.report_generation_db_migrations_path

    store = ReportJobStore(db_path, migrations_path)
    now = datetime.now(timezone.utc)
    now_str = now.isoformat()

    try:
        job = store.get_report_job(cluster_pk_hash)

        if job is None:
            # No job exists, create new one
            store.create_report_job(cluster_pk_hash, ReportJobStatus.RUNNING, now_str)
            return InitReportJobResponse(
                next_status="running",
                meta=InitReportJobResponseMeta(
                    report_url=None,
                    last_update_utc=None,
                    message="A new job is running."
                )
            )

        status, created_at_str, updated_at_str = job

        if status == ReportJobStatus.RUNNING:
            # Job is running, calculate wait time
            avg_runtime = store.get_avg_runtime_of_latest_done_jobs()
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            created_at = created_at.replace(tzinfo=timezone.utc)

            if avg_runtime is not None:
                expected_done = created_at + avg_runtime
                if expected_done < now:
                    wait_duration = timedelta(minutes=5)
                else:
                    wait_duration = expected_done - now
            else:
                wait_duration = timedelta(minutes=5)

            wait_str = format_remaining_time(wait_duration)
            return InitReportJobResponse(
                next_status="waiting",
                meta=InitReportJobResponseMeta(
                    report_url=None,
                    last_update_utc=updated_at_str,
                    message=f"An existing job is already running. Estimated time remaining: {wait_str}."
                )
            )

        elif status == ReportJobStatus.DONE:
            # Job is done
            return InitReportJobResponse(
                next_status="done",
                meta=InitReportJobResponseMeta(
                    report_url=None,  # No report_url, search via cluster_pk_hash
                    last_update_utc=updated_at_str,
                    message="Report already generated by a previous run; no new run created. Returning existing report details."
                )
            )

        elif status == ReportJobStatus.ERROR:
            # Check if error expired (5 minutes)
            updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            updated_at = updated_at.replace(tzinfo=timezone.utc)
            elapsed = now - updated_at
            five_minutes = timedelta(minutes=5)

            if elapsed < five_minutes:
                remaining = five_minutes - elapsed
                remaining_str = format_remaining_time(remaining)
                return InitReportJobResponse(
                    next_status="waiting",
                    meta=InitReportJobResponseMeta(
                        report_url=None,
                        last_update_utc=updated_at_str,
                        message=f"An error happened before. Please wait {remaining_str} to trigger a new job."
                    )
                )
            else:
                # Error expired, delete job so it can be resumed
                store.delete_report_job(cluster_pk_hash)
                return InitReportJobResponse(
                    next_status="resuming",
                    meta=InitReportJobResponseMeta(
                        report_url=None,
                        last_update_utc=None,
                        message="Previous error expired; job is ready for resume now."
                    )
                )

        else:
            raise ValueError(f"Unexpected job status: {status}")

    finally:
        store.close()
