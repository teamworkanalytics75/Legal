"""Data loading utilities for ML system.

This module provides data loaders for:
1. Legal documents from lawsuit_docs MySQL database
2. Agent execution history from jobs.db SQLite database
3. Feature extraction and preprocessing
"""

import sqlite3
import types
try:
    import mysql.connector as _mysql_connector  # type: ignore
    import mysql as _mysql_pkg  # type: ignore
    MYSQL_AVAILABLE = True
except Exception:
    # Create a minimal shim so tests can patch ml_system.data.data_loader.mysql.connector.connect
    _mysql_connector = None
    _mysql_pkg = types.SimpleNamespace(connector=types.SimpleNamespace(connect=None))
    MYSQL_AVAILABLE = False
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Expose a module-level `mysql` name for compatibility with tests and patching
mysql = _mysql_pkg


class LegalDataLoader:
    """Loads legal case data from lawsuit_docs MySQL database."""

    def __init__(self, mysql_config: Optional[Dict] = None):
        """Initialize with MySQL connection config.

        Args:
            mysql_config: MySQL connection parameters
        """
        self.mysql_config = mysql_config or self._get_default_mysql_config()

    def _get_default_mysql_config(self) -> Dict:
        """Get default MySQL configuration."""
        return {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'lawsuit_docs',
            'charset': 'utf8mb4'
        }

    def load_case_law_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load case law data from lawsuit_docs database.

        Args:
            limit: Maximum number of cases to load

        Returns:
            DataFrame with case data and labels
        """
        try:
            if not MYSQL_AVAILABLE:
                logger.warning("MySQL connector not available; returning empty DataFrame")
                return pd.DataFrame()

            connection = _mysql_connector.connect(**self.mysql_config)  # type: ignore
            cursor = connection.cursor()

            # Query to get case data
            query = """
            SELECT
                id,
                case_name,
                citation,
                court,
                date_filed,
                opinion_text,
                case_type,
                plaintiff,
                defendant,
                legal_issues,
                topic,
                date_ingested
            FROM case_law
            WHERE opinion_text IS NOT NULL
            AND LENGTH(opinion_text) > 100
            ORDER BY date_ingested DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()

            df = pd.DataFrame(data, columns=columns)

            # Create labels from case metadata
            df = self._create_case_labels(df)

            cursor.close()
            connection.close()

            logger.info(f"Loaded {len(df)} cases from lawsuit_docs database")
            return df

        except Exception as e:
            logger.error(f"Error loading case law data: {e}")
            return pd.DataFrame()

    def _create_case_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels from case metadata.

        Args:
            df: Raw case data

        Returns:
            DataFrame with added labels
        """
        # Create outcome labels (simplified heuristic)
        df['outcome_label'] = self._infer_outcome_labels(df)

        # Create complexity labels
        df['complexity_label'] = self._infer_complexity_labels(df)

        # Create jurisdiction labels
        df['jurisdiction_label'] = self._infer_jurisdiction_labels(df)

        # Create legal domain labels
        df['domain_label'] = self._infer_domain_labels(df)

        return df

    def _infer_outcome_labels(self, df: pd.DataFrame) -> List[str]:
        """Infer case outcome labels from text patterns."""
        outcomes = []
        for _, row in df.iterrows():
            text = str(row.get('opinion_text', '')).lower()

            # Simple heuristic based on keywords
            if any(word in text for word in ['granted', 'sustained', 'affirmed', 'won', 'prevailed']):
                outcomes.append('win')
            elif any(word in text for word in ['denied', 'dismissed', 'reversed', 'lost', 'failed']):
                outcomes.append('loss')
            elif any(word in text for word in ['settled', 'settlement', 'agreed']):
                outcomes.append('settlement')
            else:
                outcomes.append('unknown')

        return outcomes

    def _infer_complexity_labels(self, df: pd.DataFrame) -> List[str]:
        """Infer complexity labels from case characteristics."""
        complexities = []
        for _, row in df.iterrows():
            text_length = len(str(row.get('opinion_text', '')))
            issues_count = len(str(row.get('legal_issues', '')).split(',')) if row.get('legal_issues') else 0

            if text_length > 50000 or issues_count > 5:
                complexities.append('high')
            elif text_length > 10000 or issues_count > 2:
                complexities.append('medium')
            else:
                complexities.append('low')

        return complexities

    def _infer_jurisdiction_labels(self, df: pd.DataFrame) -> List[str]:
        """Infer jurisdiction labels from court information."""
        jurisdictions = []
        for _, row in df.iterrows():
            court = str(row.get('court', '')).lower()

            if 'federal' in court or 'district' in court or 'circuit' in court:
                jurisdictions.append('federal')
            elif 'supreme' in court:
                jurisdictions.append('supreme')
            else:
                jurisdictions.append('state')

        return jurisdictions

    def _infer_domain_labels(self, df: pd.DataFrame) -> List[str]:
        """Infer legal domain labels from case type and issues."""
        domains = []
        for _, row in df.iterrows():
            case_type = str(row.get('case_type', '')).lower()
            issues = str(row.get('legal_issues', '')).lower()

            if 'employment' in case_type or 'discrimination' in issues:
                domains.append('employment')
            elif 'contract' in case_type or 'contract' in issues:
                domains.append('contract')
            elif 'tort' in case_type or 'negligence' in issues:
                domains.append('tort')
            elif 'criminal' in case_type:
                domains.append('criminal')
            else:
                domains.append('other')

        return domains


class AgentDataLoader:
    """Loads agent execution data from jobs.db SQLite database."""

    def __init__(self, db_path: str = "jobs.db"):
        """Initialize with SQLite database path.

        Args:
            db_path: Path to jobs.db SQLite database
        """
        self.db_path = db_path

    def load_agent_execution_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load agent execution data from jobs.db.

        Args:
            limit: Maximum number of records to load

        Returns:
            DataFrame with agent execution data and labels
        """
        try:
            connection = sqlite3.connect(self.db_path)

            # Query to get job and run data
            query = """
            SELECT
                j.id as job_id,
                j.phase,
                j.type as job_type,
                j.status as job_status,
                j.priority,
                j.budget_tokens,
                j.budget_seconds,
                j.created_at,
                j.started_at,
                j.ended_at,
                r.id as run_id,
                r.agent_name,
                r.status as run_status,
                r.tokens_in,
                r.tokens_out,
                r.duration_seconds,
                r.error_message,
                r.retry_count
            FROM jobs j
            LEFT JOIN runs r ON j.id = r.job_id
            WHERE j.status IN ('succeeded', 'failed', 'dead')
            ORDER BY j.created_at DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, connection)
            connection.close()

            # Create labels from execution data
            df = self._create_agent_labels(df)

            logger.info(f"Loaded {len(df)} agent execution records from jobs.db")
            return df

        except Exception as e:
            logger.error(f"Error loading agent execution data: {e}")
            return pd.DataFrame()

    def _create_agent_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels from agent execution data.

        Args:
            df: Raw agent execution data

        Returns:
            DataFrame with added labels
        """
        # Create success labels
        df['success_label'] = (df['job_status'] == 'succeeded').astype(int)

        # Create performance labels
        df['performance_label'] = self._infer_performance_labels(df)

        # Create efficiency labels
        df['efficiency_label'] = self._infer_efficiency_labels(df)

        # Create error type labels
        df['error_type_label'] = self._infer_error_type_labels(df)

        return df

    def _infer_performance_labels(self, df: pd.DataFrame) -> List[str]:
        """Infer performance labels from execution metrics."""
        performances = []
        for _, row in df.iterrows():
            duration = row.get('duration_seconds', 0) or 0
            tokens_out = row.get('tokens_out', 0) or 0

            if duration > 0:
                tokens_per_second = tokens_out / duration
                if tokens_per_second > 100:
                    performances.append('high')
                elif tokens_per_second > 50:
                    performances.append('medium')
                else:
                    performances.append('low')
            else:
                performances.append('unknown')

        return performances

    def _infer_efficiency_labels(self, df: pd.DataFrame) -> List[str]:
        """Infer efficiency labels from resource usage."""
        efficiencies = []
        for _, row in df.iterrows():
            budget_tokens = row.get('budget_tokens', 0) or 0
            tokens_out = row.get('tokens_out', 0) or 0

            if budget_tokens > 0:
                efficiency = tokens_out / budget_tokens
                if efficiency > 0.8:
                    efficiencies.append('high')
                elif efficiency > 0.5:
                    efficiencies.append('medium')
                else:
                    efficiencies.append('low')
            else:
                efficiencies.append('unknown')

        return efficiencies

    def _infer_error_type_labels(self, df: pd.DataFrame) -> List[str]:
        """Infer error type labels from error messages."""
        error_types = []
        for _, row in df.iterrows():
            error_msg = str(row.get('error_message', '')).lower()

            if not error_msg or error_msg == 'none':
                error_types.append('none')
            elif 'timeout' in error_msg:
                error_types.append('timeout')
            elif 'memory' in error_msg:
                error_types.append('memory')
            elif 'network' in error_msg or 'connection' in error_msg:
                error_types.append('network')
            elif 'validation' in error_msg or 'invalid' in error_msg:
                error_types.append('validation')
            else:
                error_types.append('other')

        return error_types


class CombinedDataLoader:
    """Combines data from both legal documents and agent execution."""

    def __init__(self, mysql_config: Optional[Dict] = None, sqlite_path: str = "jobs.db"):
        """Initialize with both data sources.

        Args:
            mysql_config: MySQL connection config for lawsuit_docs
            sqlite_path: Path to jobs.db SQLite database
        """
        self.legal_loader = LegalDataLoader(mysql_config)
        self.agent_loader = AgentDataLoader(sqlite_path)

    def load_all_data(self, legal_limit: Optional[int] = None, agent_limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from both sources.

        Args:
            legal_limit: Max legal cases to load
            agent_limit: Max agent records to load

        Returns:
            Tuple of (legal_data, agent_data) DataFrames
        """
        legal_data = self.legal_loader.load_case_law_data(legal_limit)
        agent_data = self.agent_loader.load_agent_execution_data(agent_limit)

        return legal_data, agent_data

    def create_training_datasets(self, legal_limit: Optional[int] = None, agent_limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Create training datasets for different ML tasks.

        Args:
            legal_limit: Max legal cases to load
            agent_limit: Max agent records to load

        Returns:
            Dictionary of training datasets
        """
        legal_data, agent_data = self.load_all_data(legal_limit, agent_limit)

        datasets = {
            'case_outcome': legal_data[['case_name', 'opinion_text', 'court', 'case_type', 'outcome_label']],
            'document_classification': legal_data[['case_name', 'opinion_text', 'case_type', 'domain_label', 'jurisdiction_label', 'complexity_label']],
            'agent_performance': agent_data[['job_type', 'phase', 'tokens_in', 'tokens_out', 'duration_seconds', 'success_label', 'performance_label']],
            'agent_efficiency': agent_data[['job_type', 'phase', 'budget_tokens', 'tokens_out', 'duration_seconds', 'efficiency_label']]
        }

        # Remove rows with missing labels
        for name, dataset in datasets.items():
            datasets[name] = dataset.dropna(subset=[col for col in dataset.columns if col.endswith('_label')])

        return datasets


# Convenience functions
def load_legal_data(limit: Optional[int] = None) -> pd.DataFrame:
    """Load legal case data."""
    loader = LegalDataLoader()
    return loader.load_case_law_data(limit)

def load_agent_data(limit: Optional[int] = None) -> pd.DataFrame:
    """Load agent execution data."""
    loader = AgentDataLoader()
    return loader.load_agent_execution_data(limit)

def load_combined_data(legal_limit: Optional[int] = None, agent_limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from both sources."""
    loader = CombinedDataLoader()
    return loader.load_all_data(legal_limit, agent_limit)
