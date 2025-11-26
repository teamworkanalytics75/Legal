#!/usr/bin/env python3
"""
Create SQL database schema for §1782 case law analysis.

Sets up tables to store PDF analysis results, entities, relations,
causal links, and Bayesian evidence from NLP processing.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/case_law/database_setup.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def create_1782_database(db_path: Path) -> bool:
    """
    Create SQLite database with schema for §1782 case law analysis.

    Args:
        db_path: Path to SQLite database file

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating §1782 database at: {db_path}")

    try:
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create tables

        # Cases table - basic case information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                cluster_id INTEGER PRIMARY KEY,
                case_name TEXT NOT NULL,
                court TEXT,
                date_filed TEXT,
                date_decided TEXT,
                pdf_path TEXT,
                json_path TEXT,
                text_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Entities table - extracted entities from NLP
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                confidence REAL,
                start_pos INTEGER,
                end_pos INTEGER,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Relations table - entity relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                subject_entity TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity TEXT NOT NULL,
                confidence REAL,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Causal relations table - causal inference results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS causal_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                cause TEXT NOT NULL,
                effect TEXT NOT NULL,
                confidence REAL,
                evidence TEXT,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Bayesian evidence table - evidence statements for BN
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bayesian_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                node TEXT NOT NULL,
                evidence_text TEXT NOT NULL,
                probability REAL,
                context TEXT,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Analysis summary table - high-level analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_summary (
                cluster_id INTEGER PRIMARY KEY,
                num_entities INTEGER,
                num_relations INTEGER,
                num_causal_links INTEGER,
                num_bn_nodes INTEGER,
                num_evidence_statements INTEGER,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Create additional tables for §1782 analysis

        # Intel factors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intel_factors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                factor_number INTEGER,  -- 1-4
                factor_name TEXT,       -- 'foreign_participant', 'receptivity', etc.
                detection_method TEXT,  -- 'explicit' or 'semantic'
                text_span TEXT,
                weight REAL,           -- estimated importance
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Statutory prerequisites table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statutory_prerequisites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                prereq_name TEXT NOT NULL,  -- 'foreign_tribunal', 'interested_person', etc.
                satisfied BOOLEAN,
                text_evidence TEXT,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Outcomes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                cluster_id INTEGER PRIMARY KEY,
                outcome TEXT NOT NULL,  -- 'granted', 'denied', 'partial'
                confidence REAL,
                disposition_text TEXT,
                judge_name TEXT,
                date_decided TEXT,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Citations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                cited_case TEXT NOT NULL,
                context TEXT,
                favorable BOOLEAN,  -- inferred from surrounding language
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Structural features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS structural_features (
                cluster_id INTEGER PRIMARY KEY,
                page_count INTEGER,
                word_count INTEGER,
                paragraph_count INTEGER,
                num_section_headers INTEGER,
                has_proposed_order BOOLEAN,
                has_memorandum BOOLEAN,
                has_affidavit BOOLEAN,
                has_exhibits BOOLEAN,
                has_declaration BOOLEAN,
                FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_cluster ON entities (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_cluster ON relations (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_causal_cluster ON causal_relations (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bayesian_cluster ON bayesian_evidence (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intel_factors_cluster ON intel_factors (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_statutory_cluster ON statutory_prerequisites (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_cluster ON citations (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_case ON citations (cited_case)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_outcome ON outcomes (outcome)')

        # Commit changes
        conn.commit()
        conn.close()

        logger.info("Database schema created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


def populate_from_enhanced_extraction(db_path: Path, extraction_results: List[Dict]) -> bool:
    """
    Populate database with enhanced extraction results.

    Args:
        db_path: Path to SQLite database
        extraction_results: List of case analysis dictionaries

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Populating database with {len(extraction_results)} enhanced extractions")

    if not extraction_results:
        logger.warning("No extraction results provided")
        return False

    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Insert data for each analysis
        for analysis in extraction_results:
            cluster_id = int(analysis['cluster_id'])

            # Insert case
            cursor.execute('''
                INSERT OR REPLACE INTO cases
                (cluster_id, case_name, pdf_path, json_path, text_length)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                cluster_id,
                analysis.get('case_name', 'Unknown'),
                analysis.get('pdf_path'),
                analysis.get('json_path'),
                analysis.get('text_length', 0)
            ))

            # Insert structural features
            structural = analysis.get('structural', {})
            cursor.execute('''
                INSERT OR REPLACE INTO structural_features
                (cluster_id, page_count, word_count, paragraph_count, num_section_headers,
                 has_proposed_order, has_memorandum, has_affidavit, has_exhibits, has_declaration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cluster_id,
                structural.get('page_count', 0),
                structural.get('word_count', 0),
                structural.get('paragraph_count', 0),
                structural.get('num_section_headers', 0),
                structural.get('has_proposed_order', False),
                structural.get('has_memorandum', False),
                structural.get('has_affidavit', False),
                structural.get('has_exhibits', False),
                structural.get('has_declaration', False)
            ))

            # Insert outcome
            outcome = analysis.get('outcome', {})
            judge_info = analysis.get('judge_info', {})
            date_info = analysis.get('date_info', {})
            cursor.execute('''
                INSERT OR REPLACE INTO outcomes
                (cluster_id, outcome, confidence, disposition_text, judge_name, date_decided)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                cluster_id,
                outcome.get('outcome', 'unclear'),
                outcome.get('confidence', 0.0),
                outcome.get('disposition_text'),
                judge_info.get('primary_judge'),
                date_info.get('decision_date')
            ))

            # Insert Intel factors
            intel_factors = analysis.get('intel_factors', {})
            for factor_name, factor_data in intel_factors.items():
                if factor_data.get('detected'):
                    factor_number = int(factor_name.split('_')[1]) if '_' in factor_name else None
                    cursor.execute('''
                        INSERT INTO intel_factors
                        (cluster_id, factor_number, factor_name, detection_method, text_span, weight)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        cluster_id,
                        factor_number,
                        factor_name,
                        factor_data.get('detection_method', 'unknown'),
                        '; '.join(factor_data.get('all_contexts', [])[:3]),  # First 3 contexts
                        factor_data.get('total_weight', 0.0)
                    ))

            # Insert statutory prerequisites
            statutory_prereqs = analysis.get('statutory_prereqs', {})
            for prereq_name, prereq_data in statutory_prereqs.items():
                if prereq_data.get('found'):
                    cursor.execute('''
                        INSERT INTO statutory_prerequisites
                        (cluster_id, prereq_name, satisfied, text_evidence)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        cluster_id,
                        prereq_name,
                        prereq_data.get('satisfied', False),
                        '; '.join(prereq_data.get('contexts', [])[:2])  # First 2 contexts
                    ))

            # Insert citations
            citations = analysis.get('citations', [])
            for citation in citations:
                cursor.execute('''
                    INSERT INTO citations
                    (cluster_id, cited_case, context, favorable)
                    VALUES (?, ?, ?, ?)
                ''', (
                    cluster_id,
                    citation.get('case_name', ''),
                    citation.get('context', ''),
                    citation.get('favorable')
                ))

        # Commit changes
        conn.commit()
        conn.close()

        logger.info(f"Successfully populated database with {len(extraction_results)} case analyses")
        return True

    except Exception as e:
        logger.error(f"Error populating database: {e}")
        return False


def populate_from_nlp_results(db_path: Path, nlp_results_file: Path) -> bool:
    """
    Populate database with NLP analysis results.

    Args:
        db_path: Path to SQLite database
        nlp_results_file: Path to combined NLP analysis JSON

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Populating database from: {nlp_results_file}")

    if not nlp_results_file.exists():
        logger.error(f"NLP results file not found: {nlp_results_file}")
        return False

    try:
        # Load NLP results
        with open(nlp_results_file, 'r', encoding='utf-8') as f:
            nlp_data = json.load(f)

        analyses = nlp_data.get('analyses', {})
        if not analyses:
            logger.warning("No analyses found in NLP results")
            return False

        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Insert data for each analysis
        for cluster_id, analysis in analyses.items():
            cluster_id = int(cluster_id)
            case_meta = analysis.get('case_metadata', {})
            summary = analysis.get('summary', {})

            # Insert case
            cursor.execute('''
                INSERT OR REPLACE INTO cases
                (cluster_id, case_name, pdf_path, json_path, text_length)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                cluster_id,
                case_meta.get('case_name', 'Unknown'),
                case_meta.get('pdf_path'),
                case_meta.get('json_path'),
                case_meta.get('text_length', 0)
            ))

            # Insert entities
            entities = analysis.get('entities', {}).get('entities', [])
            for entity in entities:
                cursor.execute('''
                    INSERT INTO entities
                    (cluster_id, entity_text, entity_type, confidence, start_pos, end_pos)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    cluster_id,
                    entity.get('text', ''),
                    entity.get('type', ''),
                    entity.get('confidence', 0.0),
                    entity.get('start', 0),
                    entity.get('end', 0)
                ))

            # Insert relations
            relations = analysis.get('entities', {}).get('relations', [])
            for relation in relations:
                cursor.execute('''
                    INSERT INTO relations
                    (cluster_id, subject_entity, predicate, object_entity, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    cluster_id,
                    relation.get('subject', ''),
                    relation.get('predicate', ''),
                    relation.get('object', ''),
                    relation.get('confidence', 0.0)
                ))

            # Insert causal relations
            causal_relations = analysis.get('causal', {}).get('causal_relations', [])
            for causal in causal_relations:
                cursor.execute('''
                    INSERT INTO causal_relations
                    (cluster_id, cause, effect, confidence, evidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    cluster_id,
                    causal.get('cause', ''),
                    causal.get('effect', ''),
                    causal.get('confidence', 0.0),
                    causal.get('evidence', '')
                ))

            # Insert Bayesian evidence
            bayesian_evidence = analysis.get('bayesian', {}).get('evidence', [])
            for evidence in bayesian_evidence:
                cursor.execute('''
                    INSERT INTO bayesian_evidence
                    (cluster_id, node, evidence_text, probability, context)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    cluster_id,
                    evidence.get('node', ''),
                    evidence.get('text', ''),
                    evidence.get('probability', 0.0),
                    evidence.get('context', '')
                ))

            # Insert analysis summary
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_summary
                (cluster_id, num_entities, num_relations, num_causal_links, num_bn_nodes, num_evidence_statements)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                cluster_id,
                summary.get('num_entities', 0),
                summary.get('num_relations', 0),
                summary.get('num_causal_links', 0),
                summary.get('num_bn_nodes', 0),
                summary.get('num_evidence_statements', 0)
            ))

        # Commit changes
        conn.commit()
        conn.close()

        logger.info(f"Successfully populated database with {len(analyses)} case analyses")
        return True

    except Exception as e:
        logger.error(f"Error populating database: {e}")
        return False


def main():
    """Main execution function."""
    print("Setting up §1782 case law analysis database...")

    # Define paths
    db_path = Path("data/case_law/1782_analysis.db")
    nlp_results_file = Path("data/case_law/nlp_analysis_results/combined_nlp_analysis.json")

    # Create database
    if not create_1782_database(db_path):
        logger.error("Failed to create database")
        sys.exit(1)

    # Populate from NLP results (if available)
    if nlp_results_file.exists():
        if populate_from_nlp_results(db_path, nlp_results_file):
            logger.info("Database populated with NLP results")
        else:
            logger.warning("Failed to populate database with NLP results")
    else:
        logger.info("NLP results not yet available - database ready for future population")

    logger.info("Database setup completed")
    print(f"\n[SUCCESS] Database ready at: {db_path}")


if __name__ == "__main__":
    main()
