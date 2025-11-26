#!/usr/bin/env python3
"""
Comprehensive OCR + NLP Analysis Pipeline for 1782 Discovery Cases
================================================================

This script performs:
1. OCR on scanned PDFs using pytesseract
2. Text extraction from readable PDFs
3. Comprehensive NLP analysis using spaCy, NLTK, and transformers
4. Pattern detection and knowledge graph construction
5. Legal outcome prediction and analysis

Features:
- Automatic OCR for scanned documents
- Multi-stage NLP analysis
- Legal pattern recognition
- Entity and relationship extraction
- Knowledge graph construction
- Outcome prediction modeling
"""

import os
import json
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import PyPDF2
import fitz  # PyMuPDF for PDF image extraction

# OCR and Image Processing
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR not available. Install: pip install pytesseract pillow")

# NLP Libraries
try:
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from textblob import TextBlob
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    NLP_AVAILABLE = True
except ImportError as e:
    NLP_AVAILABLE = False
    print(f"⚠️ NLP libraries not available: {e}")

# Graph Analysis
try:
    import networkx as nx
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    print("⚠️ NetworkX not available for graph analysis")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Comprehensive1782Analyzer:
    """Comprehensive analyzer for 1782 discovery cases."""

    def __init__(self, pdf_directory: str = "data/case_law/1782_recap_api_pdfs"):
        """Initialize the analyzer."""
        self.pdf_directory = Path(pdf_directory)
        self.output_dir = Path("data/case_law/analysis_results")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize NLP models
        self._setup_nlp_models()

        # Legal patterns and keywords
        self._setup_legal_patterns()

        # Analysis results storage
        self.extracted_texts = {}
        self.analysis_results = {}
        self.patterns_found = defaultdict(list)

        logger.info("✅ Comprehensive 1782 Analyzer initialized")

    def _setup_nlp_models(self):
        """Setup NLP models and components."""
        if not NLP_AVAILABLE:
            logger.warning("NLP libraries not available - limited analysis")
            return

        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded")
        except OSError:
            logger.error("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        try:
            # Initialize sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Sentence transformer failed: {e}")
            self.sentence_model = None

        try:
            # Setup NLTK
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("✅ NLTK components loaded")
        except Exception as e:
            logger.warning(f"⚠️ NLTK setup failed: {e}")
            self.stop_words = set()
            self.lemmatizer = None

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3)
        )

    def _setup_legal_patterns(self):
        """Setup legal patterns for 1782 analysis."""
        self.legal_patterns = {
            'outcome_keywords': [
                'granted', 'denied', 'approved', 'rejected', 'sustained', 'overruled',
                'motion granted', 'motion denied', 'petition granted', 'petition denied',
                'discovery granted', 'discovery denied', 'application granted', 'application denied'
            ],
            'jurisdiction_keywords': [
                'district court', 'federal court', 'circuit court', 'bankruptcy court',
                'magistrate judge', 'district judge', 'circuit judge'
            ],
            'discovery_keywords': [
                'discovery', 'deposition', 'interrogatory', 'request for production',
                'subpoena', 'document production', 'witness testimony'
            ],
            'international_keywords': [
                'foreign proceeding', 'international', 'cross-border', 'multinational',
                'foreign court', 'foreign tribunal', 'arbitration', 'foreign litigation'
            ],
            'procedure_keywords': [
                'motion', 'petition', 'application', 'order', 'memorandum', 'brief',
                'hearing', 'oral argument', 'written submission'
            ]
        }

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from PDF using multiple methods."""
        try:
            # Method 1: Direct text extraction
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

            # If we got substantial text, return it
            if len(text.strip()) > 100:
                logger.info(f"✅ Direct text extraction: {pdf_path.name} ({len(text)} chars)")
                return text.strip()

            # Method 2: OCR for scanned PDFs
            if OCR_AVAILABLE:
                logger.info(f"🔄 Attempting OCR: {pdf_path.name}")
                ocr_text = self._extract_text_with_ocr(pdf_path)
                if ocr_text and len(ocr_text.strip()) > 50:
                    logger.info(f"✅ OCR successful: {pdf_path.name} ({len(ocr_text)} chars)")
                    return ocr_text.strip()

            logger.warning(f"⚠️ No text extracted: {pdf_path.name}")
            return None

        except Exception as e:
            logger.error(f"❌ Error extracting text from {pdf_path.name}: {e}")
            return None

    def _extract_text_with_ocr(self, pdf_path: Path) -> Optional[str]:
        """Extract text using OCR on PDF images."""
        if not OCR_AVAILABLE:
            logger.warning(f"OCR not available for {pdf_path.name}")
            return None

        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            ocr_text = ""

            for page_num in range(min(3, len(doc))):  # OCR first 3 pages max
                page = doc[page_num]

                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # Convert to PIL Image
                import io
                image = Image.open(io.BytesIO(img_data))

                # Perform OCR
                page_text = pytesseract.image_to_string(image, config='--psm 6')
                ocr_text += page_text + "\n"

            doc.close()
            return ocr_text

        except Exception as e:
            logger.error(f"OCR failed for {pdf_path.name}: {e}")
            return None

    def analyze_text_with_nlp(self, text: str, filename: str) -> Dict[str, Any]:
        """Perform comprehensive NLP analysis on text."""
        if not NLP_AVAILABLE or not self.nlp:
            return {"error": "NLP not available"}

        analysis = {
            'filename': filename,
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'entities': [],
            'legal_patterns': {},
            'sentiment': {},
            'key_phrases': [],
            'outcome_prediction': None
        }

        try:
            # spaCy analysis
            doc = self.nlp(text)

            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            analysis['entities'] = entities

            # Legal pattern detection
            text_lower = text.lower()
            for pattern_type, keywords in self.legal_patterns.items():
                found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
                analysis['legal_patterns'][pattern_type] = found_keywords

            # Sentiment analysis
            blob = TextBlob(text)
            analysis['sentiment'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }

            # Extract key phrases (noun phrases)
            key_phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    key_phrases.append(chunk.text.strip())
            analysis['key_phrases'] = key_phrases[:20]  # Top 20

            # Outcome prediction
            analysis['outcome_prediction'] = self._predict_outcome(text)

        except Exception as e:
            logger.error(f"NLP analysis failed for {filename}: {e}")
            analysis['error'] = str(e)

        return analysis

    def _predict_outcome(self, text: str) -> Dict[str, Any]:
        """Predict case outcome based on text patterns."""
        text_lower = text.lower()

        # Count positive and negative outcome indicators
        positive_indicators = ['granted', 'approved', 'sustained', 'motion granted', 'petition granted']
        negative_indicators = ['denied', 'rejected', 'overruled', 'motion denied', 'petition denied']

        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)

        total_indicators = positive_count + negative_count

        if total_indicators == 0:
            return {'prediction': 'unclear', 'confidence': 0.0, 'reasoning': 'No clear outcome indicators'}

        if positive_count > negative_count:
            prediction = 'granted'
            confidence = positive_count / total_indicators
        elif negative_count > positive_count:
            prediction = 'denied'
            confidence = negative_count / total_indicators
        else:
            prediction = 'mixed'
            confidence = 0.5

        return {
            'prediction': prediction,
            'confidence': confidence,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'reasoning': f'Found {positive_count} positive and {negative_count} negative indicators'
        }

    def process_all_pdfs(self) -> Dict[str, Any]:
        """Process all PDFs in the directory."""
        logger.info("🚀 Starting comprehensive PDF analysis...")

        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"📁 Found {len(pdf_files)} PDF files")

        results = {
            'total_files': len(pdf_files),
            'processed_files': 0,
            'successful_extractions': 0,
            'ocr_files': 0,
            'analysis_results': {},
            'summary_stats': {},
            'patterns_summary': {}
        }

        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")

            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                results['successful_extractions'] += 1
                self.extracted_texts[pdf_file.name] = text

                # Perform NLP analysis
                analysis = self.analyze_text_with_nlp(text, pdf_file.name)
                results['analysis_results'][pdf_file.name] = analysis

                # Track OCR usage
                if 'OCR' in str(text) or len(text) < 500:  # Heuristic for OCR
                    results['ocr_files'] += 1

                results['processed_files'] += 1

            # Progress update every 10 files
            if i % 10 == 0:
                logger.info(f"📊 Progress: {i}/{len(pdf_files)} files processed")

        # Generate summary statistics
        results['summary_stats'] = self._generate_summary_stats(results['analysis_results'])
        results['patterns_summary'] = self._summarize_patterns(results['analysis_results'])

        logger.info(f"✅ Analysis complete: {results['processed_files']}/{results['total_files']} files processed")
        return results

    def _generate_summary_stats(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results."""
        if not analysis_results:
            return {}

        stats = {
            'total_files_analyzed': len(analysis_results),
            'total_text_length': sum(result.get('text_length', 0) for result in analysis_results.values()),
            'average_text_length': 0,
            'outcome_predictions': Counter(),
            'entity_counts': Counter(),
            'court_distributions': Counter(),
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
        }

        # Calculate averages and distributions
        text_lengths = [result.get('text_length', 0) for result in analysis_results.values()]
        stats['average_text_length'] = np.mean(text_lengths) if text_lengths else 0

        # Count outcomes
        for result in analysis_results.values():
            outcome = result.get('outcome_prediction', {}).get('prediction', 'unclear')
            stats['outcome_predictions'][outcome] += 1

            # Count entities
            for entity in result.get('entities', []):
                stats['entity_counts'][entity.get('label', 'UNKNOWN')] += 1

            # Sentiment distribution
            sentiment = result.get('sentiment', {}).get('polarity', 0)
            if sentiment > 0.1:
                stats['sentiment_distribution']['positive'] += 1
            elif sentiment < -0.1:
                stats['sentiment_distribution']['negative'] += 1
            else:
                stats['sentiment_distribution']['neutral'] += 1

        return stats

    def _summarize_patterns(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize legal patterns found across all documents."""
        pattern_summary = defaultdict(list)

        for filename, result in analysis_results.items():
            legal_patterns = result.get('legal_patterns', {})
            for pattern_type, keywords in legal_patterns.items():
                pattern_summary[pattern_type].extend(keywords)

        # Count frequency of each pattern
        pattern_counts = {}
        for pattern_type, keywords in pattern_summary.items():
            pattern_counts[pattern_type] = Counter(keywords)

        return dict(pattern_counts)

    def save_results(self, results: Dict[str, Any]):
        """Save analysis results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        results_file = self.output_dir / f"1782_analysis_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save summary report
        summary_file = self.output_dir / f"1782_summary_report_{timestamp}.md"
        self._generate_summary_report(results, summary_file)

        logger.info(f"💾 Results saved to: {results_file}")
        logger.info(f"📋 Summary report saved to: {summary_file}")

    def _generate_summary_report(self, results: Dict[str, Any], output_file: Path):
        """Generate a markdown summary report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 1782 Discovery Cases - Comprehensive Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("## 📊 Overview\n\n")
            f.write(f"- **Total PDFs:** {results['total_files']}\n")
            f.write(f"- **Successfully Processed:** {results['processed_files']}\n")
            f.write(f"- **Text Extracted:** {results['successful_extractions']}\n")
            f.write(f"- **OCR Required:** {results['ocr_files']}\n\n")

            # Summary Statistics
            stats = results.get('summary_stats', {})
            f.write("## 📈 Summary Statistics\n\n")
            f.write(f"- **Average Text Length:** {stats.get('average_text_length', 0):.0f} characters\n")
            f.write(f"- **Total Text Analyzed:** {stats.get('total_text_length', 0):,} characters\n\n")

            # Outcome Predictions
            f.write("## ⚖️ Outcome Predictions\n\n")
            outcome_counts = stats.get('outcome_predictions', {})
            for outcome, count in outcome_counts.most_common():
                percentage = (count / results['processed_files']) * 100
                f.write(f"- **{outcome.title()}:** {count} cases ({percentage:.1f}%)\n")
            f.write("\n")

            # Entity Analysis
            f.write("## 🏷️ Entity Analysis\n\n")
            entity_counts = stats.get('entity_counts', {})
            for entity_type, count in entity_counts.most_common(10):
                f.write(f"- **{entity_type}:** {count} occurrences\n")
            f.write("\n")

            # Legal Patterns
            f.write("## ⚖️ Legal Patterns Found\n\n")
            patterns = results.get('patterns_summary', {})
            for pattern_type, keyword_counts in patterns.items():
                f.write(f"### {pattern_type.replace('_', ' ').title()}\n\n")
                for keyword, count in keyword_counts.most_common(5):
                    f.write(f"- **{keyword}:** {count} occurrences\n")
                f.write("\n")

            # Sentiment Analysis
            f.write("## 😊 Sentiment Analysis\n\n")
            sentiment = stats.get('sentiment_distribution', {})
            total_sentiment = sum(sentiment.values())
            if total_sentiment > 0:
                for sentiment_type, count in sentiment.items():
                    percentage = (count / total_sentiment) * 100
                    f.write(f"- **{sentiment_type.title()}:** {count} cases ({percentage:.1f}%)\n")
            f.write("\n")

            f.write("---\n")
            f.write("*Report generated by Comprehensive 1782 Analyzer*\n")

def main():
    """Main execution function."""
    print("🚀 Starting Comprehensive 1782 Discovery Cases Analysis")
    print("=" * 60)

    # Check dependencies
    if not OCR_AVAILABLE:
        print("⚠️ OCR not available - some PDFs may not be processed")
    if not NLP_AVAILABLE:
        print("⚠️ NLP libraries not available - limited analysis")

    # Initialize analyzer
    analyzer = Comprehensive1782Analyzer()

    # Process all PDFs
    results = analyzer.process_all_pdfs()

    # Save results
    analyzer.save_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE!")
    print(f"✅ Processed: {results['processed_files']}/{results['total_files']} files")
    print(f"📄 Text extracted: {results['successful_extractions']} files")
    print(f"🔍 OCR used: {results['ocr_files']} files")

    # Outcome summary
    outcome_counts = results['summary_stats'].get('outcome_predictions', {})
    if outcome_counts:
        print("\n⚖️ OUTCOME PREDICTIONS:")
        for outcome, count in outcome_counts.most_common():
            percentage = (count / results['processed_files']) * 100
            print(f"  {outcome.title()}: {count} cases ({percentage:.1f}%)")

    print(f"\n💾 Results saved to: data/case_law/analysis_results/")
    print("🎯 Ready for pattern analysis and knowledge graph construction!")

if __name__ == "__main__":
    main()
