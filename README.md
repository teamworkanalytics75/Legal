# Legal Tech - Motion to Seal Pipeline

Automated legal document generation system focused on motion to seal proceedings.

## 🎯 Core Features

- **Motion Generation**: Automated creation of legal motions with citations
- **Case Law Analysis**: ML-powered analysis of precedent cases
- **Background Agents**: Automated research and document processing
- **Feature Extraction**: NLP-based legal document feature analysis

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements_upwork.txt

# Generate a motion
python scripts/motion_generation/create_motion_local.py

# Run analysis
python scripts/analysis/analyze_burroughs_rulings.py
```

## 📁 Structure

```
LegalTech-MotionToSeal/
├── scripts/
│   ├── motion_generation/  # Motion creation scripts
│   └── analysis/           # Case analysis tools
├── background_agents/       # Automated research agents
├── ml_system/              # ML models and pipelines
├── docs/                   # Documentation
└── plans/                  # Project plans
```

## 📋 Requirements

- Python 3.9+
- Local LLM (Ollama) or OpenAI API
- Case law database (optional)

## 🔒 Note

Large data files (models, case databases) are excluded.
See setup docs for downloading required assets.

