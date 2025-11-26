# 🤖 Background AI Agent System for The Matrix

## 📋 Quick Reference

- **Purpose**: Continuously running local LLMs that work in the background on your gaming PC
- **Hardware**: 32GB RAM, multi-core CPU - perfect for running multiple models simultaneously
- **Cost**: $0 - completely free using open-source models
- **Status**: ✅ Ready to deploy

## 🎯 What This System Does

Unlike Cursor's background agents (which only work with GitHub), this is a **fully local, always-on AI system** that:

1. 🔍 **Monitors your legal document corpus** - Automatically extracts insights from new PDFs
2. 📊 **Continuous case law analysis** - Builds knowledge graphs from your 735+ Section 1782 cases
3. 💡 **Proactive research assistant** - Generates legal research summaries while you work
4. 🔗 **Citation network builder** - Maps relationships between cases and precedents
5. 📝 **Document preprocessing** - OCR, entity extraction, and classification in the background
6. 🧠 **Memory consolidation** - Updates your Bayesian network CPTs based on new evidence
7. ⚖️ **Settlement optimizer** - Runs Monte Carlo simulations on pending cases
8. 📈 **Pattern detection** - Identifies trends across your legal corpus

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│  Ollama Server (Background Service)                │
│  Running 3-4 models simultaneously on 32GB RAM     │
└─────────────────────────────────────────────────────┘
           ↓         ↓         ↓         ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Document │ │ Research │ │ Analysis │ │Citation  │
    │ Monitor  │ │ Agent    │ │ Agent    │ │ Agent    │
    │ Agent    │ │          │ │          │ │          │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘
         ↓            ↓            ↓            ↓
    ┌────────────────────────────────────────────────┐
    │  Task Queue (SQLite)                          │
    │  Priority system, checkpoint/resume           │
    └────────────────────────────────────────────────┘
         ↓            ↓            ↓            ↓
    ┌────────────────────────────────────────────────┐
    │  Results Database                              │
    │  Indexed, searchable insights                  │
    └────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Step 1: Install Ollama (1 minute)

```powershell
# Download and install Ollama for Windows
winget install Ollama.Ollama

# Or download from: https://ollama.com/download/windows
```

### Step 2: Pull Local Models (10-15 minutes)

```bash
# Legal analysis model (7B parameters, ~4GB RAM)
ollama pull llama3.2:7b

# Fast extraction model (3B parameters, ~2GB RAM)
ollama pull phi3:medium

# Research model (13B parameters, ~8GB RAM)
ollama pull mistral:13b

# Coding assistant (7B parameters, ~4GB RAM)
ollama pull codellama:7b
```

**Total RAM usage: ~16GB, leaving 16GB free for your work**

### Step 3: Install project dependencies

```powershell
cd C:\Users\User\Desktop\TheMatrix
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r background_agents/requirements.txt
```

### Step 4: Start the background system

```powershell
.\.venv\Scripts\python.exe background_agents\start_agents.py
```

> ✅ By default only the Project Organizer agent is enabled.  
> When Ollama and the desired models are ready, flip `enabled: true` for the other agents in `background_agents/config.yaml`.

### Step 5: Verify outputs

```powershell
.\.venv\Scripts\python.exe background_agents\status.py
.\.venv\Scripts\python.exe background_agents\view_insights.py --recent 10
```

## 🤖 Available Background Agents

### 1️⃣ Document Monitor Agent

**What it does:**
- Watches `1782 Case PDF Database/` and `Agents_1782_ML_Dataset/` directories
- Automatically processes new PDFs with OCR
- Extracts entities, dates, citations
- Classifies document types
- Updates your database with structured data

**Model:** `phi3:medium` (fast, efficient extraction)

**Priority:** HIGH (runs every 5 minutes)

**Output:** JSON files in `background_agents/outputs/document_analysis/`

---

### 2️⃣ Legal Research Agent

**What it does:**
- Analyzes your 735+ Section 1782 cases
- Identifies key legal principles and patterns
- Generates case summaries and abstracts
- Extracts precedent relationships
- Builds searchable knowledge base

**Model:** `llama3.2:7b` (balanced reasoning and speed)

**Priority:** MEDIUM (runs every 30 minutes)

**Output:** Markdown summaries in `background_agents/outputs/research/`

---

### 3️⃣ Citation Network Builder

**What it does:**
- Maps citation relationships between cases
- Identifies most influential precedents
- Detects citation clusters and patterns
- Calculates PageRank for case importance
- Visualizes network graphs

**Model:** `mistral:13b` (deep analysis)

**Priority:** LOW (runs every 2 hours)

**Output:** NetworkX graphs + visualizations in `background_agents/outputs/networks/`

---

### 4️⃣ Pattern Detection Agent

**What it does:**
- Analyzes outcomes across similar cases
- Identifies success/failure patterns
- Extracts judicial reasoning trends
- Detects jurisdiction-specific patterns
- Updates Bayesian network CPTs

**Model:** `llama3.2:7b` (statistical reasoning)

**Priority:** LOW (runs every 4 hours)

**Output:** JSON insights in `background_agents/outputs/patterns/`

---

### 5️⃣ Settlement Optimizer Agent

**What it does:**
- Runs Monte Carlo simulations on pending cases
- Updates settlement recommendations
- Analyzes risk profiles
- Generates strategy reports
- Monitors case law changes affecting valuations

**Model:** Python (no LLM needed - deterministic)

**Priority:** LOW (runs every 6 hours)

**Output:** Reports in `background_agents/outputs/settlements/`

---

### 6️⃣ Code Generation Agent

**What it does:**
- Generates utility scripts based on your patterns
- Creates data processing pipelines
- Builds visualization code
- Generates test cases
- Refactors existing code

**Model:** `codellama:7b` (specialized for code)

**Priority:** LOW (runs on-demand)

**Output:** Python scripts in `background_agents/outputs/code/`

---

## 💡 Usage Examples

### Example 1: Monitor Document Processing

```bash
# Watch real-time document processing
python background_agents/monitor.py --agent document_monitor

# Output:
# [2025-10-21 14:23:45] 📄 Processing: harvard_case_2019.pdf
# [2025-10-21 14:23:52] ✅ Extracted 45 entities, 12 citations
# [2025-10-21 14:23:52] 💾 Saved to: documents/harvard_case_2019.json
```

### Example 2: Query Research Agent

```bash
# Ask questions about your corpus
python background_agents/query.py "What are common denial reasons in 1782 cases?"

# Output:
# 🔍 Analyzing 735 cases...
#
# Top 5 Denial Reasons:
# 1. Lack of foreign tribunal proceeding (38% of denials)
# 2. Fishing expedition concerns (27%)
# 3. Burden on respondent (18%)
# 4. Evidence not relevant to foreign matter (12%)
# 5. Discovery too broad (5%)
#
# 📊 Full report: background_agents/outputs/research/denial_analysis_20251021.md
```

### Example 3: View Citation Network

```bash
# Generate citation network visualization
python background_agents/visualize.py --type citation_network

# Opens interactive graph showing:
# - Most cited cases (larger nodes)
# - Citation relationships (edges)
# - Influence clusters (colors)
# - Timeline progression
```

### Example 4: Get Daily Summary

```bash
# Get summary of what agents discovered today
python background_agents/daily_summary.py

# Output:
# 📊 Background Agent Summary - Oct 21, 2025
#
# 📄 Documents Processed: 12 PDFs
# 🔍 New Insights: 34 patterns identified
# 📚 Research Notes: 8 case summaries generated
# 🔗 Citations Mapped: 156 new relationships
# 💡 Recommendations: 3 settlement updates
#
# 🎯 Top Insight:
# "Increased success rate for 1782 petitions when
#  demonstrating foreign tribunal acceptance (85% vs 42%)"
```

### Example 5: Scheduled Activity Digest (for all agents)

Generate a shared digest of recent work so any agent can quickly orient:

```bash
# One‑off, last 7 days
python scripts/recent_activity_digest.py --days 7

# Run daily (every 24h) with a 7‑day lookback
python background_agents/schedule_activity_digest.py --interval-hours 24 --days-window 7
```

Outputs are written to:
- `reports/analysis_outputs/activity_digest.md`
- `reports/analysis_outputs/activity_digest.json`

Note: `background_agents/daily_summary.py` will also refresh the digest after printing the daily summary.

## ⚙️ Configuration

### Priority Levels

Edit `background_agents/config.yaml`:

```yaml
agents:
  document_monitor:
    enabled: true
    priority: high
    interval_minutes: 5
    model: "phi3:medium"

  legal_research:
    enabled: true
    priority: medium
    interval_minutes: 30
    model: "llama3.2:7b"
    max_concurrent_analyses: 5

  citation_network:
    enabled: true
    priority: low
    interval_hours: 2
    model: "mistral:13b"

  pattern_detection:
    enabled: true
    priority: low
    interval_hours: 4
    model: "llama3.2:7b"

  settlement_optimizer:
    enabled: true
    priority: low
    interval_hours: 6
    model: null  # deterministic

system:
  max_ram_usage_gb: 16  # Leave 16GB free
  max_cpu_cores: 12     # Leave 4 cores free
  checkpoint_interval_minutes: 15
  log_level: "INFO"
```

### Directory Monitoring

```yaml
monitored_directories:
  - path: "C:/Users/User/Desktop/TheMatrix/1782 Case PDF Database"
    watch_extensions: [".pdf", ".docx"]
    auto_process: true

  - path: "C:/Users/User/Desktop/TheMatrix/Agents_1782_ML_Dataset"
    watch_extensions: [".pdf", ".json"]
    auto_process: true

  - path: "C:/Users/User/Desktop/TheMatrix/case_law"
    watch_extensions: [".pdf", ".txt", ".html"]
    auto_process: true
```

## 📊 Performance Metrics

### Expected Resource Usage

| Agent | RAM Usage | CPU Usage | GPU Usage | Runs per Day |
|-------|-----------|-----------|-----------|--------------|
| Document Monitor | 2-4GB | 20-40% | Optional | 288 (every 5 min) |
| Legal Research | 4-6GB | 40-60% | Optional | 48 (every 30 min) |
| Citation Network | 8-10GB | 60-80% | Optional | 12 (every 2 hours) |
| Pattern Detection | 4-6GB | 40-60% | Optional | 6 (every 4 hours) |
| Settlement Optimizer | 1-2GB | 10-20% | No | 4 (every 6 hours) |

**Total Peak Usage:** ~16GB RAM, 40-60% CPU average

**Your Available:** 32GB RAM, high-core CPU → **Perfect fit!**

## 🎯 Practical Benefits

### 1. **Wake Up to Insights**
Every morning, check what your agents discovered overnight:
- New case summaries
- Updated citation networks
- Pattern insights
- Settlement recommendations

### 2. **Continuous Learning**
Your system gets smarter as you add more cases:
- Bayesian networks update automatically
- Citation importance recalculates
- Pattern detection improves

### 3. **Zero-Cost Research**
Instead of paying for API calls:
- Run unlimited analyses locally
- No usage limits
- Complete privacy

### 4. **Proactive Assistance**
Agents notify you when they find:
- Relevant new precedents
- Pattern changes
- Important citations
- Settlement opportunities

## 🔧 Advanced Features

### Custom Agent Creation

```python
from background_agents import BackgroundAgent, AgentConfig

# Create custom agent
class CustomResearchAgent(BackgroundAgent):
    def __init__(self):
        config = AgentConfig(
            name="Custom Researcher",
            model="llama3.2:7b",
            priority="medium",
            interval_minutes=60
        )
        super().__init__(config)

    async def process(self, task):
        # Your custom logic
        result = await self.llm_query(
            prompt=f"Analyze: {task.content}",
            temperature=0.7,
            max_tokens=2000
        )
        return result

# Register and start
agent_system.register(CustomResearchAgent())
```

### Integration with Existing Systems

```python
# Integrate with your Bayesian Network
from background_agents import PatternDetectionAgent
from bayesian_network.code.experiments.The_Matrix2_0_0_STABLE import BayesianNetworkEngine

agent = PatternDetectionAgent()
agent.on_pattern_found = lambda pattern: bn_engine.update_cpt(pattern)

# Integrate with writer agents
from background_agents import ResearchAgent
from writer_agents.code.master_supervisor import MasterSupervisor

research_agent.on_insight_found = lambda insight: supervisor.add_context(insight)
```

### Notification System

```yaml
notifications:
  enabled: true
  methods:
    - type: "file"
      path: "background_agents/notifications.txt"

    - type: "email"
      enabled: false
      smtp_server: "smtp.gmail.com"
      recipient: "you@email.com"

    - type: "desktop"
      enabled: true
      min_priority: "medium"
```

## 📚 Output Structure

```
background_agents/
├── outputs/
│   ├── document_analysis/
│   │   ├── 20251021_143245_harvard_case.json
│   │   ├── 20251021_145612_stanford_case.json
│   │   └── summary_20251021.md
│   │
│   ├── research/
│   │   ├── case_summaries/
│   │   ├── legal_principles/
│   │   ├── trend_analysis/
│   │   └── daily_insights_20251021.md
│   │
│   ├── networks/
│   │   ├── citation_graph_20251021.gexf
│   │   ├── influence_scores.json
│   │   └── network_visualization.html
│   │
│   ├── patterns/
│   │   ├── success_factors.json
│   │   ├── judicial_trends.json
│   │   └── bayesian_updates.json
│   │
│   └── settlements/
│       ├── monte_carlo_results/
│       ├── recommendations/
│       └── risk_profiles/
│
├── logs/
│   ├── document_monitor.log
│   ├── research_agent.log
│   └── system.log
│
└── checkpoints/
    ├── document_monitor_state.json
    └── research_agent_state.json
```

## 🚀 Running as Windows Service

To run agents 24/7 automatically:

```powershell
# Install as Windows service
python background_agents/install_service.py

# Start service
net start TheMatrixBackgroundAgents

# Check status
python background_agents/status.py

# Stop service
net stop TheMatrixBackgroundAgents
```

Service will:
- ✅ Start automatically on boot
- ✅ Restart on failure
- ✅ Run in background (no console window)
- ✅ Log to files
- ✅ Respect resource limits

## 💰 Cost Comparison

| Approach | Cost per Month | Speed | Privacy |
|----------|---------------|-------|---------|
| **OpenAI API (GPT-4)** | $200-500 | Fast | ❌ Cloud |
| **Anthropic API (Claude)** | $150-400 | Fast | ❌ Cloud |
| **Background Agents (Local)** | **$0** | Medium-Fast | ✅ 100% Private |

**Savings over 1 year:** $1,800 - $6,000

## 🎓 Learning from Background Work

The system maintains a knowledge base that improves over time:

```bash
# Ask questions about what agents learned
python background_agents/ask.py "What did you learn about Harvard discrimination cases?"

# Output draws from all agent discoveries:
# - Document analysis findings
# - Research insights
# - Citation patterns
# - Statistical patterns
```

## 🔐 Security & Privacy

**100% Local Processing:**
- ✅ No data leaves your PC
- ✅ No API keys needed (for local models)
- ✅ No usage tracking
- ✅ Complete control over data

**Optional Cloud Integration:**
- Can still use OpenAI for complex tasks
- Hybrid mode: local for routine, cloud for complex
- Cost optimization: 90% local, 10% cloud

## 🎯 Next Steps

1. **[Install Ollama](https://ollama.com/download/windows)** (2 minutes)
2. **Pull models** (10-15 minutes)
3. **Run `start_agents.py`** (30 seconds)
4. **Check back in 1 hour** to see first results
5. **Wake up tomorrow** with overnight insights

## 📊 Success Metrics

After 24 hours of running, you should see:
- ✅ 100+ documents analyzed
- ✅ 50+ research insights generated
- ✅ Complete citation network built
- ✅ Pattern reports available
- ✅ Settlement recommendations updated

**Your gaming PC finally working as hard as it should!** 🎮🤖

---

**Status:** ✅ Ready to deploy
**Cost:** $0 (completely free)
**Privacy:** 100% local
**Requirements:** 16GB RAM available (you have 32GB ✅)

Let's make your powerful gaming PC actually do something useful in the background! 🚀
