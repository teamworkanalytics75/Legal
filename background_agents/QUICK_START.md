# 🚀 Quick Start Guide - Background Agents

## ⚡ 5-Minute Setup

### Step 1: Install Ollama (2 minutes)

**Windows 11:**
```powershell
# Option 1: Using winget
winget install Ollama.Ollama

# Option 2: Download installer
# Go to: https://ollama.com/download/windows
# Run the installer
```

**Verify Installation:**
```powershell
ollama --version
```

### Step 2: Pull AI Models (10-15 minutes)

```bash
# Legal analysis model (~4GB download)
ollama pull llama3.2:7b

# Fast extraction model (~2GB download)
ollama pull phi3:medium

# Research model (~8GB download) - optional but recommended
ollama pull mistral:13b
```

**Progress Tracking:**
- Each model will show download progress
- Models are cached, so you only download once
- Total download: ~14GB (be patient!)

### Step 3: Install Python Dependencies (1 minute)

```powershell
cd C:\Users\User\Desktop\TheMatrix

# Install/upgrade dependencies directly with the project interpreter
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r background_agents/requirements.txt
```

### Step 4: Start the System (30 seconds)

```powershell
# Start the background system (Project Organizer is enabled by default)
.\.venv\Scripts\python.exe background_agents\start_agents.py
```

**You should see:**
```
🚀 Starting Background Agent System...

📋 Registered Agents:
  ✅ document_monitor (Priority: HIGH)
  ✅ legal_research (Priority: MEDIUM)
  ✅ citation_network (Priority: LOW)
  ✅ pattern_detection (Priority: LOW)
  ✅ settlement_optimizer (Priority: LOW)

🔧 System Configuration:
  Max RAM: 16GB
  Max CPU Cores: 12
  Log Level: INFO

============================================================
System is running! Press Ctrl+C to stop.
============================================================
```

### Step 5: Let It Run\r\n\r\n- Project Organizer performs a full inventory scan once per day (configurable)\r\n- Enable other agents after the Ollama stack is ready by toggling their `enabled` flag

### Step 6: Check Results\r\n\r\n```powershell\r\n.\.venv\Scripts\python.exe background_agents\status.py\r\n.\.venv\Scripts\python.exe background_agents\view_insights.py --recent 10\r\n```

---

## 🎯 What Happens Next?

### Immediate (First Hour)
- ✅ Document monitor scans your 735+ cases
- ✅ Starts extracting metadata from PDFs
- ✅ Creates structured JSON files

### Within 4 Hours
- ✅ Research summaries generated
- ✅ Legal principles extracted
- ✅ Citation network built
- ✅ Pattern analysis complete

### After 24 Hours
- ✅ Complete corpus analyzed
- ✅ Daily insights report ready
- ✅ Settlement recommendations updated
- ✅ Knowledge base fully populated

---

## 🔧 Troubleshooting

### Issue: "Ollama not found"

**Solution:**
```powershell
# Restart terminal after installing Ollama
# Or add to PATH manually:
$env:PATH += ";C:\Users\User\AppData\Local\Programs\Ollama"
```

### Issue: "Model not found"

**Solution:**
```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama3.2:7b
```

### Issue: "High RAM usage"

**Solution:**
Edit `background_agents/config.yaml`:
```yaml
system:
  max_ram_usage_gb: 12  # Reduce from 16 to 12

agents:
  citation_network:
    enabled: false  # Disable heavy agent
```

### Issue: "No outputs generated"

**Diagnosis:**
```bash
# Check logs
cat background_agents/logs/system.log

# Check task queue
python background_agents/status.py
```

**Common causes:**
- Agents haven't run yet (wait 30+ minutes)
- No files in monitored directories
- Ollama service not running

---

## 📊 Configuration Tips

### Minimal Setup (8GB RAM available)
```yaml
agents:
  document_monitor:
    enabled: true
    model: "phi3:medium"  # Lightweight

  legal_research:
    enabled: false  # Disable to save RAM

  citation_network:
    enabled: false  # Disable heavy agent

  pattern_detection:
    enabled: false

  settlement_optimizer:
    enabled: true  # No RAM needed (deterministic)
```

### Balanced Setup (16GB RAM available)
```yaml
agents:
  document_monitor:
    enabled: true
    model: "phi3:medium"

  legal_research:
    enabled: true
    model: "llama3.2:7b"

  citation_network:
    enabled: true
    interval_hours: 4  # Run less frequently

  pattern_detection:
    enabled: true
    interval_hours: 6

  settlement_optimizer:
    enabled: true
```

### Maximum Setup (32GB RAM available - Your PC!)
```yaml
agents:
  document_monitor:
    enabled: true
    model: "phi3:medium"
    interval_minutes: 5

  legal_research:
    enabled: true
    model: "llama3.2:7b"
    interval_minutes: 30

  citation_network:
    enabled: true
    model: "mistral:13b"  # Use powerful model
    interval_hours: 2

  pattern_detection:
    enabled: true
    model: "llama3.2:7b"
    interval_hours: 4

  settlement_optimizer:
    enabled: true
    interval_hours: 6
```

---

## 🎮 Running as Windows Service

To run 24/7 automatically:

### Create Service Script

Save as `background_agents/install_service.ps1`:

```powershell
# Run as Administrator

$serviceName = "TheMatrixBackgroundAgents"
$pythonPath = "C:\Users\User\Desktop\TheMatrix\.venv\Scripts\python.exe"
$scriptPath = "C:\Users\User\Desktop\TheMatrix\background_agents\start_agents.py"

# Create service using NSSM (Non-Sucking Service Manager)
# Download from: https://nssm.cc/download

nssm install $serviceName $pythonPath $scriptPath
nssm set $serviceName AppDirectory "C:\Users\User\Desktop\TheMatrix"
nssm set $serviceName DisplayName "The Matrix Background Agents"
nssm set $serviceName Description "Continuously running AI agents for legal analysis"
nssm set $serviceName Start SERVICE_AUTO_START

# Start service
net start $serviceName

Write-Host "✅ Service installed and started!"
```

### Manual Start (Without Service)

**Option 1: Keep Terminal Open**
```bash
python background_agents/start_agents.py
# Keep this terminal running
```

**Option 2: Background Process**
```powershell
# PowerShell: Start in background
Start-Process python -ArgumentList "background_agents/start_agents.py" -WindowStyle Hidden
```

**Option 3: Task Scheduler**
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At startup
4. Action: Start a program
5. Program: `C:\Users\User\Desktop\TheMatrix\.venv\Scripts\python.exe`
6. Arguments: `background_agents/start_agents.py`
7. Start in: `C:\Users\User\Desktop\TheMatrix`

---

## 💡 Usage Examples

### Example 1: Process Specific Documents

```python
from background_agents.core import TaskQueue, Task, TaskPriority

queue = TaskQueue("background_agents/agents.db")

# Add document processing task
task = Task.create(
    agent_name="document_monitor",
    task_type="process_file",
    data={"file_path": "path/to/your/case.pdf"},
    priority=TaskPriority.HIGH
)

queue.add_task(task)
```

### Example 2: Request Research Analysis

```python
# Request case summary
task = Task.create(
    agent_name="legal_research",
    task_type="research",
    data={
        "research_type": "summary",
        "cases": [
            {"name": "Harvard v. Doe", "facts": "...", "outcome": "..."},
            # More cases...
        ]
    },
    priority=TaskPriority.MEDIUM
)

queue.add_task(task)
```

### Example 3: Run Settlement Analysis

```python
# Run Monte Carlo simulation
task = Task.create(
    agent_name="settlement_optimizer",
    task_type="optimize",
    data={
        "case_name": "harvard_discrimination",
        "success_probability": 0.65,
        "damages_mean": 1000000,
        "damages_std": 200000,
        "legal_costs": 75000,
        "risk_aversion": 0.5
    },
    priority=TaskPriority.HIGH
)

queue.add_task(task)
```

---

## 📈 Performance Expectations

### Your Gaming PC (32GB RAM)

| Agent | RAM Usage | CPU Usage | Runtime |
|-------|-----------|-----------|---------|
| Document Monitor | 2-4GB | 20-30% | Continuous |
| Legal Research | 4-6GB | 40-50% | 30min intervals |
| Citation Network | 8-10GB | 60-70% | 2hr intervals |
| Pattern Detection | 4-6GB | 40-50% | 4hr intervals |
| Settlement Optimizer | 1-2GB | 10-20% | 6hr intervals |

**Total Peak Usage:** ~16GB RAM, 40-50% CPU average

**Your Available:** 32GB RAM → **Perfect! Plenty of headroom**

### Processing Speed

- **Document Analysis**: ~30-60 seconds per document
- **Case Summary**: ~2-3 minutes per case
- **Citation Network**: ~10-15 minutes for 100 cases
- **Pattern Detection**: ~5-10 minutes for 50 cases
- **Settlement Simulation**: ~1-2 seconds (10K iterations)

### Daily Throughput

**24 hours of running:**
- ✅ 100-200 documents analyzed
- ✅ 50+ case summaries generated
- ✅ Complete citation network for corpus
- ✅ 6 pattern analysis runs
- ✅ 4 settlement optimization updates

---

## 🎯 Success Checklist

After first hour of running:

- [ ] System shows "running" status
- [ ] At least 5 tasks completed
- [ ] Output files created in `background_agents/outputs/`
- [ ] No errors in `background_agents/logs/system.log`
- [ ] RAM usage < 16GB
- [ ] CPU usage < 70% average

After first day:

- [ ] 50+ documents processed
- [ ] Multiple research summaries generated
- [ ] Citation network built
- [ ] Pattern insights available
- [ ] Settlement analyses complete

---

## 🚀 You're Ready!

Your gaming PC is now working for you 24/7:
- ✅ Zero cost (completely free)
- ✅ 100% private (all local)
- ✅ Always learning from your corpus
- ✅ Generating insights while you work

**Next:** Let it run for 24 hours, then check insights with:
```bash
python background_agents/view_insights.py
```

**Enjoy your automated legal AI assistant!** 🤖⚖️


