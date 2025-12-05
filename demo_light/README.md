# Agentic Reviewer Demo

**LLM-powered semantic auditing — no API keys, runs locally, free.**

```
┌─────────────────────────────────────────────────────────────┐
│  Input: "Delete my data permanently"                        │
│  Predicted: Access Request (85% confidence)                 │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  🤖 Local LLM Analysis (via Ollama)                  │   │
│  │  • Evaluate prediction accuracy                      │   │
│  │  • Suggest corrections                               │   │
│  │  • Generate explanations                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  Verdict: ❌ Incorrect                                      │
│  Suggested: Erasure                                         │
│  Reason: "Text requests deletion, not data access"         │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start (3 steps)

### 1. Install Ollama (free, local LLM runtime)

Download from **[ollama.ai/download](https://ollama.ai/download)** — works on Mac, Windows, Linux.

Then start it:
```bash
ollama serve
```

### 2. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/agentic-reviewer-demo.git
cd agentic-reviewer-demo
pip install -r requirements.txt
```

### 3. Run

```bash
python run_demo.py
```

**The script auto-downloads the LLM model on first run.** No API keys. No cloud. Everything local.

Check `outputs/` for generated artifacts.

---

## What It Does

1. **Generates** synthetic GDPR/CCPA data subject requests
2. **Reviews** each prediction using LLM semantic analysis
3. **Corrects** misclassifications with reasoning
4. **Produces** traceable artifacts (CSV, JSON, Markdown report)

---

## Output Artifacts

```
outputs/2024_12_04_153000/
├── 00_config.json          # Run configuration (reproducibility)
├── 01_synthetic_data.csv   # Generated classification samples
├── 02_review_results.json  # Full LLM responses
├── 03_labeled_dataset.csv  # Corrected labels with reasoning
├── 04_report.md            # Analysis report
└── 05_metrics.json         # Accuracy statistics
```

---

## Terminal Output

```
╔══════════════════════════════════════════════════════════════╗
║                    AGENTIC REVIEWER DEMO                     ║
╠══════════════════════════════════════════════════════════════╣
║ Run ID: 2024_12_04_153000                                    ║
║ Samples: 15 | Seed: 42 | LLM: On                             ║
╠══════════════════════════════════════════════════════════════╣
║ PHASE 1: Generate Synthetic Data       ✓ 15 samples          ║
║ PHASE 2: LLM Review                    ✓ 15/15 reviews       ║
║ PHASE 3: Generate Report               ✓ 450 words           ║
║ PHASE 4: Save Artifacts                ✓ 5 files             ║
╠══════════════════════════════════════════════════════════════╣
║ RESULTS                                                      ║
║ ├─ Correct:    10 (66.7%)                                   ║
║ ├─ Incorrect:   4 (26.7%) → corrections suggested           ║
║ └─ Uncertain:   1 (6.6%)                                    ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Demo Options

```bash
# Standard demo (12 samples)
python run_demo.py

# More samples
python run_demo.py --samples 25

# Reproducible run
python run_demo.py --seed 42

# Quick preview without LLM (for CI/testing)
python run_demo.py --mock
```

### Why Ollama?

| Feature | Ollama | Cloud APIs |
|---------|--------|------------|
| **Cost** | Free | Pay per token |
| **API Keys** | None needed | Required |
| **Privacy** | Data stays local | Sent to cloud |
| **Setup** | 1 download | Account signup |
| **Works offline** | Yes | No |

---

## What This Demonstrates

| Skill | Evidence |
|-------|----------|
| **LLM Integration** | Ollama orchestration, prompt engineering |
| **Data Engineering** | Synthetic generation, schema validation |
| **MLOps Patterns** | Reproducibility, artifact management |
| **Code Quality** | Clean architecture, type hints, error handling |

---

## Project Structure

```
agentic-reviewer-demo/
├── run_demo.py              # Single entry point
├── requirements.txt         # Minimal dependencies
├── core/
│   ├── synthetic_generator.py
│   ├── review_engine.py
│   └── report_generator.py
├── configs/
│   └── labels.yaml          # GDPR/CCPA label definitions
└── outputs/                 # Generated runs
```

---

## Production Version

This is a **minimal demo** for quick validation. For the full production implementation with:

- 🔒 Security layer (prompt injection detection)
- ⚡ LRU caching + circuit breaker
- 🌐 FastAPI REST interface
- 📊 System monitoring & health checks
- 📝 SQLite audit logging

See: **[agentic-reviewer](https://github.com/YOUR_USERNAME/agentic-reviewer)**

---

## License

MIT

---

*Built to demonstrate LLM-powered classification auditing.*

