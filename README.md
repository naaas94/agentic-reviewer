# Agentic Reviewer

**Semantic Auditing for Text Classification Predictions**

A local-first LLM system that audits text classification predictions through semantic evaluation, alternative suggestions, and natural language explanations. Built with production-oriented patterns and designed for iterative enhancement.

---

## Project Status & Roadmap

> **Current Version:** v1.0 — Core Semantic Auditing  
> **In Development:** RAG Edition (Retrieval-Augmented Generation)

This project follows an incremental development philosophy. The table below provides transparency on what's implemented, what's in progress, and what's planned:

| Capability | Status | Notes |
|------------|--------|-------|
| Multi-Task Unified Agent | ✅ **Implemented** | Single LLM call for evaluation, proposal, and reasoning |
| FastAPI REST Interface | ✅ **Implemented** | Full API with health checks, metrics, review endpoints |
| Security Layer | ✅ **Implemented** | Prompt injection detection, input validation, rate limiting |
| LRU Caching + Circuit Breaker | ✅ **Implemented** | Production resilience patterns |
| Audit Logging (SQLite) | ✅ **Implemented** | Complete audit trail for compliance |
| System Monitoring | ✅ **Implemented** | Health checks, memory management, performance metrics |
| RAG: Vector Store | 🔲 **Planned** | FAISS/ChromaDB integration (next phase) |
| RAG: Embeddings Service | 🔲 **Planned** | sentence-transformers integration |
| RAG: Document Retrieval | 🔲 **Planned** | Policy document context injection |
| Portfolio Demo CLI | ✅ **Implemented** | Single-command demo with artifact generation |
| Synthetic Data Generator | ✅ **Implemented** | Realistic GDPR/CCPA request generation |
| LLM Report Generation | ✅ **Implemented** | Automated Markdown analysis reports |
| Evaluation Framework | 🔲 **Planned** | Quantitative accuracy metrics |

**Why this approach?** Building AI systems requires iterative validation. The core semantic auditing pipeline is functional and demonstrable today. RAG enhancement represents the next evolution—documented extensively in the implementation plan, with infrastructure designed to support it.

---

## 🚀 Portfolio Demo (Quick Validation)

**One command to validate the entire system:**

```bash
python run_portfolio_demo.py
```

This generates traceable artifacts demonstrating the complete pipeline:

```
outputs/runs/2024_12_04_153000/
├── 00_config.json              # Run configuration (reproducibility)
├── 01_synthetic_data.csv       # Generated GDPR/CCPA requests
├── 02_review_results.json      # Full LLM responses with metadata
├── 03_labeled_dataset.csv      # Corrected labels with reasoning
├── 04_report.md                # LLM-generated analysis report
└── 05_metrics.json             # Performance and accuracy stats
```

### Demo Options

```bash
# Full demo with LLM (requires Ollama)
python run_portfolio_demo.py --samples 20

# Dry run without Ollama (generates mock results)
python run_portfolio_demo.py --no-llm

# Reproducible run
python run_portfolio_demo.py --seed 42 --samples 15

# Windows PowerShell (without venv activation)
.\.venv\Scripts\python.exe run_portfolio_demo.py --no-llm
```

### Terminal Output

```
╔════════════════════════════════════════════════════════════════╗
║               AGENTIC REVIEWER — PORTFOLIO DEMO                ║
╠════════════════════════════════════════════════════════════════╣
║ Run ID: 2024_12_04_153000                                      ║
║ Model: mistral | Samples: 15 | Seed: 42                        ║
╠════════════════════════════════════════════════════════════════╣
║ PHASE 1: Synthetic Data Generation      ✓ 15 samples           ║
║ PHASE 2: LLM Review Loop                ✓ 15 reviews           ║
║ PHASE 3: Report Generation              ✓ 650 words            ║
║ PHASE 4: Artifact Generation            ✓ 5 files saved        ║
╠════════════════════════════════════════════════════════════════╣
║ RESULTS                                                        ║
║ ├─ Correct:    10 (66.7%)                                      ║
║ ├─ Incorrect:   4 (26.7%) → alternatives suggested             ║
║ └─ Uncertain:   1 (6.6%)                                       ║
╚════════════════════════════════════════════════════════════════╝
```

### What This Demonstrates

| Capability | Evidence |
|------------|----------|
| **Data Engineering** | Synthetic generation, schema validation, CSV/JSON I/O |
| **LLM Integration** | Ollama orchestration, prompt engineering, async processing |
| **MLOps Patterns** | Reproducibility (seeds), artifact management, metrics collection |
| **Observability** | Structured logging, performance timing, health checks |
| **Code Quality** | Type hints, error handling, modular architecture |

---

## What's Working Today

### Core Semantic Auditing Pipeline

The system evaluates whether a predicted classification label semantically fits the input text, using a three-phase agent approach:

```
Input Text + Predicted Label + Confidence
                ↓
    ┌───────────────────────────┐
    │     Unified Agent         │
    │  ┌─────────────────────┐  │
    │  │ 1. EVALUATE         │  │  → Correct / Incorrect / Uncertain
    │  │ 2. PROPOSE          │  │  → Alternative label if incorrect
    │  │ 3. REASON           │  │  → Natural language explanation
    │  └─────────────────────┘  │
    └───────────────────────────┘
                ↓
    Verdict + Reasoning + Explanation
```

### Domain: GDPR/CCPA Data Subject Requests

The system is configured for privacy request classification:
- **Access Request** — User asks to view their data
- **Erasure** — User requests data deletion (right to be forgotten)
- **Rectification** — User requests data correction
- **Portability** — User requests data export
- **Objection** — User objects to processing
- **Complaint** — Formal complaint about data handling
- **General Inquiry** — Questions about privacy policies

### Production-Oriented Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Unified Agent  │    │   LLM Service   │
│                 │    │                 │    │                 │
│ • FastAPI       │◄──►│ • Multi-Task    │◄──►│ • Ollama        │
│ • Validation    │    │ • Circuit       │    │ • Retry Logic   │
│ • Rate Limiting │    │   Breaker       │    │ • Caching       │
│ • Security      │    │ • Fallback      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Cache Layer    │    │  Monitoring     │
│                 │    │                 │    │                 │
│ • SQLite        │    │ • LRU Cache     │    │ • Health Checks │
│ • Sample        │    │ • Persistence   │    │ • Memory Mgmt   │
│   Selection     │    │ • TTL Support   │    │ • Logging       │
│ • Validation    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama with Mistral model
- 4GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/agentic-reviewer.git
cd agentic-reviewer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate

# Windows CMD:
.venv\Scripts\activate.bat

# Windows PowerShell (if execution policy allows):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> **Windows PowerShell Note:** If script execution is restricted, you can run commands directly using the virtual environment's Python executable without activation:
> ```powershell
> .\.venv\Scripts\python.exe -m pip install -r requirements.txt
> .\.venv\Scripts\python.exe demo.py
> ```

### LLM Setup

```bash
# Install Ollama (see https://ollama.ai/download for your platform)
# On Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral model
ollama pull mistral

# Start Ollama service (in a separate terminal)
ollama serve
```

### Run the Demo

```bash
# Comprehensive system demonstration
python demo.py

# Windows PowerShell (without venv activation):
.\.venv\Scripts\python.exe demo.py
```

The demo script validates the entire pipeline without requiring external services (beyond Ollama), showing:
- Data loading from `data/input.csv` (auto-generated if missing)
- Sample selection strategies (low-confidence, random, all)
- Agent initialization and label configuration
- Configuration status and cache operations
- (If Ollama running) Live LLM processing with sample reviews

### Start the API Server

```bash
python main.py
```

### Test an API Request

```bash
# Health check
curl http://localhost:8000/health

# Review a prediction
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Delete my data permanently",
    "predicted_label": "Access Request",
    "confidence": 0.85
  }'
```

**Expected Response:**
```json
{
  "sample_id": "api_1234567890",
  "verdict": "Incorrect",
  "reasoning": "The text is about deletion, not access",
  "suggested_label": "Erasure",
  "explanation": "This text clearly requests data deletion under the right to be forgotten",
  "success": true,
  "metadata": {
    "model_name": "mistral",
    "tokens_used": 150,
    "latency_ms": 200,
    "agent_type": "unified"
  }
}
```

---

## API Reference

### Endpoints (Implemented)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check with cache and system metrics |
| `GET` | `/metrics` | Detailed performance metrics |
| `GET` | `/stats` | Review statistics and verdict distribution |
| `POST` | `/review` | Review a single prediction |
| `GET` | `/reviews/{verdict}` | Get reviews filtered by verdict |
| `GET` | `/cache/stats` | Cache performance statistics |
| `POST` | `/cache/cleanup` | Clean expired cache entries |
| `GET` | `/security/stats` | Security validation statistics |
| `GET` | `/security/drift` | Drift detection results |

### Authentication

When `AR_API_KEY` is set, endpoints require Bearer token:

```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/stats
```

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
AR_MODEL_NAME=mistral           # Ollama model name
AR_OLLAMA_URL=http://localhost:11434
AR_TEMPERATURE=0.1              # Low for deterministic responses
AR_MAX_TOKENS=512
AR_TIMEOUT=30

# API Configuration
AR_API_HOST=0.0.0.0
AR_API_PORT=8000
AR_API_KEY=your-secure-api-key  # Optional, enables auth
AR_RATE_LIMIT_MAX=100

# Performance Configuration
AR_BATCH_SIZE=10
AR_MAX_CONCURRENT=5
AR_CACHE_MAX_SIZE_MB=100

# Security Configuration
AR_ENABLE_SANITIZATION=true
```

### Custom Labels

Edit `configs/labels.yaml` to adapt to your classification domain:

```yaml
labels:
  - name: "Access Request"
    definition: "User asks to view data held about them under GDPR/CCPA rights."
    examples:
      - "What information do you have about me?"
      - "Send me a copy of my data."
```

### Custom Prompts

Modify prompt templates in `prompts/` to adjust agent behavior:
- `evaluator_prompt.txt` — Verdict generation logic
- `proposer_prompt.txt` — Alternative label suggestion
- `reasoner_prompt.txt` — Explanation generation

---

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_agents.py -v

# Windows PowerShell (without venv activation):
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'pandas'` | Dependencies not installed | Run `pip install -r requirements.txt` in activated venv |
| `ModuleNotFoundError: No module named 'fastapi'` | Incomplete installation | Reinstall: `pip install fastapi uvicorn aiohttp jinja2 requests` |
| PowerShell: "execution of scripts is disabled" | Restricted execution policy | Use direct Python path: `.\.venv\Scripts\python.exe demo.py` |
| `.venv\Scripts\Activate.ps1` not found | Incomplete virtual environment | Recreate venv: `Remove-Item -Recurse .venv; python -m venv .venv` |
| Ollama connection refused | Ollama not running | Start Ollama service: `ollama serve` |
| Mistral model not found | Model not pulled | Pull model: `ollama pull mistral` |

### Windows PowerShell Setup

For Windows users encountering execution policy restrictions, the recommended workflow bypasses activation entirely:

```powershell
# Navigate to project directory
cd C:\path\to\agentic-reviewer

# Create virtual environment (one-time)
python -m venv .venv

# Install dependencies (one-time)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run demo
.\.venv\Scripts\python.exe demo.py

# Run API server
.\.venv\Scripts\python.exe main.py

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

This approach maintains isolation benefits without requiring script execution permissions.

---

## What's Coming: RAG Edition

The next evolution integrates **Retrieval-Augmented Generation** to enhance reasoning with external knowledge. This is actively planned, with detailed implementation specifications in:

- [`RAG_IMPLEMENTATION_PLAN.md`](RAG_IMPLEMENTATION_PLAN.md) — Comprehensive 18-24 week implementation roadmap
- [`RAG_ARCHITECTURE_EVOLUTION.md`](RAG_ARCHITECTURE_EVOLUTION.md) — Technical architecture decisions
- [`TECHNICAL_REVIEW_POC_ROADMAP.md`](TECHNICAL_REVIEW_POC_ROADMAP.md) — PoC staging and focus points

### RAG Edition Benefits (Planned)

| Enhancement | Impact |
|-------------|--------|
| **Policy Document Retrieval** | Ground reasoning in actual GDPR/CCPA text |
| **Reduced Hallucinations** | External knowledge constrains LLM responses |
| **Citation Support** | Reference specific document sections |
| **Audit Trail Enhancement** | Record which documents informed decisions |

### Planned RAG Stack

```bash
# Future dependencies (not yet required)
faiss-cpu==1.7.4              # Vector database
sentence-transformers==2.2.2  # Local embeddings
```

### Why RAG Isn't Implemented Yet

Building production RAG requires careful foundation work:

1. **Evaluation First** — Need baseline metrics before measuring RAG improvements
2. **Infrastructure Stability** — Core pipeline must be rock-solid
3. **Ablation Testing** — Each RAG component validated independently
4. **No Vendor Lock-in** — Local-first approach (FAISS, not cloud vector DBs)

The implementation plan documents this disciplined approach. RAG will be added when the foundation is ready—not before.

---

## Project Structure

```
agentic-reviewer/
├── agents/
│   ├── base_agent.py          # LLM abstraction with caching, circuit breaker
│   ├── unified_agent.py       # Multi-task processor (single LLM call)
│   ├── evaluator.py           # Verdict generation
│   ├── proposer.py            # Alternative label suggestion
│   └── reasoner.py            # Explanation generation
├── core/
│   ├── config.py              # Hierarchical configuration with validation
│   ├── cache.py               # LRU cache with TTL and persistence
│   ├── security.py            # Prompt injection & adversarial detection
│   ├── monitoring.py          # Health checks and metrics
│   ├── review_loop.py         # Batch and single-sample orchestration
│   ├── data_loader.py         # CSV data ingestion (configurable via data_path)
│   ├── sample_selector.py     # Low-confidence, random, all strategies
│   ├── synthetic_generator.py # Portfolio demo data generation
│   ├── report_generator.py    # LLM-powered Markdown reports
│   └── logger.py              # SQLite audit logging
├── configs/
│   └── labels.yaml            # Classification label definitions
├── prompts/
│   ├── evaluator_prompt.txt   # Jinja2 template for evaluation
│   ├── proposer_prompt.txt    # Jinja2 template for proposals
│   └── reasoner_prompt.txt    # Jinja2 template for reasoning
├── data/
│   └── input.csv              # Sample predictions for testing
├── tests/
│   ├── test_agents.py         # Agent unit tests
│   └── test_security.py       # Security validation tests
├── run_portfolio_demo.py      # Portfolio demo CLI (single entry point)
├── demo.py                    # Component demonstration script
├── main.py                    # FastAPI server
└── requirements.txt           # Python dependencies
```

---

## Design Principles

### Local-First
No cloud dependencies for core functionality. Ollama runs locally, SQLite stores data locally, caching is in-memory. This enables:
- **Privacy** — Data never leaves your machine
- **Cost Control** — No API usage fees
- **Reliability** — No network dependency for inference

### Production Patterns, PoC Scale
The architecture implements enterprise patterns (circuit breakers, caching, security layers) at a scale appropriate for demonstration and iteration. This showcases engineering maturity without over-engineering.

### Explainability by Design
Every classification audit produces:
1. **Verdict** — Binary/ternary correctness assessment
2. **Reasoning** — Technical justification
3. **Explanation** — Stakeholder-friendly narrative

These artifacts support regulatory compliance and human oversight requirements.

### Iterative Enhancement
The codebase is structured for staged improvement:
- Clear separation of concerns enables component replacement
- Configuration-driven behavior enables experimentation
- Modular agents enable A/B testing of approaches

---

## Contributing

```bash
# Development setup (Linux/macOS)
git clone https://github.com/your-org/agentic-reviewer.git
cd agentic-reviewer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests before committing
python -m pytest tests/ -v
```

```powershell
# Development setup (Windows PowerShell)
git clone https://github.com/your-org/agentic-reviewer.git
cd agentic-reviewer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run tests before committing
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Code Standards
- Follow PEP 8
- Use type hints
- Write docstrings for public methods
- Add tests for new features

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **LLM Runtime:** [Ollama](https://ollama.ai/)
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Inspired by:** Research on semantic auditing, LLM agents, and responsible AI practices

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| `README.md` | This file — project overview and quick start |
| `TECHNICAL_REVIEW_POC_ROADMAP.md` | Detailed technical review and PoC staging |
| `RAG_IMPLEMENTATION_PLAN.md` | 18-24 week RAG implementation specification |
| `RAG_ARCHITECTURE_EVOLUTION.md` | Technical architecture for RAG integration |
| `RAG_IMPLEMENTATION_CHECKLIST.md` | Task checklist for RAG development |

---

*This project demonstrates AI engineering practices including LLM integration, production resilience patterns, security awareness, and iterative development methodology. The documentation reflects the actual state of implementation—what works today and what's planned for tomorrow.*
