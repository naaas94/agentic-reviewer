# Agentic Reviewer

**Semantic Auditing for Text Classification Predictions**

A production-ready system that uses LLM agents to audit and improve text classification predictions through semantic evaluation, alternative suggestions, and natural language explanations.

**Status:** Active development with security features and ground truth validation in progress.

---

## System Overview

The Agentic Reviewer addresses the critical need for explainable, auditable text classification systems. Rather than treating classification as a black-box prediction, this system decomposes the reasoning process into specialized agents that provide transparent, reference-backed analysis of classification decisions.

### Core Architecture

The system implements a unified agent approach that processes evaluation, proposal, and reasoning in a single coordinated call, reducing latency while maintaining consistency across all tasks. This design prioritizes explainability, auditability, and regulatory compliance.

### Key Capabilities

**Multi-Task Unified Agent**
- Single LLM call processing for evaluation, proposal, and reasoning
- 3x reduction in latency and token usage compared to individual agents
- Coordinated decision-making across all tasks with consistent reasoning

**Enterprise Security**
- Input sanitization against injection attacks
- API key authentication with secure access control
- Rate limiting to prevent abuse and ensure fair usage
- CORS protection with configurable cross-origin policies
- SSL/TLS support for production HTTPS encryption

**High Performance**
- Advanced LRU caching with persistence and memory limits
- Circuit breaker pattern for automatic failure detection and recovery
- Concurrent processing with configurable batch limits
- Memory management with automatic cache eviction and cleanup

**Production Monitoring**
- Comprehensive health checks for system status monitoring
- Real-time performance and usage statistics dashboard
- Complete audit trail for compliance and debugging
- Cache analytics for memory usage and hit rate monitoring

---

## Architecture

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
│ • Validation    │    │ • Alerts        │    │ • Alerts        │
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
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### LLM Setup

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral model
ollama pull mistral

# Start Ollama service
ollama serve
```

### System Execution

```bash
# Development mode
python main.py

# Or use the deployment script
python deploy.py --mode dev
```

### API Testing

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

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
AR_MODEL_NAME=mistral
AR_OLLAMA_URL=http://localhost:11434
AR_TEMPERATURE=0.1
AR_MAX_TOKENS=512
AR_TIMEOUT=30

# API Configuration
AR_API_HOST=0.0.0.0
AR_API_PORT=8000
AR_API_KEY=your-secure-api-key
AR_RATE_LIMIT_MAX=1000

# Performance Configuration
AR_BATCH_SIZE=10
AR_MAX_CONCURRENT=10
AR_CACHE_MAX_SIZE_MB=200

# Security Configuration
AR_ENABLE_SANITIZATION=true
```

### Configuration Management

The system uses a hierarchical configuration system with validation:

```python
from core.config import config

# Access configuration
print(config.llm.model_name)  # mistral
print(config.api.port)        # 8000
print(config.performance.batch_size)  # 10
```

---

## Deployment

### Docker Deployment

```bash
# Generate deployment files
python deploy.py --mode docker --ssl

# Set API key
export AR_API_KEY="your-secure-api-key"

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### Production Deployment

```bash
# Production deployment with SSL
python deploy.py --mode prod --ssl

# Follow the printed instructions to:
# 1. Update service file paths
# 2. Install systemd service
# 3. Start the service
```

### Manual Production Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate SSL certificates
openssl genrsa -out ssl/key.pem 2048
openssl req -new -x509 -key ssl/key.pem -out ssl/cert.pem -days 365

# 3. Set environment variables
export AR_API_KEY="your-secure-api-key"
export AR_LOG_LEVEL=INFO

# 4. Start with SSL
python main.py
```

---

## API Reference

### Authentication

All API endpoints require Bearer token authentication:

```bash
curl -H "Authorization: Bearer your-api-key" \
  http://localhost:8000/api/endpoint
```

### Endpoints

#### `POST /review`

Review a single prediction:

```json
{
  "text": "Delete my data permanently",
  "predicted_label": "Access Request", 
  "confidence": 0.85,
  "use_unified_agent": true
}
```

Response:

```json
{
  "sample_id": "api_1234567890",
  "verdict": "Incorrect",
  "reasoning": "The text is about deletion, not access",
  "suggested_label": "Erasure",
  "explanation": "This text clearly requests data deletion",
  "success": true,
  "metadata": {
    "model_name": "mistral",
    "tokens_used": 150,
    "latency_ms": 200,
    "agent_type": "unified"
  }
}
```

#### `GET /health`

System health check:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime": 3600.5,
  "system_health": {
    "cpu_usage": 15.2,
    "memory_usage": 45.8,
    "disk_usage": 23.1
  },
  "cache_health": {
    "status": "healthy",
    "entries": 1250,
    "memory_usage_mb": 45.2,
    "utilization_percent": 45.2
  }
}
```

#### `GET /metrics`

System metrics:

```json
{
  "system": {
    "cpu_usage": 15.2,
    "memory_usage": 45.8,
    "disk_usage": 23.1
  },
  "application": {
    "total_requests": 1250,
    "avg_response_time": 245.6,
    "error_rate": 0.02
  },
  "cache": {
    "entries": 1250,
    "hit_rate": 0.85,
    "memory_usage_mb": 45.2
  }
}
```

#### `GET /stats`

Review statistics:

```json
{
  "total_reviews": 1250,
  "verdict_distribution": {
    "Correct": 850,
    "Incorrect": 320,
    "Uncertain": 80
  },
  "avg_confidence": 0.72,
  "recent_reviews_24h": 45,
  "cache_stats": {
    "entries": 1250,
    "hit_rate": 0.85,
    "memory_usage_mb": 45.2
  }
}
```

---

## Testing

### Test Execution

```bash
# Run test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_agents.py -v
```

### Component Testing

```bash
# Test unified agent
python -c "
from agents.unified_agent import UnifiedAgent
agent = UnifiedAgent()
result = agent.process_sample_sync('Test text', 'Test label', 0.8)
print(result)
"

# Test API endpoint
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "predicted_label": "test", "confidence": 0.8}'
```

---

## Monitoring and Logging

### Health Monitoring

```bash
# Check system health
curl http://localhost:8000/health

# Get detailed metrics
curl http://localhost:8000/metrics

# Monitor cache performance
curl http://localhost:8000/cache/stats
```

### Log Files

- **Application logs**: `logs/app.log`
- **Audit logs**: `outputs/audit.log`
- **Error logs**: `logs/error.log`

### Cache Management

```bash
# Clean up expired cache entries
curl -X POST http://localhost:8000/cache/cleanup \
  -H "Authorization: Bearer your-api-key"

# Get cache statistics
curl http://localhost:8000/cache/stats \
  -H "Authorization: Bearer your-api-key"
```

---

## Advanced Configuration

### Custom Labels

Edit `configs/labels.yaml` to define your classification labels:

```yaml
labels:
  - name: "Access Request"
    description: "Requests to access personal data"
    examples:
      - "I want to see my data"
      - "Show me what you have about me"
  
  - name: "Erasure"
    description: "Requests to delete personal data"
    examples:
      - "Delete my data"
      - "Remove my information"
```

### Custom Prompts

Modify prompt templates in `prompts/`:
- `evaluator_prompt.txt`: Evaluation logic
- `proposer_prompt.txt`: Alternative suggestions
- `reasoner_prompt.txt`: Explanations

### Performance Tuning

```python
# Adjust cache settings
config.performance.cache_max_size_mb = 500
config.performance.cache_max_entries = 20000

# Adjust concurrency
config.performance.max_concurrent_requests = 20
config.performance.batch_size = 25

# Adjust LLM settings
config.llm.temperature = 0.05  # More deterministic
config.llm.max_tokens = 1024   # Longer responses
```

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/agentic-reviewer.git
cd agentic-reviewer

# Create development environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/ -v
```

### Code Standards

- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests for new features

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- LLM integration via [Ollama](https://ollama.ai/)
- Inspired by research on semantic auditing and LLM agents

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/agentic-reviewer/issues)
- **Documentation**: [Wiki](https://github.com/your-org/agentic-reviewer/wiki)
- **Email**: support@your-org.com

 