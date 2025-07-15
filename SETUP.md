# Agentic Reviewer Setup Guide

This guide provides comprehensive setup instructions for the Agentic Reviewer system, designed for semantic auditing of text classification predictions. The system addresses the critical need for explainable, auditable classification systems through specialized LLM agents.

---

## Prerequisites

### Python Environment
- Python 3.10 or higher
- pip package manager

### Local LLM Setup

The system requires a local LLM service for inference. Several options are supported:

#### Option A: Ollama (Recommended)
1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull a model:
   ```bash
   # For Mistral (good balance of speed/quality)
   ollama pull mistral
   
   # For Zephyr (faster, smaller)
   ollama pull zephyr
   
   # For Llama3 (larger, higher quality)
   ollama pull llama3
   ```

#### Option B: LM Studio
1. Download LM Studio from [https://lmstudio.ai](https://lmstudio.ai)
2. Download a model (Mistral, Llama, etc.)
3. Start the local server in LM Studio

#### Option C: Other Local LLM Servers
The system can work with any local LLM server that provides an Ollama-compatible API.

---

## Installation

### Repository Setup
```bash
git clone <repository-url>
cd agentic-reviewer
```

### Dependency Installation
```bash
pip install -r requirements.txt
```

### Verification
```bash
# Run the demo to verify everything works
python demo.py
```

---

## Quick Start

### Basic Review Execution
```bash
# Review samples with low confidence (< 0.7)
python run.py --strategy low_confidence --threshold 0.7

# Review random samples
python run.py --strategy random --sample-size 5

# Review all samples
python run.py --strategy all
```

### API Server
```bash
# Start the FastAPI server
python main.py

# The API will be available at http://localhost:8000
# Visit http://localhost:8000/docs for interactive documentation
```

### API Testing
```bash
curl -X POST "http://localhost:8000/review" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Delete my data permanently",
       "predicted_label": "Access Request",
       "confidence": 0.85
     }'
```

---

## Configuration

### Label Definitions

Edit `configs/labels.yaml` to define your label ontology:

```yaml
labels:
  - name: "Your Label"
    definition: "Clear definition of what this label means"
    examples:
      - "Example text 1"
      - "Example text 2"
```

### Model Configuration

The system supports different models. Change the model in your scripts:

```python
# In your code
review_loop = ReviewLoop(model_name="zephyr")  # or "llama3", "mistral"
```

### Sample Data

Place your classification data in `data/input.csv` with columns:
- `text`: Input text
- `pred_label`: Predicted label
- `confidence`: Confidence score (0.0 to 1.0)
- `id`: Optional sample ID

---

## Usage Examples

### Review Specific Samples
```python
from core.review_loop import ReviewLoop

review_loop = ReviewLoop()
result = review_loop.review_single_sample(
    text="Delete my data permanently",
    predicted_label="Access Request", 
    confidence=0.85
)
print(result)
```

### Custom Sample Selection
```python
from core.sample_selector import SampleSelector
import pandas as pd

df = pd.read_csv("data/input.csv")
selector = SampleSelector("low_confidence", threshold=0.6)
selected = selector.select_samples(df, max_samples=10)
```

### Access Review Results
```python
from core.logger import AuditLogger

logger = AuditLogger()
stats = logger.get_review_stats()
print(f"Total reviews: {stats['total_reviews']}")
print(f"Verdict distribution: {stats['verdict_distribution']}")
```

---

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull the model you need
ollama pull mistral
```

### Python Import Errors
```bash
# Make sure you're in the correct directory
cd agentic-reviewer

# Install dependencies
pip install -r requirements.txt
```

### Database Issues
```bash
# Remove existing database to start fresh
rm outputs/reviewed_predictions.sqlite
```

---

## Advanced Configuration

### Custom Prompts

Edit the prompt templates in the `prompts/` directory:
- `evaluator_prompt.txt`: For evaluating predictions
- `proposer_prompt.txt`: For suggesting alternatives
- `reasoner_prompt.txt`: For generating explanations

### Database Configuration

The system uses SQLite by default. You can modify the database path:

```python
logger = AuditLogger("path/to/your/database.sqlite")
```

### LLM Parameters

Adjust LLM parameters in `agents/base_agent.py`:

```python
# Temperature (creativity vs consistency)
"temperature": 0.1  # Lower = more consistent

# Max tokens per response
"num_predict": 512  # Adjust based on your needs
```

---

## Performance Optimization

### Model Selection
- **Mistral**: Good balance of speed and quality
- **Zephyr**: Faster, good for high-throughput scenarios
- **Llama3**: Higher quality, slower inference

### Batch Processing

For large datasets, process in batches:

```python
# Process in chunks
for chunk in pd.read_csv("data/large_dataset.csv", chunksize=100):
    # Process chunk
    pass
```

### Caching

The system automatically caches prompt hashes for versioning. Consider implementing additional caching for repeated queries.

---

## Monitoring and Logging

### Review Statistics
```python
from core.logger import AuditLogger

logger = AuditLogger()
stats = logger.get_review_stats()
print(stats)
```

### Individual Reviews
```python
# Get a specific review
review = logger.get_review("sample_001")
print(review)

# Get reviews by verdict
incorrect_reviews = logger.get_reviews_by_verdict("Incorrect")
print(f"Found {len(incorrect_reviews)} incorrect predictions")
```

### Run Metadata

Each review run is logged with metadata including:
- Model used
- Prompt versions
- Selection strategy
- Performance metrics

---

## Contributing

### Adding New Agents
1. Create a new agent class in `agents/`
2. Inherit from `BaseAgent`
3. Implement the required methods
4. Add tests in `tests/`

### Adding New Selection Strategies
1. Add strategy logic to `SampleSelector`
2. Update the `select_samples` method
3. Add tests for the new strategy

### Customizing Prompts
1. Edit prompt templates in `prompts/`
2. Test with different examples
3. Update version numbers in metadata

---

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the demo script (`python demo.py`)
3. Check the API documentation (`http://localhost:8000/docs`)
4. Examine the test files for usage examples

---

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
noteId: "09001d305fe411f0ad7e939cd7bead99"
tags: []

---

 