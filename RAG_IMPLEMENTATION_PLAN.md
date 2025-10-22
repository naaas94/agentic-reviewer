# RAG Edition Implementation Plan
## Production-Grade RAG Enhancement for Agentic Reviewer

**Document Version:** 1.0  
**Created:** October 21, 2025  
**Status:** Planning Phase  
**Estimated Timeline:** 18-24 weeks

---

## Executive Summary

This implementation plan addresses critical architectural gaps identified in the planned RAG enhancement for the Agentic Reviewer system. Rather than jumping directly into RAG features, this plan prioritizes **evaluation infrastructure, operational rigor, and systematic validation** as prerequisites for production-ready retrieval-augmented generation.

**Key Principle:** Build resilient retrieval infrastructure, establish baselines, validate systematically—BEFORE adding advanced features.

---

## Table of Contents

1. [Critical Mindset Shift](#critical-mindset-shift)
2. [Pre-Implementation Checklist](#pre-implementation-checklist)
3. [Phase 0: Foundation & Stabilization](#phase-0-foundation--stabilization-4-6-weeks)
4. [Phase 1: Baseline RAG Infrastructure](#phase-1-baseline-rag-infrastructure-2-3-weeks)
5. [Phase 2: Systematic Ablation & Enhancement](#phase-2-systematic-ablation--enhancement-8-10-weeks)
6. [Phase 3: Advanced Techniques Validation](#phase-3-advanced-techniques-validation-4-6-weeks)
7. [Phase 4: Production Operations](#phase-4-production-operations-ongoing)
8. [Success Criteria & Metrics](#success-criteria--metrics)
9. [Risk Mitigation](#risk-mitigation)
10. [Decision Gates](#decision-gates)

---

## Critical Mindset Shift

### FROM: Feature-First Approach
- ❌ "Let's add Graph RAG, PageIndex, and CAG"
- ❌ "RAG will improve accuracy"
- ❌ "Follow vendor benchmarks"
- ❌ "Build sophisticated features first"

### TO: Engineering-First Approach
- ✅ "Build resilient infrastructure with fallbacks"
- ✅ "Measure baseline, then measure improvements"
- ✅ "Validate on OUR data, not vendor data"
- ✅ "Prove value at each step before adding complexity"

---

## Pre-Implementation Checklist

**⚠️ DO NOT PROCEED TO RAG IMPLEMENTATION UNTIL ALL ITEMS ARE COMPLETE**

### System Health Requirements
- [ ] 100% test pass rate (currently ~77%)
- [ ] Ollama integration stable and reliable
- [ ] Security validation has <5% false positive rate
- [ ] Circuit breaker functioning correctly
- [ ] Cache performance within acceptable limits
- [ ] All core system APIs responding correctly

### Documentation Requirements
- [ ] Current system architecture documented
- [ ] Baseline performance metrics captured (pre-RAG)
- [ ] All configuration options documented
- [ ] Security model fully documented

### Team Readiness
- [ ] Team understands RAG evaluation metrics
- [ ] Team trained on ablation testing methodology
- [ ] Commitment to measurement-first approach
- [ ] Buy-in on NOT pursuing unvalidated techniques

---

## Phase 0: Foundation & Stabilization (4-6 weeks)

**Goal:** Achieve 100% system reliability and establish evaluation infrastructure BEFORE touching RAG.

**Status Gate:** This phase BLOCKS all RAG work. No exceptions.

---

### Stage 0.1: Core System Stabilization (2 weeks)

#### Objectives
1. Fix all failing tests
2. Stabilize Ollama integration
3. Resolve security validation false positives
4. Achieve production-ready reliability

#### Tasks

##### Week 1: Test Failures & Ollama
- [ ] **Task 0.1.1:** Run full test suite and document all failures
  - Tools: `pytest tests/ -v --tb=long`
  - Deliverable: `test_failure_analysis.md`
  
- [ ] **Task 0.1.2:** Fix Ollama connection reliability issues
  - Implement retry logic with exponential backoff
  - Add connection pooling
  - Add health check endpoint for Ollama
  - Update: `agents/base_agent.py`
  
- [ ] **Task 0.1.3:** Fix all failing agent tests
  - File: `tests/test_agents.py`
  - Target: 100% pass rate
  
- [ ] **Task 0.1.4:** Validate circuit breaker functionality
  - Add integration tests for circuit breaker
  - Test failure scenarios and recovery

##### Week 2: Security & Integration
- [ ] **Task 0.1.5:** Analyze security validation false positives
  - Review `core/security.py` validation rules
  - Adjust thresholds for production use
  - Target: <5% false positive rate
  
- [ ] **Task 0.1.6:** Integration testing
  - End-to-end API tests
  - Load testing (concurrent requests)
  - Cache stress testing
  
- [ ] **Task 0.1.7:** Achieve 100% test pass rate
  - All unit tests passing
  - All integration tests passing
  - All security tests passing

#### Success Criteria
- ✅ 100% test pass rate
- ✅ Ollama integration has <1% failure rate under normal load
- ✅ Security validation false positive rate <5%
- ✅ System handles 100 concurrent requests without errors
- ✅ Circuit breaker triggers and recovers correctly

#### Deliverables
- `test_failure_analysis.md`
- Updated test suite with 100% pass rate
- Ollama connection improvements
- Security validation tuning documentation

---

### Stage 0.2: Evaluation Framework Foundation (2 weeks)

#### Objectives
1. Choose and define north star metric
2. Create human-labeled audit set
3. Implement baseline evaluation (pre-RAG)
4. Define SLOs for latency, accuracy, cost

#### Tasks

##### Week 3: Metrics & Audit Set
- [ ] **Task 0.2.1:** Select North Star Metric
  - **Recommendation:** Answer Faithfulness (best for auditability use case)
  - Alternative considerations: Retrieval nDCG, Answer Relevance
  - Document decision rationale
  - File: Create `docs/metrics_selection.md`
  
- [ ] **Task 0.2.2:** Create human-labeled audit set
  - Select 100-500 representative samples from production data
  - Manual labeling by domain experts
  - Include edge cases, ambiguous samples, and clear examples
  - Format: JSON with ground truth labels and explanations
  - File: Create `data/audit_set_v1.json`
  - Schema:
    ```json
    {
      "sample_id": "string",
      "text": "string",
      "predicted_label": "string",
      "ground_truth_label": "string",
      "correct_verdict": "Correct|Incorrect|Uncertain",
      "explanation": "string",
      "annotator_confidence": "float",
      "edge_case": "boolean"
    }
    ```
  
- [ ] **Task 0.2.3:** Design RAG evaluation metrics suite
  - Primary: Answer Faithfulness (north star)
  - Secondary metrics:
    - Context Precision
    - Context Recall
    - Answer Relevance
    - Citation Accuracy
    - Cost per Answer
    - Latency p50/p95/p99
  - File: Create `core/evaluation/rag_metrics.py`

##### Week 4: Baseline Measurement & SLOs
- [ ] **Task 0.2.4:** Implement evaluation framework
  - Create `RAGEvaluationFramework` class
  - Implement all metric calculations
  - LLM-as-judge calibration against human audit set
  - Target correlation: >0.85 with human labels
  - File: `core/evaluation/rag_metrics.py`
  
- [ ] **Task 0.2.5:** Measure PRE-RAG baseline performance
  - Run current system on audit set
  - Record accuracy WITHOUT retrieval
  - This is our comparison baseline for RAG improvements
  - Deliverable: `baselines/pre_rag_baseline.json`
  
- [ ] **Task 0.2.6:** Define Service Level Objectives (SLOs)
  - **Latency SLOs:**
    - End-to-end p95: < 2000ms
    - LLM generation p95: < 1500ms
    - (Future) Retrieval p95: < 200ms
    - (Future) Reranking p95: < 100ms
  - **Accuracy SLOs:**
    - Answer Faithfulness: > 0.85
    - Context Precision: > 0.80
    - Citation Accuracy: > 0.90
  - **Cost SLOs:**
    - Cost per answer: < $0.01
    - (Future) Embedding cost per query: < $0.001
    - (Future) Vector DB cost per query: < $0.0005
  - **Availability SLOs:**
    - System uptime: > 99.5%
    - (Future) Retrieval service uptime: > 99.5%
  - File: Create `docs/slos.md`

#### Success Criteria
- ✅ North star metric chosen and documented
- ✅ Human audit set created (100-500 samples)
- ✅ Evaluation framework implemented and tested
- ✅ LLM-as-judge calibration >0.85 correlation with humans
- ✅ Pre-RAG baseline measured and documented
- ✅ SLOs defined for all key metrics

#### Deliverables
- `docs/metrics_selection.md`
- `data/audit_set_v1.json`
- `core/evaluation/rag_metrics.py`
- `baselines/pre_rag_baseline.json`
- `docs/slos.md`
- Evaluation dashboard (basic version)

---

### Stage 0.3: Operational Readiness (2 weeks)

#### Objectives
1. Implement token budgeting and cost tracking
2. Add comprehensive observability
3. Build A/B testing framework
4. Create retrieval debugging tools
5. Set up CI/CD for model evaluation

#### Tasks

##### Week 5: Token Tracking & Observability
- [ ] **Task 0.3.1:** Implement token budgeting
  - Track prompt tokens per request
  - Track completion tokens per request
  - Track total cost per request
  - Add alerts for budget overruns
  - File: Update `core/monitoring.py`
  
- [ ] **Task 0.3.2:** Enhanced observability infrastructure
  - Add detailed latency tracking (by component)
  - Add error tracking with categorization
  - Add performance metrics dashboards
  - Track cache hit rates (query cache, future: embedding cache)
  - File: Update `core/monitoring.py`
  
- [ ] **Task 0.3.3:** Create operational dashboards
  - System health dashboard (extend existing)
  - Token usage and cost dashboard
  - Latency distribution dashboard
  - Error rate dashboard
  - Tool: Can use built-in FastAPI or external (Grafana, etc.)

##### Week 6: A/B Testing & CI/CD
- [ ] **Task 0.3.4:** Build A/B testing framework
  - Traffic splitting functionality (95/5, 90/10, etc.)
  - Experiment configuration management
  - Success metric tracking
  - Guardrail metric tracking
  - Automatic rollback on guardrail violations
  - File: Create `core/operations/ab_testing.py`
  
- [ ] **Task 0.3.5:** Set up CI/CD for evaluation
  - Automated evaluation on audit set for each change
  - Regression detection (alert if metrics drop >2%)
  - Automated comparison reports
  - Gate deployments on evaluation results
  - File: Create `.github/workflows/evaluation_pipeline.yml` (or equivalent)
  
- [ ] **Task 0.3.6:** Create debugging tools
  - Request/response inspection tool
  - Performance profiling tool
  - (Future) Retrieval debugging: "why was this retrieved?"
  - (Future) Relevance score visualization
  - File: Create `tools/debug_inspector.py`

#### Success Criteria
- ✅ Token tracking operational with alerts
- ✅ Observability dashboards displaying real-time metrics
- ✅ A/B testing framework tested with dummy experiments
- ✅ CI/CD pipeline runs evaluation on every PR
- ✅ Debugging tools functional and documented

#### Deliverables
- Enhanced `core/monitoring.py`
- `core/operations/ab_testing.py`
- `.github/workflows/evaluation_pipeline.yml`
- `tools/debug_inspector.py`
- Operational runbooks for monitoring and debugging

---

### Phase 0 Exit Criteria (BLOCKING)

**⚠️ ALL CRITERIA MUST BE MET BEFORE PROCEEDING TO PHASE 1**

- [ ] ✅ System achieves 100% test pass rate
- [ ] ✅ Ollama integration reliable (<1% failure rate)
- [ ] ✅ Security validation tuned (<5% false positives)
- [ ] ✅ North star metric selected and documented
- [ ] ✅ Human audit set created (100-500 samples)
- [ ] ✅ Evaluation framework implemented and calibrated (>0.85 correlation)
- [ ] ✅ Pre-RAG baseline measured on audit set
- [ ] ✅ SLOs defined and documented
- [ ] ✅ Token tracking and cost monitoring operational
- [ ] ✅ Observability dashboards deployed
- [ ] ✅ A/B testing framework ready
- [ ] ✅ CI/CD evaluation pipeline operational

**Decision Point:** Phase 0 Review Meeting
- Review all deliverables
- Validate baseline measurements
- Confirm team readiness
- Sign-off required from: Engineering Lead, Product Owner, QA Lead

---

## Phase 1: Baseline RAG Infrastructure (2-3 weeks)

**Goal:** Implement DENSE-ONLY retrieval with production resilience. No fancy features. This is the foundation for all future enhancements.

**Critical Rule:** FREEZE prompt template v1.0 and model for entire ablation process (through end of Phase 2).

---

### Stage 1.1: Vector Database Setup (Week 7)

#### Objectives
1. Set up vector database infrastructure
2. Implement embedding generation
3. Create document ingestion pipeline
4. Build retrieval service with resilience

#### Tasks

- [ ] **Task 1.1.1:** Select and deploy vector database
  - **Options:** FAISS (simplest), ChromaDB (good balance), Pinecone (managed)
  - **Recommendation:** Start with FAISS for simplicity
  - Deploy with proper configuration
  - Implement connection pooling
  - Add health check endpoint
  - File: Create `core/rag/vector_store.py`

- [ ] **Task 1.1.2:** Implement embedding generation
  - **Model Selection:** Evaluate BGE, MiniLM, domain-specific embeddings
  - Don't default to generic embeddings without evaluation
  - Test embedding models on sample data
  - Document model selection rationale
  - Implement embedding service with caching
  - File: Create `core/rag/embedding_service.py`

- [ ] **Task 1.1.3:** Build document ingestion pipeline
  - Document loader for policy documents
  - Simple chunking strategy (start with fixed-size: 512 tokens, 128 overlap)
  - Metadata extraction (document name, date)
  - Batch processing for efficiency
  - File: Create `core/rag/document_processor.py`

- [ ] **Task 1.1.4:** Implement retrieval service with resilience
  - Basic dense retrieval (top-k)
  - Circuit breaker for vector DB calls
  - Fallback strategy: if retrieval fails → use base LLM (no RAG)
  - Retry logic with exponential backoff
  - Connection pooling
  - File: Create `core/rag/retrieval_service.py`

#### Success Criteria
- ✅ Vector database deployed and accessible
- ✅ Embedding model selected and justified
- ✅ Documents ingested and indexed successfully
- ✅ Retrieval service returns relevant chunks
- ✅ Circuit breaker and fallback functional
- ✅ Health checks passing

---

### Stage 1.2: RAG Integration (Week 8)

#### Objectives
1. Integrate retrieval with unified agent
2. Implement context injection in prompts
3. Add RAG-specific observability
4. Implement graceful degradation

#### Tasks

- [ ] **Task 1.2.1:** Update unified agent for RAG
  - Add retrieval step before LLM call
  - Implement context injection in prompt
  - Maintain backward compatibility (RAG optional)
  - File: Update `agents/unified_agent.py`

- [ ] **Task 1.2.2:** Prompt template updates
  - Create RAG-enabled prompt template v1.0
  - FREEZE this template for ablation testing
  - Include retrieved context formatting
  - Include citation instructions
  - Files: Update `prompts/*.txt`

- [ ] **Task 1.2.3:** RAG-specific observability
  - Track retrieval latency (p50/p95/p99)
  - Track embedding generation time
  - Track vector DB query time
  - Track number of chunks retrieved
  - Track retrieval failures
  - Track fallback activations
  - File: Update `core/monitoring.py`

- [ ] **Task 1.2.4:** Implement graceful degradation
  - If vector DB fails → fallback to base LLM
  - If embedding service fails → fallback to base LLM
  - If retrieval times out → fallback to base LLM
  - Log all degradation events
  - Alert on high degradation rates
  - File: Update `core/rag/retrieval_service.py`

#### Success Criteria
- ✅ Unified agent successfully uses retrieved context
- ✅ Prompts properly inject retrieved chunks
- ✅ RAG-specific metrics tracked and displayed
- ✅ Graceful degradation works (test by killing vector DB)
- ✅ System remains available during retrieval failures

---

### Stage 1.3: Baseline RAG Evaluation (Week 9)

#### Objectives
1. Measure dense-only RAG performance
2. Document baseline RAG metrics
3. Compare to pre-RAG baseline
4. Identify performance gaps

#### Tasks

- [ ] **Task 1.3.1:** Run evaluation on audit set
  - Use frozen prompt template v1.0
  - Use frozen model and temperature
  - Measure all metrics from evaluation framework
  - Run on full audit set (100-500 samples)
  - File: Create `evaluation_runs/baseline_rag_dense_only.json`

- [ ] **Task 1.3.2:** Calculate metrics
  - Answer Faithfulness (north star)
  - Context Precision
  - Context Recall
  - Answer Relevance
  - Cost per answer
  - Latency p50/p95/p99
  - Cache hit rates

- [ ] **Task 1.3.3:** Compare to pre-RAG baseline
  - Calculate delta for each metric
  - Statistical significance testing
  - Identify where RAG helps vs hurts
  - Document findings
  - File: Create `analysis/rag_vs_no_rag_comparison.md`

- [ ] **Task 1.3.4:** Regression testing
  - Ensure RAG doesn't hurt baseline performance on simple cases
  - Identify any failure modes
  - Document edge cases
  - Create regression test suite

#### Success Criteria
- ✅ Baseline RAG metrics documented
- ✅ Comparison to pre-RAG shows positive delta on north star metric
- ✅ No regressions on baseline performance
- ✅ Statistical significance of improvements validated
- ✅ Edge cases and failure modes documented

#### Deliverables
- `evaluation_runs/baseline_rag_dense_only.json`
- `analysis/rag_vs_no_rag_comparison.md`
- Regression test suite
- Baseline RAG dashboard

---

### Phase 1 Exit Criteria

- [ ] ✅ Dense-only retrieval operational and resilient
- [ ] ✅ Graceful degradation tested and working
- [ ] ✅ RAG-specific observability in place
- [ ] ✅ Baseline RAG metrics measured on audit set
- [ ] ✅ Positive delta vs pre-RAG baseline on north star metric
- [ ] ✅ No regressions on simple cases
- [ ] ✅ Prompt template v1.0 frozen for ablation testing

**Decision Point:** Phase 1 Review
- Validate RAG provides measurable value
- Review operational metrics (latency, cost)
- Approve proceeding to systematic enhancement

---

## Phase 2: Systematic Ablation & Enhancement (8-10 weeks)

**Goal:** Add proven RAG components ONE AT A TIME, measuring delta at each step. Only proceed if previous component meets lift expectations.

**Critical Rule:** Keep prompt template and model FROZEN. Test components individually. Report DELTAS, not absolute numbers.

---

### Ablation Protocol

**For Each Component Addition:**
1. Add ONLY the new component (no other changes)
2. Run evaluation on same frozen audit set
3. Measure ALL metrics
4. Calculate delta vs previous stage
5. Check for regressions on baseline cases
6. Document results
7. Decision gate: meets lift expectations? If YES → proceed. If NO → iterate or skip.

---

### Stage 2.1: Hybrid Retrieval (Weeks 10-11)

#### Component: BM25 + Dense Hybrid Search

**Expected Lift:** +5-15% Answer Faithfulness

#### Tasks

- [ ] **Task 2.1.1:** Implement BM25 index
  - Build BM25 index over same documents
  - Use library like `rank-bm25` or Elasticsearch
  - File: Update `core/rag/retrieval_service.py`

- [ ] **Task 2.1.2:** Implement hybrid fusion
  - Retrieve top-100 from both BM25 and dense
  - Weighted fusion (start with 0.5/0.5)
  - Reciprocal rank fusion (RRF) as alternative
  - Configurable fusion weights
  - File: Update `core/rag/retrieval_service.py`

- [ ] **Task 2.1.3:** Ablation evaluation
  - Run on frozen audit set with frozen prompt/model
  - Measure all metrics
  - Calculate delta vs baseline RAG (dense-only)
  - File: Create `evaluation_runs/ablation_hybrid_retrieval.json`

- [ ] **Task 2.1.4:** Analyze results
  - Did Answer Faithfulness improve by +5-15%?
  - What's the cost impact?
  - What's the latency impact?
  - Any regressions?
  - File: Create `analysis/ablation_2.1_hybrid_retrieval.md`

#### Decision Gate
- **IF** Answer Faithfulness lift ≥ +5% AND no critical regressions → PROCEED
- **IF** lift < +5% → ITERATE on fusion weights or SKIP

#### Success Criteria
- ✅ BM25 + Dense hybrid retrieval operational
- ✅ Measured delta on audit set
- ✅ Answer Faithfulness improved by ≥5%
- ✅ Cost increase <30% vs baseline RAG
- ✅ Latency increase <20% vs baseline RAG
- ✅ No regressions on simple cases

---

### Stage 2.2: Reranking Layer (Weeks 12-13)

#### Component: Cross-Encoder Reranking

**Expected Lift:** +5-10% Context Precision

#### Tasks

- [ ] **Task 2.2.1:** Implement cross-encoder reranker
  - Use model like `cross-encoder/ms-marco-MiniLM-L-12-v2`
  - Rerank top-100 from hybrid retrieval → select top-5
  - Add reranking latency tracking
  - File: Update `core/rag/retrieval_service.py`

- [ ] **Task 2.2.2:** Optimize reranking performance
  - Batch reranking requests
  - Add caching for reranking results
  - Circuit breaker for reranker

- [ ] **Task 2.2.3:** Ablation evaluation
  - Run on frozen audit set with frozen prompt/model
  - Measure all metrics (now includes reranking latency)
  - Calculate delta vs Stage 2.1 (hybrid retrieval)
  - File: Create `evaluation_runs/ablation_reranking.json`

- [ ] **Task 2.2.4:** Analyze results
  - Did Context Precision improve by +5-10%?
  - What's the reranking latency?
  - Does it impact Answer Faithfulness?
  - File: Create `analysis/ablation_2.2_reranking.md`

#### Decision Gate
- **IF** Context Precision lift ≥ +5% AND reranking latency <100ms p95 → PROCEED
- **IF** lift < +5% OR latency too high → EVALUATE LLM reranker alternative or SKIP

#### Success Criteria
- ✅ Cross-encoder reranking operational
- ✅ Measured delta on audit set
- ✅ Context Precision improved by ≥5%
- ✅ Reranking latency p95 <100ms
- ✅ Answer Faithfulness maintained or improved
- ✅ No regressions

---

### Stage 2.3: Query Rewriting (Weeks 14-15)

#### Component: Query Expansion and Normalization

**Expected Lift:** +3-8% Context Recall

#### Tasks

- [ ] **Task 2.3.1:** Implement query rewriting
  - Expand acronyms specific to domain
  - Normalize variations
  - Add context if query is ambiguous
  - Use LLM for query expansion (track cost)
  - File: Create `core/rag/query_processor.py`

- [ ] **Task 2.3.2:** Implement multi-query retrieval
  - Generate 2-3 query variations
  - Retrieve with each variation
  - Merge results with deduplication
  - File: Update `core/rag/query_processor.py`

- [ ] **Task 2.3.3:** Ablation evaluation
  - Run on frozen audit set
  - Calculate delta vs Stage 2.2 (with reranking)
  - File: Create `evaluation_runs/ablation_query_rewriting.json`

- [ ] **Task 2.3.4:** Analyze results
  - Did Context Recall improve by +3-8%?
  - What's the token cost impact of query expansion?
  - File: Create `analysis/ablation_2.3_query_rewriting.md`

#### Decision Gate
- **IF** Context Recall lift ≥ +3% AND cost increase <20% → PROCEED
- **IF** lift < +3% → SIMPLIFY or SKIP

#### Success Criteria
- ✅ Query rewriting operational
- ✅ Measured delta on audit set
- ✅ Context Recall improved by ≥3%
- ✅ Token cost increase <20%
- ✅ Latency impact acceptable
- ✅ No regressions

---

### Stage 2.4: Context-Aware Chunking (Weeks 16-17)

#### Component: Semantic Chunking with Metadata

**Expected Lift:** +2-5% Answer Relevance

#### Tasks

- [ ] **Task 2.4.1:** Implement semantic chunking
  - Chunk at semantic boundaries (preserve sentences)
  - Use sentence-window retrieval (retrieve context around chunk)
  - Maintain 512 token chunks with 128 token overlap
  - File: Update `core/rag/document_processor.py`

- [ ] **Task 2.4.2:** Enhance metadata schema
  - Source document name
  - Last updated date
  - Label category (GDPR, CCPA, etc.)
  - Extracted entities (if applicable)
  - Annotation confidence (if applicable)
  - File: Update `core/rag/document_processor.py`

- [ ] **Task 2.4.3:** Implement metadata filtering
  - Filter retrieved chunks by metadata
  - Add metadata to prompt context
  - File: Update `core/rag/retrieval_service.py`

- [ ] **Task 2.4.4:** Re-index documents with new chunking
  - Process all documents with semantic chunking
  - Add metadata to all chunks
  - Rebuild vector index

- [ ] **Task 2.4.5:** Ablation evaluation
  - Run on frozen audit set
  - Calculate delta vs Stage 2.3 (with query rewriting)
  - File: Create `evaluation_runs/ablation_contextual_chunking.json`

- [ ] **Task 2.4.6:** Analyze results
  - Did Answer Relevance improve by +2-5%?
  - Did metadata filtering help?
  - File: Create `analysis/ablation_2.4_contextual_chunking.md`

#### Decision Gate
- **IF** Answer Relevance lift ≥ +2% → PROCEED
- **IF** lift < +2% → Metadata only or SKIP

#### Success Criteria
- ✅ Semantic chunking operational
- ✅ Metadata schema implemented
- ✅ Documents re-indexed successfully
- ✅ Measured delta on audit set
- ✅ Answer Relevance improved by ≥2%
- ✅ No regressions

---

### Stage 2.5: Adaptive Routing (Weeks 18-19)

#### Component: Simple vs Complex Query Routing

**Expected Lift:** +3-7% on complex query accuracy

#### Tasks

- [ ] **Task 2.5.1:** Implement query classifier
  - Classify queries as simple or complex
  - Simple: direct retrieval
  - Complex: multi-hop or iterative retrieval
  - Use lightweight classifier or LLM-based
  - File: Create `core/rag/query_router.py`

- [ ] **Task 2.5.2:** Implement routing logic
  - Simple queries: standard retrieval pipeline
  - Complex queries: enhanced retrieval (more chunks, iterative refinement)
  - Track routing decisions
  - File: Update `core/rag/retrieval_service.py`

- [ ] **Task 2.5.3:** Ablation evaluation
  - Run on frozen audit set
  - Separate analysis for simple vs complex queries
  - Calculate delta vs Stage 2.4 (with contextual chunking)
  - File: Create `evaluation_runs/ablation_adaptive_routing.json`

- [ ] **Task 2.5.4:** Analyze results
  - Did complex query accuracy improve by +3-7%?
  - What % of queries are classified as complex?
  - Is routing justified by gains?
  - File: Create `analysis/ablation_2.5_adaptive_routing.md`

#### Decision Gate
- **IF** complex query lift ≥ +3% AND >10% queries are complex → KEEP
- **IF** lift < +3% OR few complex queries → SKIP

#### Success Criteria
- ✅ Query routing operational
- ✅ Measured delta on audit set
- ✅ Complex query accuracy improved by ≥3%
- ✅ Routing justified by query distribution
- ✅ Simple queries not negatively impacted
- ✅ No regressions

---

### Phase 2 Exit Criteria

- [ ] ✅ All five ablation stages completed
- [ ] ✅ Each component validated with frozen evaluation
- [ ] ✅ Deltas documented for each stage
- [ ] ✅ Cumulative improvement on north star metric >15%
- [ ] ✅ Cost per answer within SLO (<$0.01)
- [ ] ✅ Latency p95 within SLO (<2000ms end-to-end)
- [ ] ✅ No regressions on baseline cases
- [ ] ✅ All ablation reports completed

**Decision Point:** Phase 2 Review
- Review cumulative improvements
- Evaluate cost-accuracy tradeoffs
- Decide which components to keep in production
- Approve moving to advanced techniques (if justified)

---

## Phase 3: Advanced Techniques Validation (4-6 weeks)

**Goal:** Validate advanced RAG techniques (PageIndex, CAG, Graph RAG) on OUR data before adoption. Only pursue if clear value demonstrated.

**Critical Rule:** Do NOT adopt any technique unless it meets adoption criteria on internal data.

---

### Stage 3.1: PageIndex Validation (Weeks 20-21)

#### Technique: PageIndex Context-Aware Retrieval

**Vendor Claim:** 98% accuracy on FinanceBench

**Our Goal:** Validate on internal dataset and measure cost-accuracy tradeoff

#### Tasks

- [ ] **Task 3.1.1:** Implement PageIndex
  - Research PageIndex methodology
  - Implement context-aware retrieval logic
  - File: Create `core/rag/advanced/pageindex.py`

- [ ] **Task 3.1.2:** Create validation protocol
  - Select representative subset of audit set
  - Define success criteria (see below)
  - File: Create `validation_protocols/pageindex_validation.md`

- [ ] **Task 3.1.3:** Run validation on internal data
  - **NOT on FinanceBench** (vendor data)
  - Run on our audit set
  - Measure accuracy, cost, latency
  - Compare to baseline RAG (end of Phase 2)
  - File: Create `evaluation_runs/advanced_pageindex.json`

- [ ] **Task 3.1.4:** Analyze results
  - Accuracy lift on our data
  - Token cost increase
  - Latency increase
  - Edge case performance
  - File: Create `analysis/advanced_pageindex_analysis.md`

#### Adoption Criteria
- **Accuracy lift:** >5% on internal eval set (compared to Phase 2 baseline)
- **Latency increase:** <20% vs baseline
- **Cost increase:** <30% vs baseline
- **No regressions:** On simple queries

#### Decision
- **IF** meets ALL adoption criteria → KEEP for A/B testing
- **IF** fails any criteria → DOCUMENT learnings and REJECT

#### Success Criteria (for attempting)
- ✅ PageIndex implemented and functional
- ✅ Validation run on internal data (NOT FinanceBench)
- ✅ Cost-accuracy tradeoff measured
- ✅ Decision documented (adopt or reject)

---

### Stage 3.2: Contextual Caching Validation (Weeks 22-23)

#### Technique: CAG (Contextual Augmented Generation) with Caching

**Vendor Claim:** 50-70% latency reduction

**Our Goal:** Validate cache hit rates on diverse classification queries (likely much lower than 50-70%)

#### Tasks

- [ ] **Task 3.2.1:** Analyze query patterns
  - Analyze production query logs
  - Measure prefix overlap between queries
  - Estimate realistic cache hit rate
  - File: Create `analysis/query_pattern_analysis.md`

- [ ] **Task 3.2.2:** Implement contextual caching
  - Implement prefix caching for prompts
  - Implement embedding caching
  - Track cache hit rates by type
  - File: Create `core/rag/advanced/contextual_cache.py`

- [ ] **Task 3.2.3:** Run validation
  - Test on production-like query patterns
  - Measure actual cache hit rates
  - Measure latency reduction
  - Measure memory overhead
  - File: Create `evaluation_runs/advanced_contextual_caching.json`

- [ ] **Task 3.2.4:** Cache invalidation strategy
  - Define when to invalidate cache
  - Document cache staleness impact on accuracy
  - Implement cache management

- [ ] **Task 3.2.5:** Analyze results
  - What is ACTUAL cache hit rate? (likely 20-40%, not 50-70%)
  - What is ACTUAL latency reduction?
  - What is memory overhead?
  - Is it worth the complexity?
  - File: Create `analysis/advanced_caching_analysis.md`

#### Adoption Criteria
- **Cache hit rate:** >30% on production queries
- **Latency reduction:** >20% end-to-end (when cache hits)
- **Memory overhead:** <500MB
- **No accuracy degradation:** Due to cache staleness

#### Decision
- **IF** meets ALL adoption criteria → KEEP for A/B testing
- **IF** fails any criteria → DOCUMENT learnings and REJECT

#### Success Criteria (for attempting)
- ✅ Query patterns analyzed
- ✅ Contextual caching implemented
- ✅ Realistic cache hit rates measured
- ✅ Memory and latency impact measured
- ✅ Decision documented (adopt or reject)

---

### Stage 3.3: Graph RAG Evaluation (Weeks 24-25)

#### Technique: Graph RAG for Multi-Hop Reasoning

**Vendor Claim:** Better multi-hop reasoning

**Our Goal:** Determine if >20% of queries require multi-hop, and if Graph RAG is worth the complexity

#### Tasks

- [ ] **Task 3.3.1:** Analyze query complexity
  - Classify audit set queries by complexity
  - Identify queries requiring multi-hop reasoning
  - What % require multi-hop?
  - File: Create `analysis/multi_hop_query_analysis.md`

- [ ] **Task 3.3.2:** IF >20% are multi-hop → Evaluate Graph RAG
  - Research Graph RAG implementation options
  - Assess knowledge graph construction complexity
  - Estimate maintenance costs (schema drift, updates)
  - File: Create `analysis/graph_rag_feasibility.md`

- [ ] **Task 3.3.3:** IF feasible → Implement prototype
  - Build knowledge graph from policy documents
  - Implement graph traversal for multi-hop queries
  - File: Create `core/rag/advanced/graph_rag.py`

- [ ] **Task 3.3.4:** IF prototype → Run validation
  - Test ONLY on multi-hop subset of audit set
  - Compare to standard RAG on same subset
  - Measure accuracy lift on multi-hop queries
  - File: Create `evaluation_runs/advanced_graph_rag.json`

- [ ] **Task 3.3.5:** Analyze results
  - Accuracy lift on multi-hop queries
  - ETL complexity
  - Maintenance cost estimate (eng hours/week)
  - Is simpler alternative (multi-query retrieval) sufficient?
  - File: Create `analysis/advanced_graph_rag_analysis.md`

#### Adoption Criteria
- **Multi-hop queries:** >20% of production queries
- **Accuracy lift on multi-hop:** >15% vs standard RAG
- **Maintenance cost:** <2 eng days/week
- **ETL complexity:** Acceptable for team capacity

#### Decision
- **IF** meets ALL adoption criteria → KEEP for A/B testing
- **IF** fails any criteria → DOCUMENT learnings and REJECT
- **IF** <20% queries are multi-hop → SKIP Graph RAG entirely

#### Success Criteria (for attempting)
- ✅ Query complexity analyzed
- ✅ Multi-hop queries identified (% of total)
- ✅ IF justified, feasibility assessed
- ✅ IF feasible, prototype implemented and validated
- ✅ Decision documented (adopt, reject, or skip)

---

### Phase 3 Exit Criteria

- [ ] ✅ PageIndex validated on internal data
- [ ] ✅ Contextual caching validated with realistic cache hit rates
- [ ] ✅ Graph RAG evaluated (or skipped if <20% multi-hop)
- [ ] ✅ Each technique has clear adopt/reject decision
- [ ] ✅ Techniques that pass criteria queued for A/B testing
- [ ] ✅ Techniques that fail criteria documented with learnings
- [ ] ✅ NO techniques adopted without meeting criteria

**Decision Point:** Phase 3 Review
- Review all advanced technique evaluations
- Finalize list of techniques for production A/B testing
- Document lessons learned from rejected techniques
- Plan A/B testing schedule for approved techniques

---

## Phase 4: Production Operations (Ongoing)

**Goal:** Continuous monitoring, optimization, and A/B testing in production environment.

---

### Stage 4.1: SLO Monitoring (Ongoing)

#### Objectives
- Monitor all SLOs continuously
- Alert on SLO violations
- Track trends over time

#### Tasks

- [ ] **Task 4.1.1:** Implement SLO tracking dashboards
  - Latency SLO dashboard (p50/p95/p99 by component)
  - Accuracy SLO dashboard (Answer Faithfulness, Context Precision, etc.)
  - Cost SLO dashboard (cost/answer, token usage)
  - Availability SLO dashboard (uptime, error rates)
  - File: Update dashboards in monitoring system

- [ ] **Task 4.1.2:** Configure SLO alerts
  - Alert if p95 latency exceeds SLO
  - Alert if accuracy drops below SLO
  - Alert if cost/answer exceeds SLO
  - Alert if error rate spikes
  - File: Configure alerting in monitoring system

- [ ] **Task 4.1.3:** Weekly SLO review meetings
  - Review SLO adherence
  - Investigate violations
  - Plan optimization efforts
  - Deliverable: Weekly SLO report

#### Success Criteria
- ✅ All SLOs tracked and visible
- ✅ Alerts configured and tested
- ✅ SLO review process established

---

### Stage 4.2: A/B Testing & Optimization (Ongoing)

#### Objectives
- Run controlled experiments for new components
- Optimize performance and cost
- Continuous improvement

#### Tasks

- [ ] **Task 4.2.1:** Schedule A/B tests
  - Test components approved from Phase 2/3
  - Start with 95/5 traffic split
  - Monitor success metrics and guardrails
  - File: Create experiment schedules

- [ ] **Task 4.2.2:** Hyperparameter optimization
  - Optimize fusion weights for hybrid retrieval
  - Optimize top-k values
  - Optimize chunking parameters
  - Use A/B testing to validate changes

- [ ] **Task 4.2.3:** Token budget optimization
  - Reduce prompt token usage where possible
  - Optimize chunk selection to reduce noise
  - Track cost reduction vs accuracy impact

- [ ] **Task 4.2.4:** Performance optimization
  - Optimize embedding caching
  - Optimize vector DB query performance
  - Optimize reranking batch size
  - Reduce end-to-end latency

#### Success Criteria
- ✅ A/B testing framework in active use
- ✅ Optimization experiments run regularly
- ✅ Cost/answer trending downward
- ✅ Latency trending downward
- ✅ Accuracy maintained or improved

---

### Stage 4.3: Index Maintenance (Ongoing)

#### Objectives
- Keep vector index fresh and accurate
- Monitor index staleness
- Manage index growth

#### Tasks

- [ ] **Task 4.3.1:** Document update pipeline
  - How frequently are documents updated?
  - Automated re-indexing on document changes
  - Track document last-updated timestamps
  - File: Create `docs/index_maintenance.md`

- [ ] **Task 4.3.2:** Monitor index staleness
  - Alert if documents >7 days old
  - Dashboard showing document freshness
  - Track embedding model version for rollback

- [ ] **Task 4.3.3:** Manage index growth
  - Monitor index size over time
  - Plan for scaling vector DB
  - Archival strategy for old documents

#### Success Criteria
- ✅ Index maintenance process documented
- ✅ Staleness monitoring in place
- ✅ Index updates automated
- ✅ Growth managed proactively

---

### Stage 4.4: Continuous Evaluation (Ongoing)

#### Objectives
- Maintain human audit set
- Detect performance drift
- Calibrate LLM-as-judge regularly

#### Tasks

- [ ] **Task 4.4.1:** Expand human audit set
  - Add new edge cases as discovered
  - Maintain diversity in audit set
  - Target: grow to 500+ samples over time

- [ ] **Task 4.4.2:** Regular calibration
  - Quarterly re-calibration of LLM-as-judge
  - Compare LLM evaluations to human evaluations
  - Adjust judge prompts if correlation drifts

- [ ] **Task 4.4.3:** Drift detection
  - Monitor response patterns for drift
  - Alert if system behavior changes unexpectedly
  - Investigate and address drift causes

#### Success Criteria
- ✅ Audit set grows and stays current
- ✅ LLM-as-judge correlation stays >0.85
- ✅ Drift detection catches anomalies

---

### Stage 4.5: Operational Runbooks (Ongoing)

#### Objectives
- Document operational procedures
- Enable team to debug and respond to issues
- Continuous improvement of processes

#### Tasks

- [ ] **Task 4.5.1:** Create runbooks
  - Runbook: Vector DB is down (fallback procedures)
  - Runbook: Latency SLO violation (debugging steps)
  - Runbook: Accuracy drop detected (investigation checklist)
  - Runbook: Cost spike (analysis and mitigation)
  - File: Create `docs/runbooks/`

- [ ] **Task 4.5.2:** Incident response
  - Define incident severity levels
  - Define escalation paths
  - Post-incident review process

- [ ] **Task 4.5.3:** Team training
  - Train team on runbooks
  - Run incident simulations
  - Document lessons learned

#### Success Criteria
- ✅ Runbooks created and accessible
- ✅ Team trained on runbooks
- ✅ Incident response times improving

---

## Success Criteria & Metrics

### Phase 0 Success Metrics
- System test pass rate: 100%
- Ollama failure rate: <1%
- Security false positive rate: <5%
- Evaluation framework calibration: >0.85 correlation
- Pre-RAG baseline: Measured and documented

### Phase 1 Success Metrics
- Dense RAG operational: Yes
- Graceful degradation tested: Yes
- Baseline RAG vs pre-RAG: Positive delta on north star metric
- No regressions: Confirmed

### Phase 2 Success Metrics
- Hybrid retrieval delta: ≥+5% Answer Faithfulness
- Reranking delta: ≥+5% Context Precision
- Query rewriting delta: ≥+3% Context Recall
- Contextual chunking delta: ≥+2% Answer Relevance
- Adaptive routing delta: ≥+3% complex query accuracy
- Cumulative improvement: >15% on north star metric
- Cost per answer: <$0.01 (within SLO)
- Latency p95: <2000ms (within SLO)

### Phase 3 Success Metrics
- PageIndex: Meets adoption criteria or rejected with documented rationale
- Contextual caching: Realistic cache hit rate measured, adopt or reject decision
- Graph RAG: Multi-hop % identified, evaluated if justified, adopt/reject decision
- All techniques: Clear adoption decisions based on internal data validation

### Phase 4 Success Metrics
- SLO adherence: >95% of time within SLOs
- A/B testing: Regular experiments driving improvements
- Index freshness: Documents updated within 7 days
- Audit set: Growing and staying current (500+ samples)
- Team response: Incident runbooks used effectively

---

## Risk Mitigation

### Risk 1: Phase 0 Takes Longer Than Expected
**Mitigation:** Phase 0 is BLOCKING. Do not compromise. Add resources if needed. Skipping Phase 0 will cause downstream failures.

### Risk 2: RAG Doesn't Improve Accuracy
**Mitigation:** We have pre-RAG baseline. If dense RAG doesn't improve north star metric, we STOP and investigate before proceeding. Not all use cases benefit from RAG.

### Risk 3: Cost Exceeds Budget
**Mitigation:** Define cost SLO upfront. Track cost at every stage. If component exceeds cost budget, optimize or reject. Token budgeting and alerts prevent surprises.

### Risk 4: Latency Exceeds SLO
**Mitigation:** Define latency SLO upfront. Track latency by component. Optimize slow components. Graceful degradation ensures system stays responsive even if retrieval is slow.

### Risk 5: Advanced Techniques Don't Validate
**Mitigation:** This is expected. Validate each technique on internal data. If it doesn't meet adoption criteria, REJECT it. Document learnings. Not all techniques are universally beneficial.

### Risk 6: Team Resistance to Measurement-First Approach
**Mitigation:** Educate team on production RAG best practices. Show examples of failed deployments from skipping evaluation. Emphasize that measurement protects us from wasted effort.

### Risk 7: Audit Set Not Representative
**Mitigation:** Carefully curate audit set with domain experts. Include edge cases and typical cases. Grow audit set over time. Validate that audit set correlates with production performance.

### Risk 8: LLM-as-Judge Not Reliable
**Mitigation:** Calibrate against human labels. If correlation <0.85, improve judge prompts or use more human evaluation. Don't trust uncalibrated LLM judgments.

---

## Decision Gates

### Decision Gate 0: After Phase 0
**Question:** Are we ready to implement RAG?
- All Phase 0 exit criteria met?
- Team has confidence in evaluation framework?
- Pre-RAG baseline establishes clear comparison point?

**Decision:** GO / NO-GO to Phase 1

### Decision Gate 1: After Phase 1
**Question:** Does RAG provide measurable value?
- Baseline RAG shows positive delta on north star metric?
- Cost and latency acceptable?
- No major regressions?

**Decision:** GO / NO-GO to Phase 2

### Decision Gate 2: After Each Ablation Stage
**Question:** Does this component meet lift expectations?
- Delta on target metric meets expectations?
- Cost and latency impact acceptable?
- No regressions?

**Decision:** KEEP / ITERATE / SKIP to next stage

### Decision Gate 3: After Phase 2
**Question:** Is production RAG ready?
- Cumulative improvements >15% on north star?
- Cost per answer <$0.01?
- Latency p95 <2000ms?
- Confident in component choices?

**Decision:** GO to Phase 3 / OPTIMIZE PHASE 2 / DEPLOY TO PRODUCTION (skip Phase 3)

### Decision Gate 4: For Each Advanced Technique
**Question:** Should we adopt this technique?
- Meets adoption criteria on internal data?
- Cost-accuracy tradeoff justified?
- Team has capacity to maintain it?

**Decision:** ADOPT for A/B testing / REJECT with documentation

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 0: Foundation** | 4-6 weeks | System stable (100% tests), evaluation framework, pre-RAG baseline, SLOs defined, operational tooling |
| **Phase 1: Baseline RAG** | 2-3 weeks | Dense-only retrieval operational, baseline RAG metrics, frozen prompt template |
| **Phase 2: Ablation** | 8-10 weeks | Hybrid retrieval, reranking, query rewriting, contextual chunking, adaptive routing (all validated) |
| **Phase 3: Advanced Validation** | 4-6 weeks | PageIndex validated, contextual caching validated, Graph RAG evaluated |
| **Phase 4: Production Ops** | Ongoing | SLO monitoring, A/B testing, index maintenance, continuous evaluation |

**Total Estimated Timeline:** 18-24 weeks from start to production-ready RAG system with advanced techniques evaluated.

---

## Key Principles

1. **Evaluation Before Innovation:** If you can't measure it, you can't improve it.
2. **Baselines Before Enhancements:** Establish dense-only RAG baseline before adding complexity.
3. **Incremental Validation:** Add ONE component at a time, measure delta.
4. **Internal Data Validation:** Never trust vendor benchmarks. Validate on YOUR data.
5. **Cost-Accuracy Tradeoffs:** Measure cost AND accuracy for every change.
6. **Operational Resilience:** Circuit breakers, fallbacks, graceful degradation are not optional.
7. **SLOs Over Vibes:** Define SLOs upfront. "It feels better" is not a metric.
8. **Document Everything:** Successes AND failures. Failed experiments are valuable learnings.
9. **Team Over Tools:** Choose techniques the team can maintain and operate.
10. **Reject Unvalidated Complexity:** If it doesn't meet adoption criteria, reject it confidently.

---

## Conclusion

This implementation plan transforms the RAG enhancement from a feature-driven effort into a **production-engineering effort** grounded in measurement, validation, and operational excellence.

By following this plan, the Agentic Reviewer will evolve into a production-grade RAG system with:
- ✅ **Proven value** at each step (measured deltas on internal data)
- ✅ **Operational resilience** (circuit breakers, fallbacks, monitoring)
- ✅ **Cost efficiency** (token budgeting, SLO adherence)
- ✅ **Continuous improvement** (A/B testing, systematic optimization)
- ✅ **Clear decision-making** (adopt/reject based on evidence, not hype)

**Remember:** RAG is a retrieval service with LLM integration, not magic. Build it with the same rigor you'd apply to any production system: measure everything, optimize systematically, and reject unvalidated complexity.

---

## Next Steps

1. **Schedule Phase 0 Kickoff Meeting**
   - Review this plan with full team
   - Assign ownership for Phase 0 stages
   - Confirm timeline and resources

2. **Create Phase 0 Tracking**
   - Set up project board with all Phase 0 tasks
   - Assign task owners
   - Set weekly check-in meetings

3. **Begin Phase 0.1: Core System Stabilization**
   - Fix failing tests as first priority
   - Get to 100% test pass rate
   - No RAG work until Phase 0 complete

**Do not skip ahead. The foundation is critical.**

noteId: "059141e0aed811f096a62b399f5b86fc"
tags: []

---

