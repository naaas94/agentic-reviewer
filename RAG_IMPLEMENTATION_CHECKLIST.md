---
noteId: "3435dec0aed811f096a62b399f5b86fc"
tags: []

---

# RAG Implementation Checklist
## Quick Reference for Daily Progress Tracking

**Version:** 1.0  
**Last Updated:** October 21, 2025  
**Full Plan:** See `RAG_IMPLEMENTATION_PLAN.md`

---

## ⚠️ PRE-FLIGHT CHECKLIST (BLOCKING)

**DO NOT START RAG UNTIL ALL ITEMS COMPLETE:**

- [ ] System achieves 100% test pass rate
- [ ] Ollama integration stable (<1% failure rate)
- [ ] Security validation tuned (<5% false positives)
- [ ] North star metric selected: ________________
- [ ] Human audit set created (100-500 samples)
- [ ] Evaluation framework calibrated (>0.85 correlation)
- [ ] Pre-RAG baseline measured
- [ ] SLOs defined and documented
- [ ] Token tracking operational
- [ ] Observability dashboards live
- [ ] A/B testing framework ready
- [ ] CI/CD evaluation pipeline operational

**Sign-off:** Engineering Lead [ ] Product Owner [ ] QA Lead [ ]

---

## PHASE 0: FOUNDATION (4-6 WEEKS)

### Week 1-2: Core Stabilization
- [ ] Run full test suite, document failures → `test_failure_analysis.md`
- [ ] Fix Ollama connection issues (retry, pooling, health checks)
- [ ] Fix all failing agent tests (100% pass rate)
- [ ] Validate circuit breaker functionality
- [ ] Fix security validation false positives (<5%)
- [ ] Run integration tests (API, load, cache)
- [ ] **Milestone:** 100% test pass rate achieved

### Week 3-4: Evaluation Framework
- [ ] Select north star metric (recommend: Answer Faithfulness)
- [ ] Document metric selection → `docs/metrics_selection.md`
- [ ] Create human audit set (100-500 samples) → `data/audit_set_v1.json`
- [ ] Implement `RAGEvaluationFramework` class → `core/evaluation/rag_metrics.py`
- [ ] Calibrate LLM-as-judge (>0.85 correlation with humans)
- [ ] Measure pre-RAG baseline on audit set → `baselines/pre_rag_baseline.json`
- [ ] Define SLOs (latency, accuracy, cost) → `docs/slos.md`
- [ ] **Milestone:** Baseline measured, SLOs defined

### Week 5-6: Operational Readiness
- [ ] Implement token budgeting and cost tracking
- [ ] Add RAG-specific observability (retrieval latency, etc.)
- [ ] Create operational dashboards (health, cost, latency, errors)
- [ ] Build A/B testing framework → `core/operations/ab_testing.py`
- [ ] Set up CI/CD for evaluation → `.github/workflows/evaluation_pipeline.yml`
- [ ] Create debugging tools → `tools/debug_inspector.py`
- [ ] **Milestone:** Operational tooling ready

**Phase 0 Exit Gate:** All pre-flight items checked + all week 1-6 milestones complete

---

## PHASE 1: BASELINE RAG (2-3 WEEKS)

### Week 7: Vector DB Setup
- [ ] Deploy vector database (recommend: FAISS)
- [ ] Implement embedding service → `core/rag/embedding_service.py`
  - [ ] Evaluate BGE, MiniLM, domain-specific embeddings
  - [ ] Document model selection rationale
- [ ] Build document ingestion → `core/rag/document_processor.py`
  - [ ] Fixed-size chunking (512 tokens, 128 overlap)
  - [ ] Metadata extraction (document name, date)
- [ ] Implement retrieval service → `core/rag/retrieval_service.py`
  - [ ] Dense-only retrieval (top-k)
  - [ ] Circuit breaker for vector DB
  - [ ] Fallback to base LLM if retrieval fails
- [ ] **Milestone:** Vector DB operational

### Week 8: RAG Integration
- [ ] Update unified agent for RAG → `agents/unified_agent.py`
- [ ] Create RAG prompt template v1.0 → `prompts/*.txt`
- [ ] **FREEZE** prompt template and model for ablation testing
- [ ] Add RAG-specific observability (retrieval latency, failures, fallbacks)
- [ ] Test graceful degradation (kill vector DB, verify fallback)
- [ ] **Milestone:** RAG integrated

### Week 9: Baseline Evaluation
- [ ] Run evaluation on audit set with frozen prompt/model
- [ ] Calculate all metrics (faithfulness, precision, recall, cost, latency)
- [ ] Compare to pre-RAG baseline → `analysis/rag_vs_no_rag_comparison.md`
- [ ] Statistical significance testing
- [ ] Regression testing (no harm to simple cases)
- [ ] Document baseline RAG metrics → `evaluation_runs/baseline_rag_dense_only.json`
- [ ] **Milestone:** Baseline RAG validated

**Phase 1 Exit Gate:** RAG operational + positive delta on north star metric + no regressions

---

## PHASE 2: ABLATION (8-10 WEEKS)

**Remember:** Keep prompt/model FROZEN. Test ONE component at a time. Report DELTAS.

### Stage 2.1: Hybrid Retrieval (Weeks 10-11)
- [ ] Implement BM25 index
- [ ] Implement hybrid fusion (BM25 + Dense)
- [ ] Run ablation eval (frozen audit set) → `evaluation_runs/ablation_hybrid_retrieval.json`
- [ ] Analyze results → `analysis/ablation_2.1_hybrid_retrieval.md`
- [ ] **Target:** +5-15% Answer Faithfulness
- [ ] **Decision Gate:** Lift ≥+5%? [ ] YES → Proceed [ ] NO → Iterate/Skip

### Stage 2.2: Reranking (Weeks 12-13)
- [ ] Implement cross-encoder reranker (top-100 → top-5)
- [ ] Optimize reranking (batching, caching, circuit breaker)
- [ ] Run ablation eval → `evaluation_runs/ablation_reranking.json`
- [ ] Analyze results → `analysis/ablation_2.2_reranking.md`
- [ ] **Target:** +5-10% Context Precision, <100ms p95 latency
- [ ] **Decision Gate:** Lift ≥+5% AND latency OK? [ ] YES → Proceed [ ] NO → Iterate/Skip

### Stage 2.3: Query Rewriting (Weeks 14-15)
- [ ] Implement query rewriting (expansion, normalization)
- [ ] Implement multi-query retrieval
- [ ] Run ablation eval → `evaluation_runs/ablation_query_rewriting.json`
- [ ] Analyze results → `analysis/ablation_2.3_query_rewriting.md`
- [ ] **Target:** +3-8% Context Recall, <20% cost increase
- [ ] **Decision Gate:** Lift ≥+3%? [ ] YES → Proceed [ ] NO → Iterate/Skip

### Stage 2.4: Contextual Chunking (Weeks 16-17)
- [ ] Implement semantic chunking (sentence boundaries)
- [ ] Enhance metadata schema (source, date, category, entities)
- [ ] Implement metadata filtering
- [ ] Re-index all documents with new chunking
- [ ] Run ablation eval → `evaluation_runs/ablation_contextual_chunking.json`
- [ ] Analyze results → `analysis/ablation_2.4_contextual_chunking.md`
- [ ] **Target:** +2-5% Answer Relevance
- [ ] **Decision Gate:** Lift ≥+2%? [ ] YES → Proceed [ ] NO → Metadata only/Skip

### Stage 2.5: Adaptive Routing (Weeks 18-19)
- [ ] Implement query classifier (simple vs complex)
- [ ] Implement routing logic (standard vs enhanced retrieval)
- [ ] Run ablation eval → `evaluation_runs/ablation_adaptive_routing.json`
- [ ] Analyze results (separate for simple vs complex) → `analysis/ablation_2.5_adaptive_routing.md`
- [ ] **Target:** +3-7% complex query accuracy
- [ ] **Decision Gate:** Lift ≥+3% AND >10% queries complex? [ ] YES → Keep [ ] NO → Skip

**Phase 2 Exit Gate:** 
- [ ] All ablation stages complete
- [ ] Cumulative improvement >15% on north star metric
- [ ] Cost per answer <$0.01 (within SLO)
- [ ] Latency p95 <2000ms (within SLO)
- [ ] No regressions on baseline cases

---

## PHASE 3: ADVANCED VALIDATION (4-6 WEEKS)

### Stage 3.1: PageIndex (Weeks 20-21)
- [ ] Implement PageIndex → `core/rag/advanced/pageindex.py`
- [ ] Run validation on INTERNAL data (not FinanceBench)
- [ ] Measure accuracy, cost, latency vs Phase 2 baseline
- [ ] Analyze results → `analysis/advanced_pageindex_analysis.md`
- [ ] **Adoption Criteria:**
  - [ ] Accuracy lift >5% on internal data
  - [ ] Latency increase <20%
  - [ ] Cost increase <30%
  - [ ] No regressions
- [ ] **Decision:** [ ] ADOPT for A/B [ ] REJECT (document why)

### Stage 3.2: Contextual Caching (Weeks 22-23)
- [ ] Analyze query patterns (measure prefix overlap) → `analysis/query_pattern_analysis.md`
- [ ] Implement contextual caching → `core/rag/advanced/contextual_cache.py`
- [ ] Run validation (measure realistic cache hit rates)
- [ ] Analyze results → `analysis/advanced_caching_analysis.md`
- [ ] **Adoption Criteria:**
  - [ ] Cache hit rate >30% on production queries
  - [ ] Latency reduction >20% (when cache hits)
  - [ ] Memory overhead <500MB
  - [ ] No accuracy degradation
- [ ] **Decision:** [ ] ADOPT for A/B [ ] REJECT (document why)

### Stage 3.3: Graph RAG (Weeks 24-25)
- [ ] Analyze query complexity (% multi-hop?) → `analysis/multi_hop_query_analysis.md`
- [ ] **If <20% multi-hop:** [ ] SKIP Graph RAG entirely
- [ ] **If ≥20% multi-hop:**
  - [ ] Assess feasibility → `analysis/graph_rag_feasibility.md`
  - [ ] Implement prototype → `core/rag/advanced/graph_rag.py`
  - [ ] Run validation on multi-hop subset → `evaluation_runs/advanced_graph_rag.json`
  - [ ] Analyze results → `analysis/advanced_graph_rag_analysis.md`
  - [ ] **Adoption Criteria:**
    - [ ] Multi-hop queries >20%
    - [ ] Accuracy lift >15% on multi-hop queries
    - [ ] Maintenance cost <2 eng days/week
    - [ ] ETL complexity acceptable
  - [ ] **Decision:** [ ] ADOPT for A/B [ ] REJECT (document why)

**Phase 3 Exit Gate:**
- [ ] All techniques validated on internal data
- [ ] Clear adopt/reject decision for each technique
- [ ] Approved techniques queued for A/B testing
- [ ] Rejected techniques documented with learnings

---

## PHASE 4: PRODUCTION OPS (ONGOING)

### SLO Monitoring
- [ ] SLO tracking dashboards live (latency, accuracy, cost, availability)
- [ ] SLO alerts configured and tested
- [ ] Weekly SLO review meetings scheduled
- [ ] SLO adherence >95% of time

### A/B Testing & Optimization
- [ ] A/B test schedule created for approved components
- [ ] Start with 95/5 traffic splits
- [ ] Run hyperparameter optimization experiments
- [ ] Token budget optimization ongoing
- [ ] Performance optimization (caching, batching, etc.)

### Index Maintenance
- [ ] Document update pipeline documented → `docs/index_maintenance.md`
- [ ] Index staleness monitoring (alert if >7 days)
- [ ] Automated re-indexing on document changes
- [ ] Index growth monitored and managed

### Continuous Evaluation
- [ ] Human audit set growing (target: 500+ samples)
- [ ] Quarterly LLM-as-judge re-calibration
- [ ] Drift detection operational
- [ ] Correlation with humans maintained >0.85

### Operational Runbooks
- [ ] Runbooks created → `docs/runbooks/`
  - [ ] Vector DB down
  - [ ] Latency SLO violation
  - [ ] Accuracy drop detected
  - [ ] Cost spike
- [ ] Incident response process defined
- [ ] Team trained on runbooks
- [ ] Incident simulations run

---

## PROGRESS TRACKING

### Current Phase: _____________
### Current Week: _____________
### Current Status: _____________

### Blockers:
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

### Recent Wins:
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

### Next Milestone: _____________
**Target Date:** _____________
**Owner:** _____________

---

## KEY METRICS DASHBOARD

### System Health
- Test pass rate: _______% (target: 100%)
- Ollama failure rate: _______% (target: <1%)
- Security false positive rate: _______% (target: <5%)

### Evaluation Framework
- Audit set size: _______ samples (target: 100-500)
- LLM-as-judge correlation: _______ (target: >0.85)
- Pre-RAG baseline (north star metric): _______

### RAG Performance
- Current Answer Faithfulness: _______ (target: >0.85)
- Current Context Precision: _______ (target: >0.80)
- Cost per answer: $_______ (target: <$0.01)
- Latency p95: _______ms (target: <2000ms)
- Retrieval p95: _______ms (target: <200ms)
- Reranking p95: _______ms (target: <100ms)

### Cumulative Improvements (vs Pre-RAG Baseline)
- Dense-only RAG: _______% improvement
- + Hybrid retrieval: _______% improvement
- + Reranking: _______% improvement
- + Query rewriting: _______% improvement
- + Contextual chunking: _______% improvement
- + Adaptive routing: _______% improvement
- **Total cumulative improvement: _______%** (target: >15%)

### Operational Metrics
- SLO adherence: _______% (target: >95%)
- Cache hit rate: _______% (query cache)
- Index staleness: _______ days (target: <7)
- A/B experiments run: _______

---

## DECISION LOG

| Date | Phase/Stage | Decision | Rationale | Owner |
|------|-------------|----------|-----------|-------|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

---

## LESSONS LEARNED

### What Worked Well:
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

### What Didn't Work:
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

### Techniques Rejected (and Why):
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

---

## QUICK REFERENCE

### Phase 0 Priorities:
1. Fix all tests (100% pass rate)
2. Create audit set (100-500 samples)
3. Measure pre-RAG baseline
4. Define SLOs
5. Build operational tooling

### Phase 1 Priorities:
1. Dense-only retrieval
2. Circuit breakers and fallbacks
3. Baseline RAG evaluation
4. Freeze prompt template v1.0

### Phase 2 Priorities:
1. Add ONE component at a time
2. Measure delta on frozen eval set
3. Report deltas, not absolute numbers
4. Check for regressions at each step
5. Only proceed if component meets lift expectations

### Phase 3 Priorities:
1. Validate on INTERNAL data (not vendor benchmarks)
2. Measure cost-accuracy tradeoffs
3. Meet adoption criteria or reject
4. Document learnings from failures

### Phase 4 Priorities:
1. Monitor SLOs continuously
2. A/B test new components
3. Optimize cost and latency
4. Maintain audit set
5. Run incident drills

---

## CONTACTS & OWNERSHIP

- **Engineering Lead:** _____________
- **Product Owner:** _____________
- **QA Lead:** _____________
- **Phase 0 Owner:** _____________
- **Phase 1 Owner:** _____________
- **Phase 2 Owner:** _____________
- **Phase 3 Owner:** _____________
- **Phase 4 Owner:** _____________

---

**Remember:**
- Evaluation before innovation
- Baselines before enhancements
- Internal data validation (not vendor benchmarks)
- Cost-accuracy tradeoffs always
- SLOs over vibes
- Reject unvalidated complexity confidently

