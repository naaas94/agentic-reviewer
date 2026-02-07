# Autoptic FDE Dossier — Area D: Integration + Performance Engineering

---

## 1. Repo Scan Summary

| # | Artifact | Path | Dossier Signal |
|---|----------|------|----------------|
| 1 | LLM Client + Circuit Breaker | `agents/base_agent.py` — `BaseAgent._call_ollama()`, `_call_ollama_async()` | Connector boundary, retries, backoff, circuit breaker |
| 2 | Advanced Cache (LRU + TTL) | `core/cache.py` — `AdvancedCache` | Cost containment via LLM-call dedup, memory budgets |
| 3 | Centralized Config (7 dataclasses) | `core/config.py` — `SystemConfig`, `PerformanceConfig`, `APIConfig`, `LLMConfig` | Budget knobs: tokens, concurrency, cache size, rate limits |
| 4 | Security Layer | `core/security.py` — `SecurityManager`, `PromptInjectionDetector`, `AdversarialAttackDetector`, `DriftDetector` | Injection defense, adversarial guardrails, drift detection |
| 5 | Input Validators | `core/validators.py` — `sanitize_text()`, `validate_text()`, `validate_api_key()` | Schema guards, injection prevention |
| 6 | API Server + Middleware | `main.py` — FastAPI app, `security_middleware()`, `rate_limit_check()`, `verify_api_key()` | Protocol contract, rate limiting, request-size guard, security headers |
| 7 | Unified Multi-Task Agent | `agents/unified_agent.py` — `UnifiedAgent.process_batch()`, `process_sample()` | Batching, concurrency semaphore, single-call cost reduction |
| 8 | Health & Metrics | `core/monitoring.py` — `HealthChecker`, `ApplicationMetrics`, `SystemMetrics` | Observability: CPU/mem/disk, LLM error rate, response time tracking |
| 9 | Audit Logger (SQLite) | `core/logger.py` — `AuditLogger` | Token-usage persistence, run metadata |
| 10 | Label Taxonomy | `configs/labels.yaml` | Deterministic entity definitions (GDPR/CCPA) |
| 11 | Prompt Templates (Jinja2) | `prompts/evaluator_prompt.txt`, `proposer_prompt.txt`, `reasoner_prompt.txt` | Guardrailed prompt contracts |
| 12 | Requirements | `requirements.txt` | Dependency surface: `requests`, `aiohttp`, `fastapi`, `psutil`, `pydantic` |

**Best-fit dossier area:** **Area D — Integration + Performance Engineering.** This repo is rich in connector-boundary patterns (LLM client with circuit breaker + exponential backoff), cost-containment levers (LRU cache with TTL + memory budget, token limits, concurrency semaphore, multi-task batching), rate-limiting middleware, and multi-layer security guardrails (prompt injection detection, input sanitization, request-size caps, security headers). Observability is baked in via `psutil` system metrics, application-level counters, and SQLite audit logging with token tracking.

---

## 2. Dossier Insert — Area D

### Cost & Performance Levers: Ollama LLM Connector + Cache + API Gateway

---

#### Location (evidence)

| Component | File | Symbols | Search Token |
|-----------|------|---------|--------------|
| LLM HTTP Client (sync) | `agents/base_agent.py` | `BaseAgent._call_ollama()` | `"circuit_breaker:{self.ollama_url}"` |
| LLM HTTP Client (async) | `agents/base_agent.py` | `BaseAgent._call_ollama_async()` | `aiohttp.ClientTimeout(total=` |
| Circuit Breaker | `agents/base_agent.py` L147-175 | `cache_key = f"circuit_breaker:` | `"half_open"` |
| Cache-before-LLM | `agents/base_agent.py` L418-438 | `BaseAgent._call_llm_common()` | `"Cache hit for prompt hash:"` |
| Advanced Cache | `core/cache.py` L30-298 | `AdvancedCache` | `class AdvancedCache:` |
| LRU Eviction | `core/cache.py` L219-238 | `AdvancedCache._evict_entries()` | `sorted_entries = sorted(` |
| Performance Config | `core/config.py` L124-151 | `PerformanceConfig` | `class PerformanceConfig:` |
| LLM Config | `core/config.py` L14-46 | `LLMConfig` | `class LLMConfig:` |
| API Rate Limiter | `main.py` L177-200 | `rate_limit_check()` | `HTTP_429_TOO_MANY_REQUESTS` |
| Request-Size Guard | `main.py` L225-231 | `security_middleware()` | `HTTP_413_REQUEST_ENTITY_TOO_LARGE` |
| Concurrency Semaphore | `agents/unified_agent.py` L194-220 | `UnifiedAgent.process_batch()` | `asyncio.Semaphore(max_concurrent)` |
| Multi-Task Batching | `agents/base_agent.py` L482-499 | `BaseAgent.process_multi_task()` | `_create_multi_task_prompt` |
| Security Manager | `core/security.py` L475-587 | `SecurityManager.validate_input()` | `_calculate_risk_score` |
| Prompt Injection Detector | `core/security.py` L28-153 | `PromptInjectionDetector.detect_injection()` | `self.injection_patterns` |
| Input Sanitizer | `core/validators.py` L11-39 | `sanitize_text()` | `html.escape(text)` |
| Health Checker | `core/monitoring.py` L38-179 | `HealthChecker.is_healthy()` | `class HealthChecker:` |
| Drift Detector | `core/security.py` L396-472 | `DriftDetector.detect_drift()` | `self.drift_threshold = 0.1` |

---

#### Integration boundary

| Boundary | Systems Touched | Contract |
|----------|----------------|----------|
| **LLM Connector** (`_call_ollama` / `_call_ollama_async`) | Ollama API (`POST /api/generate`) | **Input:** JSON `{model, prompt, stream:false, options:{num_predict, temperature}}` **Output:** `{response, eval_count}` **Timeout:** configurable (`LLMConfig.timeout_seconds`, default 30s) **Retries:** `max_retries=3`, exponential backoff `delay * 2^attempt` **Circuit breaker:** open/half-open/closed; backoff 1-5 min; 3 successes to close |
| **API Gateway** (`main.py` FastAPI) | HTTP clients (external callers) | **Input:** Pydantic-validated `ReviewRequest` (text <=10 000 chars, confidence [0,1], label <=100 chars) **Output:** `ReviewResponse` **Auth:** Bearer token (optional, `AR_API_KEY`) **Rate limit:** 100 req/hr sliding window **Size cap:** 1 MB **Security headers:** HSTS, CSP, X-Frame-Options, X-XSS-Protection |
| **Cache Layer** (`AdvancedCache`) | In-memory (optionally persisted to `outputs/cache.pkl`) | **Input:** key (MD5 of prompt) + value + TTL **Eviction:** LRU when `max_size_mb` (100 MB) or `max_entries` (10 000) exceeded **Thread safety:** `threading.Lock` |
| **SQLite Audit** (`AuditLogger`) | Local SQLite (`outputs/reviewed_predictions.sqlite`) | **Fields:** review result + `tokens_used` + `latency_ms` + run metadata **WAL mode** for concurrent reads |

---

#### Cost containment levers

- **(EVIDENCE)** **LLM-call deduplication via cache-before-call:** `BaseAgent._call_llm_common()` checks `AdvancedCache.get(cache_key)` before every LLM invocation; cache key is MD5 of rendered prompt. (File: `agents/base_agent.py` L418-423, search token: `"Cache hit for prompt hash:"`)
- **(EVIDENCE)** **Multi-task batching — single LLM call for 3 agent tasks:** `UnifiedAgent.process_sample()` combines evaluate+propose+reason into one prompt via `process_multi_task()`, cutting token overhead ~3x vs. sequential calls. (File: `agents/unified_agent.py` L88-90, `agents/base_agent.py` L482-499)
- **(EVIDENCE)** **Token cap per request:** `LLMConfig.max_tokens = 512` (configurable via `AR_MAX_TOKENS`). Propagated as `num_predict` in Ollama payload. (File: `core/config.py` L19)
- **(EVIDENCE)** **Cache memory budget:** `PerformanceConfig.cache_max_size_mb = 100` with LRU eviction when breached. (File: `core/config.py` L131, `core/cache.py` L219-238)
- **(EVIDENCE)** **Token-usage audit logging:** Every LLM response captures `eval_count` (tokens) and `latency_ms`; stored in SQLite `reviews` table. (File: `agents/base_agent.py` L190-194, `core/logger.py` — `tokens_used INTEGER` column)
- **(INFERENCE)** **Autoptic translation:** The cache-before-LLM pattern and multi-task batching are direct analogs to Autoptic's observability-cost-containment pillar — deduplicating telemetry queries (PQL calls) before hitting expensive backends, and batching multiple diagnostic checks into single inference passes to reduce AI compute spend.

---

#### Latency levers

- **(EVIDENCE)** **Async HTTP client + semaphore-bounded concurrency:** `_call_ollama_async()` uses `aiohttp.ClientSession` with `aiohttp.ClientTimeout(total=...)`. Batch processing throttled via `asyncio.Semaphore(max_concurrent)` (default 5). (File: `agents/base_agent.py` L273-278, `agents/unified_agent.py` L208)
- **(EVIDENCE)** **Circuit breaker prevents wasted latency:** Open state short-circuits immediately (`raise LLMConnectionError`), avoiding timeout waits. Half-open allows a single probe. 3 successes required to close. Backoff: `min(60 * 2^failures, 300)` seconds. (File: `agents/base_agent.py` L147-175)
- **(EVIDENCE)** **Exponential backoff on retries:** `delay = retry_delay_seconds * (2 ** attempt)` with default `retry_delay_seconds=1.0`, `max_retries=3`. (File: `agents/base_agent.py` L236-238, `core/config.py` L21-22)
- **(EVIDENCE)** **Request-pipeline short-circuiting:** Rate limiter returns 429 before any LLM work. Security middleware rejects oversized payloads (413) before deserialization. (File: `main.py` L177-200, L225-231)
- **(INFERENCE)** **Autoptic translation:** The circuit breaker + semaphore pattern maps directly to Autoptic's resilience-agent model — detecting degraded integrations, avoiding cascading latency, and maintaining SLOs even when downstream DevOps tools (10+ integrations) experience brownouts.

---

#### Safety levers

- **(EVIDENCE)** **Multi-layer input sanitization:** `sanitize_text()` applies HTML escaping, script-tag removal, control-character stripping, whitespace normalization. (File: `core/validators.py` L11-39, search token: `html.escape(text)`)
- **(EVIDENCE)** **Suspicious-pattern rejection:** `validate_text()` blocks `javascript:`, `data:`, `vbscript:`, `<iframe`, `<object`, `<embed` patterns. (File: `core/validators.py` L70-81)
- **(EVIDENCE)** **Prompt injection detection (22 patterns, 4 severity levels):** `PromptInjectionDetector.detect_injection()` matches patterns like `ignore.*previous.*instructions` (critical), `you.*are.*now.*a` (high). Risk score = `confidence * severity_weight`, normalized. Critical violations raise `SecurityError` and halt processing. (File: `core/security.py` L32-63, L76-129)
- **(EVIDENCE)** **Adversarial attack detection:** `AdversarialAttackDetector` catches contradictory instructions, label manipulation attempts, confidence manipulation, emotional manipulation. Includes confidence-label mismatch heuristic. (File: `core/security.py` L156-266)
- **(EVIDENCE)** **LLM response sanitization:** `_sanitize_response()` strips HTML tags and truncates to 10 000 chars. (File: `agents/base_agent.py` L353-366)
- **(EVIDENCE)** **Pydantic schema validation at API boundary:** `ReviewRequest` enforces `max_length=10000` on text, `ge=0.0, le=1.0` on confidence, custom validators call `validate_text()` / `validate_label()` / `validate_confidence()`. (File: `main.py` L63-103)
- **(EVIDENCE)** **Security headers in every response:** HSTS, CSP `default-src 'self'`, X-Frame-Options DENY, X-XSS-Protection, nosniff. (File: `main.py` L243-248)
- **(EVIDENCE)** **Drift detection for behavioral anomalies:** `DriftDetector` tracks `avg_tokens`, `avg_latency`, `success_rate`, verdict distribution over sliding window (100 responses). Drift threshold = 10%. (File: `core/security.py` L396-472)
- **(EVIDENCE)** **API key validation:** Minimum 16 chars, alphanumeric + `_-` only, generated via `secrets.token_urlsafe(32)`. (File: `core/validators.py` L383-410, `core/config.py` L268-270)
- **(INFERENCE)** **Autoptic translation:** The layered security model (sanitize -> validate schema -> detect injection -> detect adversarial -> validate output -> detect drift) mirrors Autoptic's guardrails philosophy around deterministic entities constraining AI behavior. The drift detector is a lightweight analog to Autoptic's CFR-monitoring loop — detecting when the system's own behavior degrades before it impacts production.

---

#### Key code snippets

**Snippet 1 — Circuit Breaker + Exponential Backoff (LLM Connector)**
```python
# agents/base_agent.py L147-175 (search: "circuit_breaker:{self.ollama_url}")
cache_key = f"circuit_breaker:{self.ollama_url}"
circuit_state = self.cache.get(cache_key)

if circuit_state:
    state = circuit_state.get("state", "closed")
    failure_count = circuit_state.get("failure_count", 0)
    current_time = time.time()

    if state == "open":
        timeout_seconds = min(60 * (2 ** min(failure_count, 3)), 300)
        if current_time - last_failure > timeout_seconds:
            self.cache.set(cache_key, {
                "state": "half_open",
                "last_failure": last_failure,
                "failure_count": failure_count,
                "success_count": 0
            }, ttl=600)
```

**Snippet 2 — Cache-Before-LLM + Token Budget**
```python
# agents/base_agent.py L416-438 (search: "Cache hit for prompt hash:")
prompt, cache_key, prompt_hash = self._prepare_llm_call(
    prompt_template, context, expected_fields)

if config.performance.enable_caching:
    cached_result = self.cache.get(cache_key)
    if cached_result:
        logger.debug(f"Cache hit for prompt hash: {prompt_hash}")
        return cached_result

result = self._call_ollama(prompt)  # num_predict=max_tokens=512

if config.performance.enable_caching:
    self.cache.set(
        cache_key, result,
        ttl=config.performance.cache_ttl_seconds)  # default 3600s
```

**Snippet 3 — Semaphore-Bounded Batch Processing**
```python
# agents/unified_agent.py L194-220 (search: "asyncio.Semaphore(max_concurrent)")
async def process_batch(self, samples: List[Dict[str, Any]],
                        max_concurrent: int = 5) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_sample(sample: Dict[str, Any]):
        async with semaphore:
            return await self.process_sample(
                sample["text"],
                sample["predicted_label"],
                sample["confidence"])

    tasks = [process_single_sample(s) for s in samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

---

#### (INFERENCE) Autoptic Translation: BYOC Realism + Observability Spend Control

This repository demonstrates a **local-first, BYOC-compatible architecture** (Ollama on localhost, SQLite for persistence, no cloud dependencies) with cost/performance discipline that maps to several Autoptic pillars:

1. **Observability cost containment** — The LLM-call cache (`AdvancedCache`) with memory budgets and LRU eviction directly parallels controlling telemetry query costs. The multi-task batching pattern (`process_multi_task`) is analogous to combining multiple diagnostic PQL queries into a single inference pass.

2. **MTTR reduction via circuit breaker** — The 3-state circuit breaker (`open/half_open/closed`) with exponential backoff prevents cascading failures when the LLM backend degrades, equivalent to Autoptic's resilience agents detecting and routing around unhealthy integrations.

3. **Deterministic guardrails constraining AI** — Pydantic schema validation, prompt-injection detection (22 patterns), adversarial-attack detection, and response sanitization form a defense-in-depth boundary around the LLM, mirroring Autoptic's principle of deterministic entities + guardrails ensuring AI outputs remain safe.

4. **Drift detection as incident avoidance** — `DriftDetector` monitoring verdict distribution shifts and latency changes over a sliding window is a lightweight analog to Autoptic's proactive CFR-reduction loop — detecting degradation before it becomes a production incident.

---

## 3. Evidence Ledger Rows

| # | Claim | Type | File | Symbol / Line | Search Token |
|---|-------|------|------|--------------|--------------|
| 1 | Circuit breaker with 3-state machine (open/half-open/closed) protects LLM connector | EVIDENCE | `agents/base_agent.py` | `_call_ollama()` L147-175 | `"circuit_breaker:{self.ollama_url}"` |
| 2 | Exponential backoff on LLM retries: `delay * 2^attempt`, max 3 retries | EVIDENCE | `agents/base_agent.py` | `_call_ollama()` L236-238 | `config.llm.retry_delay_seconds * (2 ** attempt)` |
| 3 | Async LLM client via `aiohttp` with configurable timeout | EVIDENCE | `agents/base_agent.py` | `_call_ollama_async()` L273-278 | `aiohttp.ClientTimeout(total=` |
| 4 | LLM response caching eliminates redundant calls; key = MD5(prompt) | EVIDENCE | `agents/base_agent.py` | `_call_llm_common()` L418-438 | `"Cache hit for prompt hash:"` |
| 5 | In-memory LRU cache with TTL, 100 MB budget, 10k entry cap, pickle persistence | EVIDENCE | `core/cache.py` | `AdvancedCache` L30-38 | `class AdvancedCache:` |
| 6 | LRU eviction sorted by `last_accessed` when cache budget exceeded | EVIDENCE | `core/cache.py` | `_evict_entries()` L219-238 | `sorted_entries = sorted(` |
| 7 | Thread-safe cache operations via `threading.Lock` | EVIDENCE | `core/cache.py` | `AdvancedCache.__init__()` L45 | `self.lock = Lock()` |
| 8 | Token limit per LLM call: `max_tokens=512`, configurable via `AR_MAX_TOKENS` | EVIDENCE | `core/config.py` | `LLMConfig.max_tokens` L19 | `max_tokens: int = 512` |
| 9 | Concurrency limit: `max_concurrent_requests=5`, configurable via `AR_MAX_CONCURRENT` | EVIDENCE | `core/config.py` | `PerformanceConfig` L128 | `max_concurrent_requests: int = 5` |
| 10 | Rate limiting: 100 req/hr sliding window, returns 429 with Retry-After | EVIDENCE | `main.py` | `rate_limit_check()` L177-200 | `HTTP_429_TOO_MANY_REQUESTS` |
| 11 | Request-size guard: 1 MB max, returns 413 | EVIDENCE | `main.py` | `security_middleware()` L225-231 | `HTTP_413_REQUEST_ENTITY_TOO_LARGE` |
| 12 | Security headers: HSTS, CSP, X-Frame-Options, nosniff, XSS-Protection | EVIDENCE | `main.py` | `security_middleware()` L243-248 | `X-Content-Type-Options` |
| 13 | Pydantic schema validation: text max 10k chars, confidence [0,1], custom validators | EVIDENCE | `main.py` | `ReviewRequest` L63-103 | `class ReviewRequest(BaseModel):` |
| 14 | Input sanitization: HTML escape + script removal + control-char strip | EVIDENCE | `core/validators.py` | `sanitize_text()` L11-39 | `html.escape(text)` |
| 15 | Suspicious-pattern rejection: javascript:, data:, vbscript:, iframe, object, embed | EVIDENCE | `core/validators.py` | `validate_text()` L70-81 | `suspicious_patterns` |
| 16 | Prompt injection detection: 22 regex patterns, 4 severity levels (critical/high/medium/low) | EVIDENCE | `core/security.py` | `PromptInjectionDetector` L28-74 | `self.injection_patterns` |
| 17 | Adversarial attack detection: contradictory instructions, label/confidence manipulation | EVIDENCE | `core/security.py` | `AdversarialAttackDetector` L156-266 | `self.adversarial_patterns` |
| 18 | Risk score computation: `confidence * severity_weight`, normalized | EVIDENCE | `core/security.py` | `SecurityManager._calculate_risk_score()` L566-587 | `severity_weights` |
| 19 | LLM response sanitization: HTML strip + 10k char truncation | EVIDENCE | `agents/base_agent.py` | `_sanitize_response()` L353-366 | `def _sanitize_response` |
| 20 | Drift detection: sliding window (100 responses), 10% change threshold on tokens/latency/verdicts | EVIDENCE | `core/security.py` | `DriftDetector.detect_drift()` L415-450 | `self.drift_threshold = 0.1` |
| 21 | Multi-task batching: 3 agent tasks in 1 LLM call | EVIDENCE | `agents/base_agent.py` | `process_multi_task()` L482-499 | `_create_multi_task_prompt` |
| 22 | Semaphore-bounded batch concurrency: `asyncio.Semaphore(max_concurrent)` | EVIDENCE | `agents/unified_agent.py` | `process_batch()` L208 | `asyncio.Semaphore(max_concurrent)` |
| 23 | System metrics via psutil: CPU%, memory%, disk% | EVIDENCE | `core/monitoring.py` | `HealthChecker.get_system_metrics()` L46-66 | `psutil.cpu_percent(interval=1)` |
| 24 | Application metrics: total_requests, success_rate, avg_response_time, llm_calls, llm_errors | EVIDENCE | `core/monitoring.py` | `ApplicationMetrics` L26-35 | `class ApplicationMetrics:` |
| 25 | Health threshold: CPU <90%, memory <90%, disk <95%, success_rate >95%, latency <30s | EVIDENCE | `core/monitoring.py` | `HealthChecker.is_healthy()` L145-167 | `system_metrics.cpu_percent < 90` |
| 26 | Token-usage audit logging in SQLite | EVIDENCE | `core/logger.py` | `AuditLogger` | `tokens_used INTEGER` |
| 27 | Bearer-token API authentication with min 16-char key, generated via `secrets.token_urlsafe(32)` | EVIDENCE | `main.py` L142-174, `core/config.py` L268-270 | `verify_api_key()` | `HTTPBearer(auto_error=False)` |
| 28 | Cross-dependency validation: cache_max_entries <100k, auth requires key | EVIDENCE | `core/config.py` | `SystemConfig._validate_cross_dependencies()` L241-266 | `"Cache max entries too high for memory safety"` |
| 29 | Cache dedup maps to Autoptic observability-cost containment (PQL dedup) | INFERENCE | — | — | — |
| 30 | Circuit breaker maps to Autoptic resilience-agent integration protection | INFERENCE | — | — | — |
| 31 | Drift detector maps to Autoptic proactive CFR-reduction loop | INFERENCE | — | — | — |
| 32 | Guardrail stack maps to Autoptic deterministic-entities-constraining-AI principle | INFERENCE | — | — | — |
