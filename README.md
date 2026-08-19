# Enterprise Agentic ESG Intelligence Platform

A Retrieval-Augmented Generation platform over Indonesian banking sustainability disclosures, built to demonstrate production-grade RAG engineering.

## Overview

The corpus is two domains, kept strictly separate at query time:

| Domain | Payload tag | Contents | Chunks |
|---|---|---|---|
| Company disclosures | `company_documents` | 21 Indonesian banks' IDX ESG Disclosure Guide Book (EDGB) filings | 1,292 |
| ESG standards | `standards` | GRI 1/2/3, the EDGB, Guidelines for Banks | 349 |

All 1,641 chunks live in one Qdrant collection (`esg_documents`), with `domain` as an indexed, filterable payload field: single-domain and cross-domain retrieval are the same operation with a different filter, not two different code paths.

The system answers three distinct question shapes, each through its own pipeline:
- **Chat**: a grounded question about one report or standard ("what does BCA report for Scope 1 emissions"), served by a bounded agentic retrieve→generate→judge loop.
- **Compliance-check**: a requirement-by-requirement comparison ("does BCA's report meet GRI 305"), served by a fully deterministic checklist-anchored pipeline.
- **Rank**: a cross-bank comparison on a disclosed metric or requirement ("which banks have the earliest net-zero target"), served by the metrics/compliance-sweep machinery fanned out across all 21 banks.

This is a portfolio project aimed at a technical hiring audience: the point is to show the reasoning behind retrieval design, where agentic control flow earns its cost, and how the system was evaluated and corrected, not to ship a general-purpose ESG chatbot. The 21-bank corpus is fixed and curated; there is no arbitrary document upload in v1.

## Architecture

Three decisions carry the design, each detailed below along with the mistakes that led to it.

### 1. Bounded agentic retry loop (chat pipeline)

After generation, the model performs a groundedness self-check against its own retrieved context and returns **sufficient** (stop) or **insufficient** with a rewritten query (retry). This is genuinely agentic: control flow branches on the model's own judgment of its own answer, not on a keyword or similarity heuristic. It is also bounded: a hard cap of 2 retries (3 total attempts) is enforced in code, not by the model, and the model's only available action is "retrieve again with a rewritten query", no other tool, no domain outside the existing two.

Most queries resolve in 0–1 iterations. Iteration count and stop reason are logged as Langfuse trace metadata on every request, read from the same values the loop uses internally (not a parallel computation): a live trace shows `attempts=1 retries=0 stop_reason=sufficient` in both the log line and the trace, and a drift toward most requests hitting the cap is the signal something upstream needs attention.

### 2. Deterministic compliance-check pipeline: EDGB→GRI anchoring, not similarity ranking

The compliance-check pipeline is deliberately **not** agentic. A compliance answer's correctness depends on coverage: a requirement silently missed is a correctness bug, so "what to check" comes from a curated checklist enumerated directly in code, never from a similarity search that could fail to surface an item.

**Why the anchoring exists at all: a real degeneracy finding.** The corpus turned out to be disclosure-index filings keyed by EDGB codes (`E-01`, `E-02`, …), not GRI-coded narrative reports; GRI codes appear nowhere as row labels. A GRI-coded checklist item therefore has no literal match in the source tables, so the original score-band assessor fell back to fuzzy topic-keyword similarity, and that similarity was driven by shared EDGB template text, not by what each bank actually disclosed. Every bank scored within about 0.03 of every other bank on the same requirement; on a real case, OCBC reporting `0` scored higher than BCA reporting `1.813` on the same requirement.

The fix, `row_anchors`, curates a literal EDGB code or row label per GRI requirement, matched against a candidate row's *first cell only* (an EDGB code appears in the corpus twice with different meanings: once as an index-table page reference, once leading the actual value row, and a naive substring match confuses the two). Anchors are tried most-specific-first and all retrieved chunks are scanned, not just the top hit.

Measured before/after on all 21 banks, environmental checklist:

| requirement | anchor(s) | before (score bands) | after (row-anchored) |
|---|---|---|---|
| GRI 305-3 (Scope 3) | none, genuinely unanchorable | 21/21 "addressed" (false, no Scope 3 row exists in this corpus) | 21/21 "not directly disclosed" |
| GRI 306-3 (waste) | `E-05` | 21/21 "partial" (a threshold accident, every bank scored 0.4398–0.4476, straddling the 0.35–0.50 band) | 21/21 "addressed" (every bank's real waste row read directly) |

Per-requirement retrieval still gets one corrective retry, but the trigger is a numeric similarity-score threshold, not an LLM judgment: a compliance query fans out to up to 20–30 requirements, so a per-item model judgment would multiply that pipeline's cost the way the chat loop's single per-turn judgment doesn't. This is the deliberate contrast between the two pipelines: the chat loop is genuinely model-driven and bounded; the compliance pipeline is genuinely model-free in its control flow.

A follow-on finding from the same anchoring work, measured on the environmental checklist's 6 anchored numeric requirements (the table above shows two of those seven rows; GRI 305-3 is the one deliberately-unanchorable exception): because these are mandated disclosure-index filings, every one of those 6 reads 21/21 "addressed" for every bank once anchored correctly, a uniform answer by design, given the corpus. This generalizes as a conclusion for numeric requirements on a mandated disclosure index, but the equivalent before/after measurement for social/governance numeric items hasn't been separately measured, so what's presented here is the environmental-checklist result plus that reasoning, not something independently re-verified across all three checklist categories. The compliance sweep therefore cannot rank banks on numeric requirements by design; ranking on those is routed to the metrics pipeline instead, which extracts and compares actual values.

### 3. Multimodal ingestion split: tables and charts never share a path

Ingestion routes visual content down two paths that never cross. Tables go through `pdfplumber` structured extraction (`content_type=table`); charts, diagrams, and infographics are rendered as full-page images (PyMuPDF) and sent to vision captioning (`gpt-4.1-mini`, `content_type=chart_caption`). The split exists because a dedicated table parser reads digits reliably, while a vision model asked to transcribe a financial or emissions table can misread them: keeping tables off the vision path removes that risk by construction.

The split was tested against a real failure, not assumed correct: at pilot scale, the captioner misclassified dense bordered tables as charts, hitting 10 of 20 `chart_caption` chunks (50%), 3 of which also restated real numeric values through the vision path, exactly the risk the split exists to prevent. Fixed with an explicit prompt rule ("a bordered/gridded/multi-column table is a table, not a graphic") and verified afterward at **zero false positives across the full corpus**, including 419 genuinely table-dense company-report pages that produced no captions.

## Evaluation results

### RAGAS

Most recent full run (`gpt-4.1-mini` evaluator): 34/34 samples scored, 7 skipped (pipeline-not-wired or refusal-path cases: questions that route through `/rank` or `/compliance-check`, or are decided pre-retrieval, out of scope for this harness).

| metric | score |
|---|---|
| faithfulness | 0.941 |
| answer_relevancy | 0.862 |
| context_precision | 0.620 |
| context_recall | 0.873 |

One caveat, stated plainly: a single question (`xd-001`) scored an exact `answer_relevancy=0.000` in this run despite a substantive, correctly-cited answer, traced to RAGAS's own `answer_relevancy` implementation, which zeroes the whole score if any one of three internal "is this answer noncommittal" sub-calls false-positives, independent of the answer's actual cosine similarity to the question. The mechanism is confirmed (read directly from RAGAS's source); the specific run is not reproducible: five repeat runs of the same question all landed in a 0.660–0.690 band instead.

### Custom eval runner

A second, purpose-built harness (`backend/evals/eval_runner.py`) scores three independent axes per question: **routing** (did it hit the right pipeline), **retrieval** (did it surface the expected source page), and **answer** (did the generated answer match ground truth), against a 41-question hand-built set (`eval_set.json`), including numeric-tolerant matching (`NUMEQ`, e.g. `6.032` vs `6,032`) and a `DIVERGENT`/`PASS_DIVERGENT` verdict for questions the system is *expected* to disagree with a naive reading of the source (e.g. a case where the correct answer is the source PDF's own inconsistent figure, not a "corrected" one).

Latest measured breakdown:

| axis | pass | fail | numeq | skip | diverged |
|---|---|---|---|---|---|
| routing | 34 | 1 | – | 6 | – |
| answer | 23 | 5 | 8 | 4 | 1 |

Retrieval improved from 31 to 32 pass after a keyword-leg-first tie-break fix (`cd-016` flipped FAIL → PASS, zero regressions); the full post-fix fail/numeq/skip/diverged split for that axis wasn't separately tracked, so it isn't reproduced here.

These are dated snapshots, not fixed final scores. Specific questions (`cd-005`, `xd-001`, `mt-001#t1`) have been observed flipping pass/fail across otherwise-identical live-pipeline runs, mostly numeral-format non-determinism (`6.032` vs `6,032`) or intermittent retrieval; these are tracked as known flaky cases rather than smoothed into the reported numbers.

## Stack & hosting

| Layer | Choice |
|---|---|
| Backend | FastAPI, Python 3.11+, async |
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind |
| Vector store | Qdrant Cloud, single collection, domain as an indexed payload filter |
| Auth | Supabase: JWT verified locally against its published JWKS (no DB queries); Supabase Storage for source page images |
| Embeddings / generation | OpenAI (`text-embedding-3-small`; `gpt-4.1-mini` default, `gpt-4.1` opt-in "high accuracy" toggle) |
| Hybrid search | In-house dense + BM25, fused by RRF |
| Re-ranking | Cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, multilingual, since the corpus is substantially Bahasa Indonesia), served torch-free via a quantized ONNX export |
| Observability | Langfuse: one trace per request, per-call cost and token counts, agentic-loop iteration count as trace metadata |
| Evaluation | RAGAS (isolated venv: its LangChain dependency chain caps `openai<2`, incompatible with the app's `openai` 2.x) + the custom `eval_runner.py` harness above |
| Deploy | Railway (backend), Vercel (frontend) |

Backend on Railway, frontend on Vercel, Qdrant Cloud, and Supabase's free tier were chosen as managed PaaS across the board: no infrastructure to provision or patch, deploy-on-push from a single repo, and low operational overhead appropriate for a solo-built project at this scale. Both platforms' auto-deploy-on-push behavior is path-filtered (`vercel.json` / `railway.json`) so a docs-only commit doesn't rebuild either service: verified empirically with three real test commits, each triggering exactly the platform the config predicted and neither for the docs-only case.

## Setup / local development

**Prerequisites:** Python 3.11+, Node 18+, Docker (for local Qdrant), an OpenAI API key, and a Supabase project (for auth).

```bash
# 1. Environment
cp .env.example .env
# fill in OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY

# 2. Vector DB (local Qdrant via Docker)
docker compose up qdrant

# 3. Backend
cd backend
python -m venv .venv && .venv\Scripts\activate    # Windows; use source .venv/bin/activate on macOS/Linux
pip install -e .[dev]
pytest -q                                          # backend test suite
uvicorn app.main:app --reload                      # http://localhost:8000

# 4. Frontend (separate terminal, from repo root)
cd frontend
npm install
npm run dev                                        # http://localhost:3005 (pinned, see .env.example for why)
```

**Running the evals:**

```bash
# Custom harness (routing / retrieval / answer axes)
python -m evals.eval_runner

# RAGAS, deliberately isolated: its LangChain pin would downgrade the app's openai SDK
# by a major version if installed into the same environment. See evals/requirements-ragas.txt
# for the full reasoning and setup steps.
python -m venv <path-outside-the-repo>/esg-ragas
<path>/esg-ragas/Scripts/python.exe -m pip install -r backend/evals/requirements-ragas.txt
<path>/esg-ragas/Scripts/python.exe -m evals.ragas_eval
```

**Docker (full stack):** `docker compose up` builds and runs the backend against `backend/Dockerfile` alongside Qdrant. See `docker-compose.yml` for how to point the backend container at local vs. cloud Qdrant.

## Live demo

**https://esg-rag-platform.vercel.app/chat**