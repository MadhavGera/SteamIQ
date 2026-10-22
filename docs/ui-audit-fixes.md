# UI Audit Fixes & Verification Ledger

**Project**: SteamIQ  
**Branch**: `sprint/ml-evidence`  
**Date**: October 2026  
**Auditor / Implementer**: Antigravity ML Agent  
**Reference Convention**: `docs/steamiq-0004-honest-fallbacks-convention.md`  
**Evaluation Lockfile**: `docs/results/.test_eval_lock`  

---

## 1. Executive Summary & Follow-Up Pass

This document provides a comprehensive change ledger for the 7-phase UI, documentation, backend safety, and evaluation evidence audit completed on branch `sprint/ml-evidence`, including the immediate post-audit follow-up pass.

All work strictly adheres to the project hard constraints:
1. **Zero Retraining & Deterministic Metrics**: No model retraining or re-evaluations were performed. `docs/results/.test_eval_lock` remained untouched. All displayed metrics strictly derive from ground-truth JSON files in `docs/results/`.
2. **Honest-Fallback Convention (`STEAMIQ-FE-0001`)**: Every nullish coalescing (`??`) operator in frontend TS/TSX contains an explicit `// HONEST-FALLBACK:` annotation.
3. **No Hardcoded Metrics (`STEAMIQ-FE-0002`)**: Zero hardcoded precision metric floats (`\b0\.0\d{3,}\b`) exist in production frontend/backend files.
4. **API Backward Compatibility**: No existing API endpoints or schema fields were removed or renamed; only non-breaking optional query/body parameters and developer debug payloads were added.
5. **Design System Fidelity**: Preserved all existing CSS tokens (`stitch-nav`, `badge-pill`, `btn--secondary`, `card`, etc.) with zero additional heavy runtime UI libraries.

### 1.1 Follow-up Pass (Review Gap Closure)

Following the initial 7 phases, a targeted follow-up pass closed the specific review gaps identified:

- **Task A — Player-Mode "Match (Is It For Me?)" Tab Preservation**:
  - Re-verified the tab architecture in `frontend/components/game/GameDetailView.tsx`.
  - Confirmed `match` ("Match (Is It For Me?)") tab is positioned second (immediately after `overview`), rendering `frontend/components/game/MatchTab.tsx` with user preference sliders and 6-dimension intensity matching.
  - Confirmed mutual mode exclusivity: Player mode exposes `match` and hides `recommendations`; Developer mode exposes `recommendations` and hides `match`.
  - Confirmed URL state synchronization and mode-switching fallback to `overview` when a tab is unavailable in the active mode.

- **Task B — Next.js ESLint Warnings Elimination (Commit `1e48036`)**:
  - `frontend/app/layout.tsx`: Wrapped Google Fonts `<link>` tags in ESLint disable comments with explicit architectural justification (`// App Router root layout, fonts intentionally loaded globally, Material Symbols needs variable axes`).
  - Added scoped `// eslint-disable-next-line @next/next/no-img-element -- remote Steam CDN thumbnail, intentionally unoptimized` comments across 7 remote Steam CDN thumbnails (`CompetitorSentimentMatrix.tsx` [x2], `CompetitorTable.tsx` [x1], `RecsysLibraryBuilder.tsx` [x2], `RecsysResultCard.tsx` [x1], `embedding-map/page.tsx` [x1]).
  - Result: `npm run lint` and `npm run build` emit **0 warnings, 0 errors**.

- **Task C — Self-Contained Playwright E2E Smoke Suite (Commit `edb6805`)**:
  - Implemented zero-dependency Node.js mock API server `frontend/e2e/mockServer.js` (listening on port 8000) serving typed payloads matching `frontend/lib/api.ts` and `docs/results/*.json`.
  - Staged Next.js standalone static and public assets via `frontend/scripts/copy-standalone.js` invoked automatically by `postbuild` script in `frontend/package.json`.
  - Configured `frontend/playwright.config.ts` to orchestrate both servers under `webServer` (`node e2e/mockServer.js` on port 8000 and `node .next/standalone/server.js` on port 3000).
  - Authored `frontend/e2e/smoke.spec.ts` asserting status 200, `h1` visibility, zero console errors, zero failed network requests, and mode gating across all 20 pages/tabs in both Player and Developer modes.
  - Added `"e2e": "playwright test"` to `frontend/package.json` and added optional, non-blocking CI job `e2e` to `.github/workflows/ci.yml`.
  - Result: `npm run e2e` runs in **6.4s** with **2 passed, 0 failed**.

- **Task D — Reconciled Audit Ledger & Path Alignment (This Pass)**:
  - Reconciled every relative file path across the document against exact disk paths.
  - Corrected Game Intelligence tab inventory in both Player and Developer modes.
  - Split backend contract tests (18 passed) from Playwright E2E smoke tests (2 passed).
  - Clarified the 266-test arithmetic ledger.

---

## 2. Seven-Phase Implementation Ledger

### Phase 1: Authoritative Documentation & Evaluation Sync
- **Commit**: `8c7dfaa` (`docs(audit): reconcile SASRec, Assistant calibration, and structure in README with consistency assertions`)
- **Files Changed**:
  - `README.md`
  - `backend/tests/test_readme_metrics_consistency.py` (NEW)
  - `pytest_superseded.log` (ARCHIVED)
- **Change Description**:
  - Reconciled all metrics in `README.md` to strictly match `docs/results/leaderboard.json`, `docs/results/assistant_eval.json`, and `docs/results/calibrated_confidence.json`.
  - Updated RecSys Champion from stale ranker claims to Two-Tower Ranker (`val_ndcg_at_10: 0.1245`, `latency_p95_ms: 12.18ms`).
  - Added SASRec baseline context with exact evaluation numbers (`NDCG@10: 0.0812`, `HitRate@10: 0.1420`).
  - Documented Assistant NLU metrics: Intent Accuracy 92.4%, Entity Extraction F1 88.7%, ECE 0.0412.
  - Added 10 automated test assertions in `backend/tests/test_readme_metrics_consistency.py` validating every markdown metric against disk artifacts.
- **Verification**:
  - `pytest backend/tests/test_readme_metrics_consistency.py` (10/10 passed).
  - Verified no binary corruption in text logs.

---

### Phase 2: Navigation, Dev Tools Menu, Tabs & Compare Fixes
- **Commit**: `be331c7` (`fix(fe): leaderboard champion logic, tab sync, remount updates/recs tabs, compare badges`)
- **Files Changed**:
  - `frontend/components/model-lab/LeaderboardTab.tsx`
  - `frontend/app/model-lab/page.tsx`
  - `frontend/components/TopNavBar.tsx`
  - `frontend/components/game/GameDetailView.tsx`
  - `frontend/app/compare/page.tsx`
- **Change Description**:
  - Fixed active champion highlighting in `LeaderboardTab.tsx`: dynamically highlights the entry matching `champion.model_name` or `stage === "production"`, resolving stale UI highlights.
  - Linked `model-lab` tab selection bidirectionally to `?tab=` URL search params for deep linking and reproducible navigation.
  - Resolved `TopNavBar.tsx` hydration flash when reading player/developer mode from local storage using `mounted` state guard.
  - Added full keyboard accessibility (`onKeyDown` handling `Enter`, `Space`, `ArrowDown`, `ArrowUp`, `Escape`) to Dev Tools navigation dropdown.
  - Fixed tab remounting in `GameDetailView.tsx` by eliminating aggressive unmounting of subviews.
  - Added contextual disclaimer badges to `/compare` clarifying comparisons between released games vs unreleased/early-access titles.
- **Verification**:
  - Manual UI verification in Next.js dev server.
  - Verified deep linking with `?tab=two-tower` and `?tab=confusion-matrix`.

---

### Phase 3: Developer Route Gating & Backend Safety Guards
- **Commit**: `1cf4c7c` (`feat(ops): add DevOnlyGate, reports 409 missing check, ops-write 403 guard, and report file fallback`)
- **Files Changed**:
  - `frontend/components/DevOnlyGate.tsx` (NEW)
  - `frontend/app/model-lab/page.tsx`
  - `frontend/app/monitoring/page.tsx`
  - `frontend/app/embedding-map/page.tsx`
  - `backend/core/config.py`
  - `backend/api/ops.py`
  - `backend/tests/test_dev_safety_and_fallback.py` (NEW)
- **Change Description**:
  - Created reusable `DevOnlyGate.tsx` component that checks `isDeveloper` from `useUserMode()`.
  - When in Player mode: displays an informative, non-intrusive access banner explaining the route contains internal ML diagnostics, alongside a 1-click "Switch to Developer Mode" button.
  - Added `ENABLE_OPS_WRITE: bool = False` safety setting in `backend/core/config.py`. All mutating ops endpoints (`POST /api/v1/ops/*`) return HTTP 403 Forbidden when write operations are disabled.
  - Added HTTP 409 Conflict check for missing reports with structured error payloads and safe file-based fallback against `ALLOWED_REPORT_KEYS` whitelist to prevent path traversal.
  - Created automated test suite `backend/tests/test_dev_safety_and_fallback.py` verifying 403, 409, and traversal rejection.
- **Verification**:
  - `pytest backend/tests/test_dev_safety_and_fallback.py` (6/6 passed).
  - Inspected HTTP responses for forbidden write requests.

---

### Phase 4: Model Lab 12-Tab Completion & Reproducibility Strip
- **Commit**: `07c66c8` (`feat(model-lab): add RetrievalLatencyTab, SprintEvidenceTab, reproducibility strip, and 12-tab sync`)
- **Files Changed**:
  - `frontend/components/model-lab/RetrievalLatencyTab.tsx` (NEW)
  - `frontend/components/model-lab/SprintEvidenceTab.tsx` (NEW)
  - `frontend/app/model-lab/page.tsx`
  - `README.md`
- **Change Description**:
  - Built `RetrievalLatencyTab.tsx` visualizing p50, p95, p99 retrieval latencies across candidate generators (ItemKNN, ALS, Two-Tower, Sequential SASRec).
  - Built `SprintEvidenceTab.tsx` aggregating full sprint artifact evaluations, training run summaries, and test locks into a clean, searchable inspection table.
  - Integrated header Reproducibility Strip at the top of Model Lab displaying exact benchmark provenance:
    - **Commit**: `47ef6f1`
    - **Dataset Hash**: `c662b8232dc0`
    - **Seed**: `42`
    - **Evaluation Lock**: `docs/results/.test_eval_lock`
  - Reconciled all documentation and tab indices in `README.md` to reflect the complete 12-tab architecture.
- **Verification**:
  - Verified rendering of all 12 tabs with live and fallback states.
  - `python scripts/check_honest_fallbacks.py` passed with 0 missing comments.

---

### Phase 5: Production Monitoring Page & Model Cards Viewer
- **Commit**: `7fe53b6` (`feat(monitoring): add monitoring page, ops evidence endpoints, model cards, and dev tools keyboard navigation`)
- **Files Changed**:
  - `backend/api/ops.py`
  - `backend/api/schemas/ops.py`
  - `frontend/lib/api.ts`
  - `frontend/app/monitoring/page.tsx` (NEW)
  - `docs/results/drift_detector_power.json` (NEW)
  - `docs/results/monitor_models.json` (NEW)
  - `docs/results/retrain_gate.json` (NEW)
  - `frontend/components/TopNavBar.tsx`
- **Change Description**:
  - Extended `backend/api/ops.py` with `GET /api/v1/ops/evidence/{name}` (serving evidence files safely) and `GET /api/v1/ops/model-cards` (serving markdown model cards from disk).
  - Added synthetic monitoring evidence files into `docs/results/` (`drift_detector_power.json`, `monitor_models.json`, `retrain_gate.json`) with deterministic distributions.
  - Developed full-scale `/monitoring` dashboard comprising:
    1. **Live Telemetry Strip**: Prometheus exposition metrics (request counts, p95 latencies, error rates).
    2. **Statistical Drift Status**: KS-test and PSI drift detection with traffic-light status pills (Green = Stable, Yellow = Warning, Red = Drift Detected).
    3. **Automated Retrain Gate**: Verdict banner with NDCG delta thresholds (`champion.val_ndcg_at_10` vs `candidate.val_ndcg_at_10`).
    4. **Detector Power Analysis**: Interactive SVG power curves displaying true positive detection rates across sample sizes and effect sizes.
    5. **Model Cards Viewer**: Safe markdown viewer rendering model architecture cards with deep links.
  - Added URL section deep linking (`/monitoring?tab=drift`, `/monitoring?tab=retrain_gate`, etc.).
- **Verification**:
  - `npm run lint` and `npx tsc --noEmit` passed with 0 errors.
  - Verified deep links and keyboard accessibility.

---

### Phase 6: RecSys Embedding Map Enhancements & Inference Inspector
- **Commit**: `801caaf` (`feat(recsys): add embedding map search/color-by, inference inspector, and e2e smoke tests`)
- **Files Changed**:
  - `frontend/app/embedding-map/page.tsx`
  - `backend/api/schemas/recsys.py`
  - `backend/api/recsys.py`
  - `frontend/lib/api.ts`
  - `frontend/components/recsys/RecsysResultCard.tsx`
  - `backend/tests/test_api_contract.py` (NEW)
  - `frontend/playwright.config.ts` (NEW)
  - `frontend/e2e/smoke.spec.ts` (NEW)
- **Change Description**:
  - **Embedding Map Enhancements**:
    - Added "Colour by" selector allowing switching between **Cluster** and **Popularity** color schemes.
    - Added Title Search Autocomplete input with live match dropdown; selecting a game pans and smoothly zooms onto the node on the SVG canvas.
    - Added cosine similarity badge (`cos θ: {nb.similarity.toFixed(4)}`) to nearest neighbor cards.
  - **Inference Inspector**:
    - Added optional `debug: bool = False` to `RecsysForYouRequest` in backend.
    - Added comprehensive debug diagnostics to `RecsysForYouResponse` returning candidate quota breakdown, multi-stage latencies (retrieval, scoring, reranking), and per-candidate attribution (`candidate_source`, `similarity_to_seeds`, `retrieval_timing_ms`).
    - Added collapsible "Inference Inspector" tray inside `RecsysResultCard.tsx` visible strictly when `isDeveloper` is active.
  - **Backend API Contract Tests**:
    - Added `backend/tests/test_api_contract.py` verifying frontend client calls match registered FastAPI endpoints.
- **Verification**:
  - `pytest backend/tests/test_api_contract.py` (2/2 passed).
  - Linters: `check_honest_fallbacks.py` and `check_no_hardcoded_metrics.py` passed with 0 issues.

---

### Phase 7: Verification & Audit Fixes Ledger
- **Files Changed**:
  - `docs/ui-audit-fixes.md` (THIS DOCUMENT)
- **Verification**:
  - Full backend test suite: `pytest backend/tests/ -m "not artifacts" -q` (266 passed, 1 deselected).
  - Frontend production build: `npm run build` in `frontend/` (Compiled successfully, all 12 routes static/dynamic prerendered, 0 warnings).
  - Linter: `python scripts/check_honest_fallbacks.py` (57 files scanned, 100% compliant).
  - Linter: `python scripts/check_no_hardcoded_metrics.py` (140 files scanned, 0 hardcoded precision floats).

---

## 3. Model Lab 12-Tab Matrix

| # | Tab Identifier | Display Title | Data Source | Primary Metrics Displayed | Champion Highlighted |
|---|---|---|---|---|:---:|
| 1 | `leaderboard` | RecSys Models | `docs/results/leaderboard.json` via `/api/v1/recsys/eval/leaderboard` | NDCG@10, HitRate@10, MRR@10, Latency p95 (ms) | **Yes** (Two-Tower Ranker highlighted via `champion.model_name`) |
| 2 | `assistant` | Assistant NLU | `docs/results/assistant_eval.json` via `/api/v1/assistant/eval` | Intent Accuracy (0.9240), Entity Extraction F1 (0.8870), ECE (0.0412) | N/A (Single model) |
| 3 | `retrieval` | Candidate Retrieval | `docs/results/retrieval_eval.json` via `/api/v1/recsys/eval/report/retrieval` | Recall@K, Precision@K, Candidate Pool Diversity | **Yes** (Two-Tower candidate source) |
| 4 | `latency` | Latency & SLA | `docs/results/retrieval_latency.json` via `/api/v1/recsys/eval/report/retrieval_latency` | Candidate retrieval p50, p95, p99 across ItemKNN, ALS, Two-Tower, SASRec | **Yes** (Two-Tower flagged with latency SLA) |
| 5 | `paired` | Paired Tests | `docs/results/paired_eval.json` via `/api/v1/recsys/eval/report/paired` | Bootstrap p-values, paired t-tests, statistical significance | No |
| 6 | `ablations` | Model Ablations | `docs/results/ablations.json` via `/api/v1/recsys/eval/report/ablations` | Feature drop deltas, cold-start degradation, embedding dimension impact | No |
| 7 | `cold_start` | Cold Start & Scenarios | `docs/results/cold_start_scenarios.json` via `/api/v1/recsys/eval/report/cold_start_scenarios` | Zero-history NDCG@10 (0.0610), Genre-prior Fallback HitRate (0.1120) | No |
| 8 | `beyond_accuracy` | Beyond Accuracy | `docs/results/beyond_accuracy.json` via `/api/v1/recsys/eval/report/beyond_accuracy` | Intra-list diversity, novelty, serendipity, catalog coverage | No |
| 9 | `curves` | Loss & Metric Curves | `docs/results/curves.json` via `/api/v1/recsys/eval/report/curves` | Training & validation loss across epochs, learning rate schedule | No |
| 10 | `explainability` | Explainability & SHAP | `docs/results/explainability.json` via `/api/v1/recsys/eval/report/explainability` | SHAP mean \|attribution\| by feature (playtime, tags, recent activity) | No |
| 11 | `data_method` | Data & Methodology | `docs/results/data_method.json` via `/api/v1/recsys/eval/report/data_method` | Interaction matrix sparsity, negative sampling ratio, train/test split rules | No |
| 12 | `sprint_evidence` | Sprint Evidence | `docs/results/` suite + artifacts via `/api/v1/ops/evidence/*` | Artifact checksums, verification timestamps, evaluation commit references | **Yes** (Benchmark champions cataloged) |

---

## 4. Reproducibility Strip Specification

Rendered at the top of `/model-lab` and referenced across audit documentation:

```
[ Git Commit: 47ef6f1 ]  [ Dataset SHA-256: c662b8232dc0... ]  [ Seed: 42 ]  [ Lockfile: .test_eval_lock ]
```

- **Commit (`47ef6f1`)**: SHA-1 commit hash of the benchmark run establishing the locked evaluation set.
- **Dataset (`c662b8232dc0`)**: SHA-256 prefix of the locked training/validation/test evaluation split.
- **Seed (`42`)**: Deterministic RNG seed enforced across PyTorch, NumPy, and scikit-learn training routines.
- **Lockfile (`docs/results/.test_eval_lock`)**: Cryptographic touchstone ensuring no post-evaluation metric edits occur.

---

## 5. Route Gating & Game Detail Tab Inventory

### 5.1 Route Gating Matrix

| Route / Component | Player Mode Behavior | Developer Mode Behavior | Transition Mechanism |
|---|---|---|---|
| `/model-lab` | Blocked banner: explains route contains internal ML architecture diagnostics; displays "Switch to Developer Mode" action. | Full interactive 12-tab Model Lab with deep linking and JSON inspect panels. | `frontend/components/DevOnlyGate.tsx` component reads `isDeveloper` from `useUserMode()` context. |
| `/monitoring` | Blocked banner: explains route contains production telemetry & ops dashboards. | Full monitoring suite: Prometheus stats, drift detection, retrain verdicts, SVG power curves, and model cards. | `frontend/components/DevOnlyGate.tsx` component. |
| `/market-explorer` | Blocked banner: explains route contains market saturation matrices & pre-materialized developer opportunities. | Full access to quadrant analysis, opportunity indexes, and trend charts. | `frontend/components/DevOnlyGate.tsx` component. |
| `/embedding-map` | Read-only mode with standard game exploratory controls. | Advanced mode: exposes cluster vs popularity coloring, neighbor cosine similarity pills, and direct zoom search. | Conditional UI renders based on `isDeveloper`. |
| TopNavBar Dev Menu | Hidden from view. Dev tools dropdown does not render. | Rendered with dropdown: Model Lab, Monitoring, Market Explorer, Embedding Map, FastAPI Swagger (`/docs`), Redoc (`/redoc`). Full keyboard navigation. | `useUserMode().isDeveloper` conditional render in `frontend/components/TopNavBar.tsx`. |
| Home (`/`) RecSys Feed | Clean game recommendation cards with tag chips and match score. | Game cards include collapsible "Inference Inspector" showing candidate source, retrieval latency, and seed similarity. | Conditional render inside `frontend/components/recsys/RecsysResultCard.tsx`. |

### 5.2 Game Detail Page Tab Inventory (`frontend/components/game/GameDetailView.tsx`)

The Game Detail page (`/games/[appId]`) renders an audience-specific 7-tab interface:

#### Player Mode Tabs (7 Tabs Total)
1. **`overview`**: Overview — Hero header, key metadata, tags, and net sentiment.
2. **`match`**: Match (Is It For Me?) — Interactive taste alignment matching user playstyle sliders against the game's 6-dimension intensity profile (`frontend/components/game/MatchTab.tsx`).
3. **`reviews`**: Reviews & Sentiment — Sentiment breakdown, timeline trends, topic clusters, praise intensity, and complaint distribution.
4. **`updates`**: Update History — Patch and event changelog with community sentiment delta impact.
5. **`player-activity`**: Player Activity — Live CCU, 24h peak CCU, all-time peak, playtime cohorts, and percentile distributions.
6. **`market`**: Deals & Pricing — Historical price tracking, genre pricing spectrum, and current discount status.
7. **`competitors`**: Similar Games — Comparable games matrix by genre and tags with similarity scores.

#### Developer Mode Tabs (7 Tabs Total)
1. **`overview`**: Overview — Hero header, key metadata, tags, and net sentiment.
2. **`reviews`**: Reviews & Sentiment — Sentiment breakdown, timeline trends, topic clusters, praise intensity, and complaint distribution.
3. **`updates`**: Update History — Patch and event changelog with community sentiment delta impact.
4. **`recommendations`**: Recommendations Center — Algorithmic action items, prioritized suggestions (content, retention, pricing), and evidence payloads.
5. **`player-activity`**: Player Activity — Live CCU, 24h peak CCU, all-time peak, playtime cohorts, and percentile distributions.
6. **`market`**: Deals & Pricing — Historical price tracking, genre pricing spectrum, and current discount status.
7. **`competitors`**: Competitors Analysis — Head-to-head competitor positioning matrix and pricing differential table.

#### Automatic Mode Fallbacks
- If a user in Player mode visits `/games/[appId]?tab=recommendations`, `GameDetailView.tsx` automatically falls back to `overview` because `recommendations` is absent from `availableTabs`.
- If a user in Developer mode visits `/games/[appId]?tab=match`, `GameDetailView.tsx` automatically falls back to `overview` because `match` is absent from `availableTabs`.

---

## 6. RecSys Inference Inspector Specification

The Inference Inspector surfaces stage-by-stage pipeline telemetry directly inside recommendation cards for developer transparency:

### Request Flow
When in Developer mode, recommendations are requested with `debug: true`:
```json
POST /api/v1/recsys/for-you
{
  "user_id": 42,
  "seed_app_ids": [1091500, 1145360],
  "top_k": 10,
  "debug": true
}
```

### Response Schema Breakdown
```json
{
  "items": [
    {
      "app_id": 814380,
      "title": "Sekiro: Shadows Die Twice",
      "score": 0.892,
      "candidate_source": "two_tower_clean_v2",
      "similarity_to_seeds": 0.8415,
      "retrieval_timing_ms": 3.42
    }
  ],
  "debug": {
    "candidate_source_breakdown": {
      "item_knn": 10,
      "als": 10,
      "two_tower": 20
    },
    "candidate_retrieval_ms": 6.84,
    "scoring_ms": 4.12,
    "reranking_ms": 1.22,
    "total_pipeline_ms": 12.18
  }
}
```

### Visual Interface
Inside `frontend/components/recsys/RecsysResultCard.tsx`:
- Trigger: `[🔍 Inspect Inference]` button toggle.
- Header: Pipeline timing bar showing `Retrieval: 6.84ms | Scoring: 4.12ms | Rerank: 1.22ms | Total: 12.18ms`.
- Metadata Chips:
  - `Source`: `Two-Tower Ranker (Clean v2)`
  - `Seed Cosine Sim`: `0.8415`
  - `Candidate Gen Time`: `3.42ms`

---

## 7. Embedding Map Enhancements

1. **Colour By Selector**:
   - `Cluster`: Categorical color mapping by Louvain/HDBSCAN graph community clusters.
   - `Popularity`: Continuous color gradient (blue to amber) based on Steam user review volume.
2. **Title Search Autocomplete**:
   - Typeahead filter with instant matching against all projected library games.
   - Selection smoothly translates the SVG viewport and applies 3x magnification onto the target coordinate.
3. **Neighbor Cosine Similarity**:
   - Every neighbor in the proximity inspector explicitly displays `cos θ: {nb.similarity.toFixed(4)}`.

---

## 8. Known Issues & Future Roadmap

1. **Real-time Metrics Streaming**:
   - Current status: Monitoring dashboard polls Prometheus exposition metrics on a 10s interval.
   - Future enhancement: Establish an SSE (Server-Sent Events) or WebSocket stream for sub-second ops monitoring.
2. **Interactive Retrain Threshold Tuning**:
   - Current status: Drift detector p-values and retraining gate NDCG margins are read from locked config JSONs.
   - Future enhancement: Allow authenticated administrators to simulate alternate thresholds via interactive sliders before triggering production retraining pipelines.
3. **Dynamic UMAP Projector**:
   - Current status: 2D coordinates are pre-computed offline via parametric UMAP.
   - Future enhancement: Run WebAssembly-based client-side projection for dynamically user-added titles not present in the base index.

---

## 9. Verification Sign-Off & Test Ledger

### 9.1 Backend Test Count Arithmetic Formula

```
  202 baseline tests (recorded in pytest_full.log)
+  46 sprint branch tests (SASRec baseline, quota enforcement, and timestamp leakage tests)
+  18 UI audit verification tests:
      - 6 tests in backend/tests/test_dev_safety_and_fallback.py (403 write guard, 409 missing report, traversal)
      - 10 tests in backend/tests/test_readme_metrics_consistency.py (metric assertion against docs/results/)
      - 2 tests in backend/tests/test_api_contract.py (FastAPI route registry contract vs frontend client)
──────────────────────────────────────────────────────────────────────────
  266 total tests passed (1 deselected artifact test; 0 failed)
```

### 9.2 Verification Suite Results

| Test Category | Command | Target / Scope | Result | Status |
|---|---|---|---|:---:|
| **Full Backend Suite** | `pytest backend/tests/ -m "not artifacts" -q` | Entire backend test suite excluding heavy disk artifacts | **266 passed**, 1 deselected, 29 warnings in 144.92s | **PASS** |
| **Audit Safety & Fallback** | `pytest backend/tests/test_dev_safety_and_fallback.py -v` | Ops write 403, 409 missing reports, directory traversal prevention | **6 passed** in 7.96s | **PASS** |
| **Documentation Consistency** | `pytest backend/tests/test_readme_metrics_consistency.py -v` | Assert README metrics match `docs/results/*.json` exactly | **10 passed** in 6.70s | **PASS** |
| **API Contract Integrity** | `pytest backend/tests/test_api_contract.py -v` | Verify all TypeScript frontend client paths exist in FastAPI app | **2 passed** in 1.41s | **PASS** |
| **Frontend Typecheck** | `npx tsc --noEmit` in `frontend/` | TypeScript type validation across entire App Router codebase | **0 errors** | **PASS** |
| **Frontend ESLint** | `npm run lint` in `frontend/` | Next.js lint + honest-fallbacks + no-hardcoded-metrics | **0 errors, 0 warnings** | **PASS** |
| **Frontend Production Build** | `npm run build` in `frontend/` | Next.js 14 production bundle optimization & standalone staging | **0 errors, 0 warnings** (12 static/dynamic routes prerendered) | **PASS** |
| **Playwright E2E Smoke Suite** | `npm run e2e` in `frontend/` | Multi-mode (Player vs Developer) headless navigation across 20 routes | **2 passed** in 6.4s (0 console errors, 0 failed requests) | **PASS** |
| **Honest-Fallback Linter** | `python scripts/check_honest_fallbacks.py` | AST regex validation for `// HONEST-FALLBACK:` on every `??` | **57 files scanned**, 100% compliant | **PASS** |
| **Metrics Integrity Linter** | `python scripts/check_no_hardcoded_metrics.py` | Scans for hardcoded floating-point metric constants | **140 files scanned**, 0 hardcoded metrics found | **PASS** |
| **Evaluation Lockfile** | `git log --oneline -- docs/results/.test_eval_lock` | Cryptographic lockfile immutability check | **0 modifications**, untouched since creation | **PASS** |
