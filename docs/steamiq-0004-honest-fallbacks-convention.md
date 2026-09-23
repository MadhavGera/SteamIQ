# SteamIQ Architectural Rule: Honest Fallbacks Convention

**Rule ID:** `STEAMIQ-FE-0001`  
**Applies To:** Frontend presentation components, hooks, and API formatting helpers  
**Target Invariant:** Zero Fabricated Data in User Interfaces

---

## 1. Principle

Any metric, score, count, price, quote, or analytical attribute rendered in the SteamIQ interface **must strictly originate from verified database records or analytical pipeline models**.

When data has not yet been ingested, materialized, or processed for a given game:
1. **Never substitute a plausible-looking placeholder** (e.g. `84%`, `3,247`, `$14.99`, or fabricated customer quotes).
2. **Render an honest empty / pending state** (e.g. `"--"`, `"Not yet scored"`, `"Review analysis not yet available"`, or a dedicated pending card).
3. **Visually differentiate pending states** from populated metrics so users are never misled.

---

## 2. The `// HONEST-FALLBACK:` Comment Convention

Any null-coalescing (`??`), logical OR (`||`), or conditional ternary used for display values in frontend components must be documented with an explicit `// HONEST-FALLBACK:` comment indicating what the fallback is and confirming it does not fabricate real domain data.

### ✅ Compliant Examples:

```tsx
// HONEST-FALLBACK: Real metric value or explicit pending state indicator
const successScoreStr = game?.success_score != null
  ? `${Math.round(Number(game.success_score))}%`
  : "--";

// HONEST-FALLBACK: Real median price from spectrum/overview or null if uncalculated
const medianPrice = spec?.median_price_usd != null ? spec.median_price_usd : market.genre_median_price_usd;

// HONEST-FALLBACK: Safe structural empty array for unpopulated list
const topics = reviewsBundle?.topics ?? [];
```

### ❌ Non-Compliant Examples (Strictly Forbidden):

```tsx
// ❌ WRONG: Fabricating a fake success score
const successScoreStr = game?.success_score ? `${game.success_score}%` : "84%";

// ❌ WRONG: Fabricating fake CCU count
const peakCcuStr = game?.peak_ccu_24h ? game.peak_ccu_24h.toLocaleString() : "3,247";

// ❌ WRONG: Hardcoded game metadata based on App ID
const title = game?.name ?? (appId === 1145360 ? "Hades" : "Hollow Knight");

// ❌ WRONG: Fabricating fake review themes and quotes when unprocessed
const snippets = reviewsBundle?.complaints ?? [{ category: "Performance", representative_snippets: ["Stutters on boss fight"] }];
```

---

## 3. Grep Auditing

Run this ripgrep check to verify that all null coalescing operators in UI presentation code adhere to the honest fallback invariant:

```bash
rg "\?\?" frontend/components
```
