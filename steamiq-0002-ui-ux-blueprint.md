# SteamIQ UI/UX Blueprint

Status: Draft — Information Architecture locked, page specs pending
Companion to: `0001-architecture-blueprint.md`

## Golden rule (pin this at the top, same as the architecture ADR)

> Frontend implementation must follow this document. Do not invent pages, sections, components, interactions, colors, layouts, navigation items, or placeholder functionality that are not defined here. If something required by backend functionality has no UI specification yet, leave it unimplemented and flag it for design instead of designing it automatically.

The Phase 1 frontend is a functional prototype only — it proved `PostgreSQL → FastAPI → Next.js` works end to end. It is not the design. No page beyond what Phase 1 already built gets implemented until its section below exists.

---

## 1. Information Architecture (locked)

```
SteamIQ
│
├── Search / Landing
├── Dashboard
│
├── Game Intelligence  (per game, tabbed)
│   ├── Overview
│   ├── Reviews                 — Review Intelligence
│   ├── Player Activity         — Player Behaviour / activity forecasting
│   ├── Market                  — Market Intelligence (success, revenue tier)
│   │                             + Pricing Intelligence (merged in, not a separate tab)
│   ├── Competitors             — Similarity Engine
│   ├── Updates                 — Update Impact Tracker
│   └── Recommendations         — Recommendation Engine
│
├── Compare Games
├── Market Explorer             — Opportunity Finder + Market Trends (merged)
├── Ask SteamIQ
└── Settings
```

**Decisions locked this round:**
- Player Behaviour does **not** get its own page — it's a tab inside Game Intelligence, alongside Reviews/Market/Competitors.
- Pricing Intelligence does **not** get its own tab — its output (comparable price range, market position) renders as a section within the Market tab, next to success/revenue-tier content.
- Opportunity Finder and Market Trends are **not** two separate pages — they merge into a single global page, **Market Explorer**, since both are cross-game/platform-wide analysis rather than per-game data.

**Still open (defer until each page is speced, don't decide now):**
- Whether Dashboard and Search/Landing are one page or two
- Exact tab order within Game Intelligence
- Whether Settings ships in the MVP at all, or is a Phase 7 stretch

---

## 2. Global design system (locked)

Dark, data-dense, Steam-adjacent without copying Steam's own UI. Flat surfaces, 1px borders instead of heavy shadows, one accent color used consistently for anything interactive or "this number matters."

### 2.1 Color tokens

| Token | Hex | Use |
|---|---|---|
| `--bg-base` | `#0B1420` | Page background |
| `--bg-surface` | `#131F2E` | Cards, panels, table containers |
| `--bg-surface-raised` | `#1A2A3D` | Hover states, nested cards, emphasis blocks (chat bubbles, opportunity cards) |
| `--border-subtle` | `#23364A` | Card borders, table dividers, nav underline track |
| `--text-primary` | `#F2F6FA` | Headings, primary body text |
| `--text-secondary` | `#9FB2C4` | Body copy, descriptions |
| `--text-muted` | `#6B7F92` | Captions, eyebrows, placeholder text |
| `--accent-primary` | `#5EC2F0` | Primary buttons, links, active nav/tab, chart primary series, KPI numbers |
| `--accent-primary-hover` | `#7ED0F5` | Hover state of the above |
| `--accent-secondary` | `#3D6E99` | Icon circles, secondary chart series, less prominent accents |
| `--success` | `#4ADE9A` | Positive sentiment, growth, low risk, SHAP positive contributors |
| `--warning` | `#F5C15C` | Moderate risk, neutral/mixed signal |
| `--danger` | `#F2685C` | Negative sentiment, complaints, decline, SHAP negative contributors, high risk |

Semantic colors (success/warning/danger) are reserved for actual sentiment/risk/trend meaning — never used decoratively. If a chart or badge isn't representing one of those three states, it uses `--accent-primary` or a neutral gray, not red/green/yellow.

Ready-to-paste block for `globals.css`:
```css
:root {
  --bg-base: #0B1420;
  --bg-surface: #131F2E;
  --bg-surface-raised: #1A2A3D;
  --border-subtle: #23364A;
  --text-primary: #F2F6FA;
  --text-secondary: #9FB2C4;
  --text-muted: #6B7F92;
  --accent-primary: #5EC2F0;
  --accent-primary-hover: #7ED0F5;
  --accent-secondary: #3D6E99;
  --success: #4ADE9A;
  --warning: #F5C15C;
  --danger: #F2685C;
}
```

### 2.2 Typography

- **Font:** Inter (already in the Phase 1 prototype — keep it, it's a good fit for a data-dense dashboard). Load via `next/font`.
- **Numeric/stat font:** JetBrains Mono for large stat callouts only (success scores, sentiment %, opportunity scores) — `font-variant-numeric: tabular-nums` if staying on Inter instead is preferred; pick one and apply it everywhere numbers are the hero of a card.

| Style | Size / Weight | Use |
|---|---|---|
| Display | 40–48px / 700 | Hero stat callouts (e.g. "84%" success score) |
| H1 | 28px / 700 | Page title |
| H2 | 20px / 600 | Section title within a page |
| H3 | 16px / 600 | Card title |
| Body | 14px / 400 | Default body copy |
| Small | 12px / 400, `--text-muted` | Captions, timestamps, source citations |
| Eyebrow | 11px / 600, uppercase, `letter-spacing: 0.05em`, `--text-muted` | Card overlines ("TOP COMPLAINTS", "OPPORTUNITY SCORE") |

### 2.3 Spacing & layout

- Base unit: 4px. Scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64.
- Card internal padding: 24px (16px on mobile).
- Grid gutter: 16px.
- Max content width: 1440px, centered — dashboards don't stretch full-width on ultrawide monitors.

### 2.4 Radius

- Cards / panels: 12px
- Buttons / inputs / dropdowns: 8px
- Badges / pills / avatar-style icon circles: fully rounded (999px)

### 2.5 Elevation

Flat by default — separation comes from `--border-subtle` 1px borders, not shadows. Reserve shadow for genuinely floating elements only: dropdown menus, modals, toasts (`0 8px 24px rgba(0,0,0,0.35)`).

### 2.6 Buttons

| Variant | Background | Text | Border | Use |
|---|---|---|---|---|
| Primary | `--accent-primary` | `--bg-base` (dark text on light-blue bg) | none | The one primary action per view |
| Secondary | transparent | `--text-primary` | 1px `--border-subtle` | Secondary actions |
| Ghost | transparent | `--text-secondary` | none, `--bg-surface-raised` on hover | Tertiary/inline actions |
| Danger | `--danger` | `--bg-base` | none | Destructive actions only |

Sizes: sm 32px / md 40px / lg 48px height, radius 8px throughout.

### 2.7 Cards

- **Standard card:** `--bg-surface` background, 1px `--border-subtle` border, 12px radius, 24px padding.
- **Emphasis card** (chat bubbles, opportunity/SHAP callouts — anything meant to stand out on a page that's otherwise light-background): `--bg-surface-raised` or `--bg-base`, no border, used sparingly — one or two per page, not as the default card style.
- **Stat card:** eyebrow label top, Display-size number in `--accent-primary` (or semantic color if it's a sentiment/risk number), one-line context label beneath.

### 2.8 Tables

- Header row: eyebrow style, bottom border `--border-subtle`, no background fill.
- Row divider: 1px `--border-subtle`, no zebra striping.
- Row hover: `--bg-surface-raised`.
- Numeric columns right-aligned, tabular figures.

### 2.9 Charts

- Primary series: `--accent-primary`.
- Secondary/comparison series: `--accent-secondary` or `--text-muted`.
- Sentiment splits (positive/negative): `--success` / `--danger` — never any other color pairing for this specific meaning.
- Grid lines: `--border-subtle` at reduced opacity, never pure white/black.
- No 3D effects, no gradient fills on bars — flat fills only.

### 2.10 Badges / status pills

Pill shape, 4px/10px padding, 12px/600 text, background = semantic color at ~15% opacity, text = full-strength semantic color.

- Positive / Low risk / Improving → `--success`
- Moderate / Mixed → `--warning`
- Negative / High risk / Declining → `--danger`
- Neutral / informational → `--accent-primary`

### 2.11 Navigation

- **Top navbar:** fixed, `--bg-base` with bottom `--border-subtle`, 64px height, logo left, primary nav right. No sidebar — this is a top-nav + in-page-tabs product, not an admin-panel-with-sidebar product. (If a future page genuinely needs a sidebar, that's a page-spec decision to make explicitly in §3, not a default to fall back on.)
- **Game Intelligence sub-nav:** horizontal tab bar directly under the page header (Overview / Reviews / Player Activity / Market / Competitors / Updates / Recommendations), active tab underlined in `--accent-primary`, sticky under the navbar on scroll.

### 2.12 Iconography

`lucide-react` (pairs with shadcn/ui, already in the stack) — outline style throughout, 20px in nav/tabs, 24px inside cards, 16px inline with text. No mixing icon sets/styles across pages.

### 2.13 Responsive breakpoints

| Breakpoint | Range | Behavior |
|---|---|---|
| Mobile | <640px | Single column, KPI cards stack, tabs become horizontally scrollable |
| Tablet | 640–1024px | 2-column grids |
| Desktop | 1024–1440px | Up to 4-column grids — the default target for dashboard/grid layouts |
| Wide | >1440px | Content capped at 1440px, centered — grids don't gain more columns past desktop |

## 3. Page specifications
*One subsection per page, filled in before that page is (re)implemented.*

### 3.1 Game Intelligence Page (`/game/[app_id]`)

**Purpose:**  
Serve as the primary analytical hub for a single game. Provides developers, publishers, and investors with comprehensive insights spanning sentiment, player retention, revenue/pricing positioning, competitor similarity, patch impact, and actionable automated recommendations.

**Entry points:**  
- Main landing page search bar result selection
- Links from Compare Games tool & Market Explorer
- Top dashboard game lists
- Direct deep links (e.g. `/game/1091500`)

**Layout (Top to Bottom):**  
1. **Sticky Top Navbar** (Global 64px, `--bg-base`)
2. **Sticky Game Header Shell** (Fixed directly below top navbar):
   - Left: Game capsule thumbnail (120x60px), Title (`H1`), App ID badge, Developer / Publisher overline, Release Date, Price tag
   - Right: Quick KPI strip (Success Score pill, Net Sentiment badge, 24h Peak CCU)
   - Bottom: Horizontal Sub-navigation Track (7 sticky tabs: `Overview`, `Reviews`, `Player Activity`, `Market`, `Competitors`, `Updates`, `Recommendations`)
3. **Tab Content Container** (Max-width 1440px, centered, 24px padding)

---

#### Tab 1: Overview (locked)

**Purpose:**  
High-level executive dashboard answering three fundamental questions in the first three seconds:
1. *How healthy is this game?* (KPI strip)
2. *Why is it succeeding or struggling?* (SHAP feature attribution + Executive Brief)
3. *Where should I investigate next?* (Deep-dive teaser cards)

**Data-availability & Progressive Disclosure Rule (Zero Fake Data):**  
Cards in the Overview tab **only render when backed by populated database tables**. No fake or placeholder cards are shown:
- **Phase 1 / 2:** KPI strip renders 2 cards (`Net Sentiment` from `feature_review_sentiment` and `Player Activity / Owners` from `raw_player_snapshots` / `raw_games`). Teaser row renders only `Review Intelligence`.
- **Phase 3:** Unlocks `Competitor Radar` teaser card (from `serving_similar_games`).
- **Phase 4:** Unlocks `Success Score` KPI card + `SHAP Feature Attribution` panel (from `serving_predictions`).
- **Phase 5:** Unlocks `Revenue Tier` KPI card + `Priority Recommendation` teaser card (from `mart_game_overview` / `serving_recommendations`).

---

**Sections (Top to Bottom):**

1. **Dynamic Hero KPI Grid (2 to 4 columns based on active phase):**
   - *Success Score Card (Phase 4+)*: XGBoost+LightGBM ensemble score (0–100%), model run ID badge, confidence interval (`serving_predictions`).
   - *Net Sentiment Card (Phase 2+)*: Net positive review % with `--success` / `--danger` semantic indicator and total review volume (`feature_review_sentiment`).
   - *Player Activity Card (Phase 1+)*: 24h Peak CCU & 30-day average trend (`raw_player_snapshots`).
   - *Revenue Tier Card (Phase 5+)*: Estimated Gross Revenue Tier bracket & price vs genre benchmark (`mart_game_overview`).

2. **Explainability & Executive Brief Panel (60/40 Split Grid):**
   - *SHAP Factor Attribution (Left 60%, Phase 4+)*: Top 3 positive contributors (e.g., "High Review Velocity", "Loved Combat System") in `--success` and top 2 negative drag factors (e.g., "Early Difficulty Spike", "Controller Input Bugs") in `--danger` (`serving_predictions`).
   - *Executive Intelligence Brief (Right 40%, Phase 2/5)*: Structured 2-paragraph brief synthesized via a deterministic template stitching the Phase 2 hierarchical review summary and Phase 4/5 performance metrics into `mart_game_overview` (zero extra LLM calls in the request path).

3. **Deep-Dive Teaser Row (Collapses to active tabs only):**
   - *Review Intelligence Teaser (Phase 2+)*: Top loved feature vs top complaint category with a link jumping to the `Reviews` tab.
   - *Competitor Radar Teaser (Phase 3+)*: Top 3 pgvector nearest games with match score % and price delta, jumping to the `Competitors` tab.
   - *Priority Recommendation Teaser (Phase 5+)*: Highest-ROI actionable recommendation card with difficulty pill, jumping to the `Recommendations` tab.

---

#### Tab 2: Reviews (Review Intelligence) (locked)

**Purpose:**  
Transform raw Steam review text into structured qualitative intelligence. Answers: *What do players specifically love, what are the primary pain points/complaints, how has sentiment evolved over time, and what is the overall synthesized verdict?*

**Entry points:**  
- Game Intelligence tab bar ("2. Reviews")
- "Review Intelligence Teaser" card on the Overview tab
- Direct URL with tab query: `/game/[app_id]?tab=reviews`

**Sections (Top to Bottom):**

1. **Sentiment Breakdown & Monthly Timeline (2-column 50/50 layout):**
   - *Sentiment Overview Panel (Left)*: Positive / Mixed / Negative percentage breakdown with overall score pill (e.g. `97% Overwhelmingly Positive`), positive review count, and negative review count (`feature_review_sentiment`).
   - *Historical Sentiment Timeline Chart (Right)*: Monthly time-series chart plotting positive volume in `--success` (`#4ADE9A`) and negative volume in `--danger` (`#F2685C`) alongside net sentiment ratio over time (`feature_review_sentiment`).

2. **Topic Discovery & Loved Features (2-column 50/50 grid):**
   - *BERTopic Topic Clusters (Left)*: Extracted semantic topic clusters from reviews (e.g. "Combat Mechanics", "Lore & Worldbuilding", "Boss Design", "Art Style") with review frequency count badges and net sentiment lean (`feature_review_topics`). Wrapped in a fallback to TF-IDF+KMeans if BERTopic pipeline unavailable.
   - *Loved Features Breakdown (Right)*: Ranked list of positive feature appreciations extracted via HDBSCAN embeddings (e.g. "Soundtrack", "Atmosphere", "Controls") sorted by praise volume (`feature_review_features`).

3. **Complaint Breakdown & Snippets Table:**
   - Filterable data table from zero-shot complaint classification (`feature_review_complaints`).
   - Columns: Issue Category (e.g. "Performance / Stuttering", "Input Latency", "Difficulty Spike"), Volume Share %, Severity Badge (`--danger` "High" / `--warning` "Moderate"), and Action button: *"View Representative Snippets"*.
   - Clicking *"View Representative Snippets"* triggers a floating modal (`SnippetModal`) displaying 3 real review quotes tagged with that complaint.

4. **Hierarchical AI Review Summary Card:**
   - Precomputed structured summary card (`mart_game_overview` / `feature_review_*`) cleanly divided into three callout sections:
     - **Core Strengths:** Bulleted summary of universally praised gameplay elements.
     - **Pain Points:** Key friction areas and technical complaints.
     - **Player Feature Requests:** Commonly requested mechanics, QoL improvements, or content desires.

**Data contracts:**
- Sentiment split & timeline: `feature_review_sentiment`
- Extracted topics: `feature_review_topics`
- Complaint classification & snippets: `feature_review_complaints`
- Feature appreciation: `feature_review_features`
- Executive summary: `mart_game_overview` (Phase 5) / precomputed summary job

**Components used:** `SentimentOverview`, `SentimentTimeline`, `TopicDistribution`, `LovedFeatures`, `ComplaintBreakdown`, `ReviewSummary`, `SnippetModal` (see §4).

**States:**
- Loading: Skeleton progress bars, shimmer chart container, and table skeletons.
- No-data-yet (game ingested in `raw_reviews`, but NLP pipeline has not run yet): Content replaced by a clean banner: *"Review intelligence in progress — run `make process-reviews` to generate sentiment, topics, and complaint analytics."*
- Empty: Game has 0 user reviews on Steam.

---

#### Tab 3: Player Activity (Player Behaviour)

**Purpose:** Track concurrent player counts, retention trends, and player activity forecasts.

- **CCU Overview Bar:** Current CCU, 24h Peak CCU, 30d Peak CCU, All-time Peak CCU (`raw_player_snapshots`).
- **Interactive Player Activity Chart:** Time-series graph with timeframe selector (`7d`, `30d`, `90d`, `1y`, `All-time`) showing hourly/daily CCU.
- **Activity Forecast Panel:** 30-day forward-looking CCU trajectory curve with confidence intervals (`serving_predictions`).

---

#### Tab 4: Market (Market & Pricing Intelligence)

**Purpose:** Evaluate commercial performance, revenue category placement, and price positioning vs competitors.

- **Revenue Tier & Market Placement Card:** Estimated revenue tier (e.g. "Tier 2: $1M–$5M"), estimated gross revenue range, and owner count bracket (`mart_game_overview`).
- **Comparable Price Spectrum:** Horizontal spectrum chart plotting the game's price against genre min, median, and max price ranges (`feature_market_features`).
- **Market Density & Opportunity Index:** Genre saturation index card and market growth indicator.

---

#### Tab 5: Competitors (Similarity Engine)

**Purpose:** Identify vector-similar games using pgvector embeddings and compare metrics side by side.

- **Competitor Similarity Table:**
  - Table showing Top 10 similar games. Columns: Rank, Game (Capsule + Title), Similarity Score % (`serving_similar_games`), Price, Review Score %, 24h Peak CCU, Common Tags/Topics.
- **Feature & Sentiment Comparison Matrix:** Side-by-side breakdown comparing review sentiment and pricing across top 3 closest competitors.

---

#### Tab 6: Updates (Update Impact Tracker)

**Purpose:** Quantify the impact of game updates, patches, and DLC releases on review sentiment and player engagement.

- **Patch Timeline List:** Chronological timeline of updates/patches (`raw_games` / `mart_trends`).
- **Update Impact Delta Cards:**
  - 7-day pre vs post-patch sentiment delta %
  - 7-day pre vs post-patch CCU delta %
  - Automated verdict pill (`--success` "Positive Reception", `--warning` "Mixed Impact", `--danger` "Player Backlash").

---

#### Tab 7: Recommendations (Recommendation Engine)

**Purpose:** Provide prioritized, explainable recommendations for improving game success, revenue, and player retention.

- **Recommendation Feed:**
  - Priority-ranked stack of actionable recommendations (`serving_recommendations`).
  - Card elements: Title, Impact Rank, Domain Tag (Pricing, Performance, Content, Marketing), Rationale, Supporting SHAP/Review Evidence Link, Implementation Difficulty Pill (`Low`, `Medium`, `High`).

---

#### States (Game Intelligence Page):
- **Loading State:** Skeleton loaders for Game Header, KPI cards, and active tab content grids.
- **Unprocessed / No-Data-Yet State:** Displays banner "Game data currently ingesting. Raw data available; NLP and ML pipelines scheduled." with a manual "Trigger Analysis" button (if permitted).
- **Error State:** 404 container "Game ID not found in Steam database" with search bar return prompt.

---

## 4. Component specifications
*Reusable pieces referenced by page specs above.*

### 4.1 `GameHeader` (`components/game/GameHeader.tsx`)
**Purpose:** Persistent sticky top shell rendered across all 7 tabs of Game Intelligence (`/game/[app_id]`). Provides primary game context, metadata badges, quick KPI strip, and sticky sub-navigation.

**TypeScript Interface:**
```typescript
export interface GameHeaderKPIs {
  successScore?: number | null;     // Phase 4+ (0-100)
  modelRunId?: string | null;       // Phase 4+ (e.g. "#8f2a")
  netSentimentPct?: number | null;  // Phase 2+ (0-100)
  sentimentLabel?: string | null;   // Phase 2+ (e.g. "Overwhelmingly Positive")
  peakCcu24h?: number | null;       // Phase 1+ (from raw_player_snapshots)
}

export interface GameHeaderProps {
  appId: number;
  title: string;
  developer?: string | null;
  publisher?: string | null;
  releaseDate?: string | null;
  priceUsd?: string | null;
  isFree?: boolean;
  headerImage?: string | null;
  activeTab: string;
  onTabChange?: (tabId: string) => void;
  kpis?: GameHeaderKPIs;
}
```

**Layout & Token Bindings:**
- Outer container: `position: sticky; top: 64px; z-index: 90; background: var(--bg-surface); border-bottom: 1px solid var(--border-subtle);`
- Left section: Capsule art (`120x60px`, `border-radius: 6px`, `border: 1px solid var(--border-subtle)`), `H1` title (28px / 700, `--text-primary`), metadata byline (`--text-secondary`, 13px), and `badge-pill` chips for App ID and price.
- Right section: Condensed KPI strip. **Progressive disclosure rule:** If `kpis.successScore` is null, the success pill is omitted. If `netSentimentPct` is null, the sentiment pill is omitted. Only available numbers render.
- Bottom sub-nav track: 7 tabs (`Overview`, `Reviews`, `Player Activity`, `Market`, `Competitors`, `Updates`, `Recommendations`), 13px / 600, active tab underlined with 2px `var(--accent-primary)`.

---

### 4.2 `StatCard` (`components/ui/StatCard.tsx`)
**Purpose:** Primary metric building block used in KPI grids across Overview, Reviews, Player Activity, and Market tabs.

**TypeScript Interface:**
```typescript
export interface StatCardProps {
  eyebrow: string;                           // 11px uppercase overline
  value: string | number | null | undefined; // Display number (JetBrains Mono)
  sub?: React.ReactNode;                     // Context label beneath number
  accentColor?: string;                      // Defaults to var(--accent-primary)
  trend?: {
    value: string;
    direction: 'up' | 'down' | 'neutral';
  };
  loading?: boolean;
}
```

**Layout & Behavior:**
- Container: `background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 20px;`
- Eyebrow: `font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);`
- Value: `font-size: 32px; font-weight: 700; font-family: var(--font-mono); font-variant-numeric: tabular-nums; line-height: 1.1;`
- **Progressive Disclosure Rule:** When `value` is `null` or `undefined` and `loading` is `false`, the component returns `null` (collapsing the grid cleanly) rather than rendering a misleading "0" or "N/A". If `loading: true`, renders a shimmer skeleton (`.skeleton`).

---

### 4.3 `SHAPDriverList` (`components/game/SHAPDriverList.tsx`)
**Purpose:** Explainable ML feature attribution list displaying positive drivers and negative drag factors from `serving_predictions`.

**TypeScript Interface:**
```typescript
export interface SHAPDriver {
  featureName: string;            // Raw feature slug (e.g. "review_velocity_30d")
  displayName: string;            // User-facing label (e.g. "High Review Velocity")
  impact: number;                 // Contribution score (e.g. +0.32 or -0.14)
  direction: 'positive' | 'negative';
  category?: 'sentiment' | 'market' | 'velocity' | 'developer';
}

export interface SHAPDriverListProps {
  drivers: SHAPDriver[];
  maxItems?: number;              // Defaults to 5 for Overview panel
  loading?: boolean;
}
```

**Layout & Token Bindings:**
- Item row: `display: flex; align-items: center; justify-content: space-between; font-size: 13px;`
- Feature label: `width: 180px; font-weight: 500; color: var(--text-primary);`
- Track: `flex: 1; height: 8px; background: var(--bg-base); border-radius: 4px; margin: 0 12px; overflow: hidden;`
- Bar fill: Positive bars use `var(--success)` (`#4ADE9A`); negative bars use `var(--danger)` (`#F2685C`).
- Score badge: JetBrains Mono tabular numeral, colored in `--success` (for positive) or `--danger` (for negative).
- **Progressive Disclosure Rule:** If `drivers` array is empty (pre-Phase 4), the entire panel returns `null`.

---

### 4.4 `TeaserCard` (`components/game/TeaserCard.tsx`)
**Purpose:** Interactive summary card on the Overview tab that previews a sub-domain and deep-links directly to its dedicated tab.

**TypeScript Interface:**
```typescript
export interface TeaserCardProps {
  title: string;
  icon?: React.ReactNode;
  targetTab: 'reviews' | 'player-activity' | 'market' | 'competitors' | 'updates' | 'recommendations';
  actionLabel?: string;          // Defaults to "View full analysis →"
  children: React.ReactNode;
  badge?: {
    text: string;
    variant?: 'success' | 'warning' | 'danger' | 'neutral';
  };
  onNavigate?: (tab: string) => void;
}
```

**Layout & Behavior:**
- Container: `background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 20px; transition: all 150ms ease;`
- Hover state: `border-color: var(--border-strong); background: var(--bg-surface-raised); cursor: pointer;`
- Header: Title (`H3`, 15px/700, `--text-primary`), optional icon, optional `badge-pill`.
- Action link: `font-size: 12px; font-weight: 600; color: var(--accent-primary); margin-top: 14px; display: inline-flex; align-items: center; gap: 4px;`
- **Progressive Disclosure Rule:** Only instantiated on the Overview page for tabs with active, populated database tables.

---

### 4.5 `SentimentOverview` (`components/game/SentimentOverview.tsx`)
**Purpose:** Multi-segment sentiment proportion bar and metric card on the Reviews tab.

**TypeScript Interface:**
```typescript
export interface SentimentOverviewProps {
  positivePct: number;
  mixedPct: number;
  negativePct: number;
  positiveCount: number;
  negativeCount: number;
  sentimentLabel: string;
  loading?: boolean;
}
```
**Tokens & Layout:** Multi-segment bar using `--success`, `--warning`, `--danger`. Score badge uses `badge-pill--success` (if >80%), `badge-pill--warning` (60–80%), or `badge-pill--danger` (<60%).

---

### 4.6 `SentimentTimeline` (`components/game/SentimentTimeline.tsx`)
**Purpose:** Monthly volume and sentiment ratio time-series chart.

**TypeScript Interface:**
```typescript
export interface MonthlySentimentData {
  month: string;           // "2024-05"
  positiveReviews: number;
  negativeReviews: number;
  netPositivePct: number;
}

export interface SentimentTimelineProps {
  data: MonthlySentimentData[];
  loading?: boolean;
}
```
**Tokens & Layout:** Positive bars/line in `--success`, negative in `--danger`, gridlines in `var(--border-subtle)` at reduced opacity.

---

### 4.7 `TopicDistribution` (`components/game/TopicDistribution.tsx`)
**Purpose:** BERTopic cluster pills display showing extracted review topics with review count and sentiment lean.

**TypeScript Interface:**
```typescript
export interface ReviewTopic {
  topicId: number;
  label: string;          // e.g. "Combat Mechanics"
  reviewCount: number;
  sentimentScore: number; // 0-1
}

export interface TopicDistributionProps {
  topics: ReviewTopic[];
  onTopicSelect?: (topicId: number) => void;
  loading?: boolean;
}
```

---

### 4.8 `LovedFeatures` (`components/game/LovedFeatures.tsx`)
**Purpose:** Ranked list of positively appreciated features extracted via HDBSCAN embeddings.

**TypeScript Interface:**
```typescript
export interface LovedFeatureItem {
  featureName: string;
  mentionCount: number;
  praiseIntensity: number; // 0-100
}

export interface LovedFeaturesProps {
  features: LovedFeatureItem[];
  loading?: boolean;
}
```

---

### 4.9 `ComplaintBreakdown` (`components/game/ComplaintBreakdown.tsx`)
**Purpose:** Zero-shot classified complaint table with severity badges and snippet modal trigger.

**TypeScript Interface:**
```typescript
export interface ComplaintCategory {
  category: string;
  volumePct: number;
  severity: 'high' | 'moderate' | 'low';
  representativeSnippets: string[];
}

export interface ComplaintBreakdownProps {
  complaints: ComplaintCategory[];
  onViewSnippets: (complaint: ComplaintCategory) => void;
  loading?: boolean;
}
```

---

### 4.10 `ReviewSummary` (`components/game/ReviewSummary.tsx`)
**Purpose:** Three-part hierarchical summary card (Core Strengths, Pain Points, Player Wishes).

**TypeScript Interface:**
```typescript
export interface ReviewSummaryProps {
  strengths: string[];
  painPoints: string[];
  featureRequests: string[];
  loading?: boolean;
}
```

---

### 4.11 `SnippetModal` (`components/game/SnippetModal.tsx`)
**Purpose:** Floating modal displaying 3 representative review quotes for a clicked complaint category.

**TypeScript Interface:**
```typescript
export interface SnippetModalProps {
  isOpen: boolean;
  categoryTitle: string;
  snippets: string[];
  onClose: () => void;
}
```
**Tokens & Elevation:** Overlay `rgba(0,0,0,0.6)`, modal container `var(--bg-surface-raised)`, border `1px solid var(--border-strong)`, shadow `var(--shadow-dropdown)` (`0 8px 24px rgba(0,0,0,0.35)`).

---

## 5. Interaction specifications
*Click/search/filter/compare behavior, per page.*

### 5.1 Tab Navigation (`/game/[app_id]`)
- **Behavior:** Clicking a tab updates URL search params (`?tab=reviews`) and updates the rendered tab component without full page reload or layout shift.

### 5.2 Timeframe Filtering on Charts
- **Behavior:** Clicking `7d | 30d | 90d | 1y | All` re-fetches or filters chart series dataset seamlessly with a short pulse loading skeleton on the chart container.

### 5.3 Review Snippet Popover Modal
- **Behavior:** Clicking "View Snippets" in the Complaint table opens `SnippetModal` with `0 8px 24px rgba(0,0,0,0.35)` shadow, showing 3 representative raw player reviews tagged with that complaint.

### 5.4 Competitor Navigation
- **Behavior:** Hovering over a competitor row highlights the row with `--bg-surface-raised`. Clicking opens the selected game's Intelligence page.

---

## Wiring this into the roadmap

Replace vague roadmap lines like "Phase 2 — Review Intelligence UI" with:

```
Phase 2 — Implement Reviews tab per UI Blueprint §3 → Game Intelligence → Reviews
Allowed components: SentimentOverview, SentimentTimeline, TopicDistribution,
                     LovedFeatures, ComplaintBreakdown, ReviewSummary, SnippetModal
Do not alter navigation or overall page structure.
```

Backend/data work (Phase 2's `feature_*` tables, sentiment job, BERTopic pipeline, RQ worker) is unaffected by any of this and continues now — only frontend implementation waits on a page's §3 entry existing.


---

## 5. Interaction specifications
*Click/search/filter/compare behavior, per page.*

### 5.1 Tab Navigation (`/game/[app_id]`)
- **Behavior:** Clicking a tab updates URL search params (`?tab=reviews`) and updates the rendered tab component without full page reload or layout shift.

### 5.2 Timeframe Filtering on Charts
- **Behavior:** Clicking `7d | 30d | 90d | 1y | All` re-fetches or filters chart series dataset seamlessly with a short pulse loading skeleton on the chart container.

### 5.3 Review Snippet Popover Modal
- **Behavior:** Clicking "View Snippets" in the Complaint table opens a modal centered on screen with `0 8px 24px rgba(0,0,0,0.35)` shadow, showing 3 representative raw player reviews tagged with that complaint.

### 5.4 Competitor Navigation
- **Behavior:** Hovering over a competitor row highlights the row with `--bg-surface-raised`. Clicking opens the selected game's Intelligence page.

---

## Wiring this into the roadmap

Replace vague roadmap lines like "Phase 2 — Review Intelligence UI" with:

```
Phase 2 — Implement Reviews tab per UI Blueprint §3 → Game Intelligence → Reviews
Allowed components: SentimentOverview, SentimentTimeline, TopicDistribution,
                     ComplaintBreakdown, LovedFeatures, ReviewSummary
Do not alter navigation or overall page structure.
```

Backend/data work (Phase 2's `feature_*` tables, sentiment job, BERTopic pipeline, RQ worker) is unaffected by any of this and continues now — only frontend implementation waits on a page's §3 entry existing.
