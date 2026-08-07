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

#### Tab 2: Reviews (Review Intelligence)

**Purpose:** Deep-dive analysis into player feedback, sentiment breakdown, topic discovery, complaint identification, and LLM summaries.

- **Sentiment Overview & Timeline (2-column layout):**
  - *Sentiment Split Bar & Metrics (Left)*: Positive / Mixed / Negative percentages with overall rating pill (`feature_review_sentiment`).
  - *Historical Sentiment Timeline Chart (Right)*: Monthly positive/negative review volume and sentiment ratio over time (`feature_review_sentiment`).
- **Topic & Feature Discovery Grid (2-column grid):**
  - *BERTopic Topic Clusters (Left)*: Top 5 extracted review topics with review count badges and sentiment lean (`feature_review_topics`).
  - *Loved Features Breakdown (Right)*: Top appreciated game features (e.g. "Soundtrack", "Gunplay") ranked by positive mention count (`feature_review_features`).
- **Complaint Analysis & Snippets Table:**
  - Filterable table listing issue categories, complaint share %, severity badge (`--danger` / `--warning`), and clickable "View Representative Snippets" button (`feature_review_complaints`).
- **AI Review Summary Card:**
  - Structured summary block divided into "Core Strengths", "Primary Complaints", and "Player Feature Requests" (`mart_game_overview` / `feature_review_*`).

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
- **Props:** `app_id: string`, `title: string`, `developer: string`, `publisher: string`, `capsule_url: string`, `release_date: string`, `price: number`, `kpis: GameHeaderKPIs`
- **Style:** Fixed/Sticky top banner, `--bg-base`, bottom border 1px `--border-subtle`.
- **Child elements:** Thumbnail, H1 title, status pills, KPI summary strip.

### 4.2 `TabSubNav` (`components/game/TabSubNav.tsx`)
- **Props:** `activeTab: string`, `tabs: TabItem[]`, `onTabChange: (tabId: string) => void`
- **Style:** Horizontal flex track, `--border-subtle` bottom line, active tab underlined with 2px `--accent-primary`.

### 4.3 `StatCard` (`components/ui/StatCard.tsx`)
- **Props:** `eyebrow: string`, `value: string | number`, `subtitle?: string`, `trend?: { value: string, direction: 'up' | 'down' | 'neutral' }`, `accentColor?: string`
- **Style:** Standard card (`--bg-surface`, 12px radius, 1px `--border-subtle` border, 24px padding). Value uses Display typography in JetBrains Mono / tabular-nums.

### 4.4 `SHAPBreakdownCard` (`components/game/SHAPBreakdownCard.tsx`)
- **Props:** `drivers: SHAPFactor[]` (`feature_name: string`, `impact: number`, `direction: 'positive' | 'negative'`)
- **Style:** Horizontal bar list. Positive bars rendered in `--success`, negative bars in `--danger`.

### 4.5 `SentimentDistributionBar` (`components/game/SentimentDistributionBar.tsx`)
- **Props:** `positivePct: number`, `neutralPct: number`, `negativePct: number`
- **Style:** Multi-segment rounded horizontal progress bar using `--success`, `--warning`, and `--danger`.

### 4.6 `TopicClusterGrid` (`components/game/TopicClusterGrid.tsx`)
- **Props:** `topics: TopicCluster[]` (`topic_name: string`, `review_count: number`, `sentiment_score: number`)
- **Style:** Flex pill container, `--bg-surface-raised` background on pills, volume count in `--text-muted`.

### 4.7 `ComplaintTable` (`components/game/ComplaintTable.tsx`)
- **Props:** `complaints: ComplaintCategory[]`
- **Style:** Standard table (§2.8). Uses `--danger` / `--warning` severity badges. Action column triggers representative snippet popover.

### 4.8 `CompetitorTable` (`components/game/CompetitorTable.tsx`)
- **Props:** `competitors: CompetitorGame[]`
- **Style:** Standard table (§2.8). Similarity score rendered in `--accent-primary` pill. Row click navigates to target game.

### 4.9 `RecommendationCard` (`components/game/RecommendationCard.tsx`)
- **Props:** `recommendation: RecommendationItem` (`title: string`, `domain: string`, `impactRank: number`, `rationale: string`, `difficulty: 'Low' | 'Medium' | 'High'`)
- **Style:** Emphasis card (`--bg-surface-raised`, 12px radius, 24px padding). Domain pill in `--accent-primary`, difficulty pill in neutral `--text-secondary`.

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
