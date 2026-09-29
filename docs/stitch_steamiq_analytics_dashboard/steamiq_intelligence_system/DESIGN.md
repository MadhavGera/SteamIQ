---
name: SteamIQ Intelligence System
# ──────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — source of truth (kept in sync with frontend/app/globals.css)
# Theme: Steam Store dark palette (reskinned 2026-09)
# Source: store.steampowered.com / shared_global.css (shared_global.css v2026)
# ──────────────────────────────────────────────────────────────────────────────
colors:
  # Backgrounds — Steam dark blue-grey store palette
  # (mapped from Steam's --gpStoreDarkerGrey / --gpStoreDarkGrey / --gpSystemDarkGrey)
  surface: '#1b2838'          # --bg-base: Steam page background
  surface-dim: '#0e141b'      # --bg-surface-lowest: darkest (Steam --gpStoreDarkestGrey)
  surface-container-lowest: '#0e141b'
  surface-container-low: '#16202d'
  surface-container: '#1e2d3d'         # primary card surface
  surface-container-high: '#2a475e'    # --gpStoreDarkGrey — hover/raised
  surface-container-highest: '#3d4450' # --gpSystemDarkGrey — elevated chips
  surface-bright: '#4e697d'            # --gpStoreGrey — highlight surfaces
  background: '#1b2838'
  # Text — Steam store body text
  on-surface: '#c6d4df'        # Steam primary body text (from store.css body.v6)
  on-surface-variant: '#acb2b8' # Steam secondary text
  on-background: '#c6d4df'
  # Borders
  outline-variant: '#2a475e'   # subtle divider (--gpStoreDarkGrey)
  outline: '#4e697d'           # stronger border (--gpStoreGrey)
  # Interactive accent — Steam's signature chalky blue
  # Source: store.css `a:hover { color: #66c0f4 }` / --gpColor-ChalkyBlue
  primary: '#66c0f4'            # Steam chalky blue — links & interactive accent
  primary-container: '#2a475e'
  on-primary: '#0e141b'
  on-primary-container: '#c6d4df'
  surface-tint: '#87d4ff'
  # Sentiment semantic colors — Steam review color coding
  # Overwhelmingly/Mostly Positive → Steam green (--gpColor-Green / GreenHi)
  # Mixed → amber-yellow; Negative → Steam red (--gpColor-Red)
  success: '#59bf40'    # Steam GreenHi — Positive reviews badge
  warning: '#c7b24a'    # Steam amber — Mixed reviews badge
  danger: '#d94126'     # Steam #D94126 — Negative reviews badge (--gpColor-Red)
  error: '#d94126'
  # Evidence-badge accent series (not on Steam store; internal SteamIQ classification)
  accent-inferred: '#c084fc'    # mauve: AI-inferred activity detection badge
  accent-review-nlp: '#2dd4bf'  # teal: review NLP evidence badge
  accent-pricing: '#fb923c'     # orange: pricing benchmark rule badge
typography:
  # Font note: Steam uses proprietary "Motiva Sans" (geometric humanist, unavailable on
  # Google Fonts). SteamIQ substitutes Nunito Sans — closest freely-licensed equivalent
  # with matching rounded stroke and x-height characteristics.
  font-stack-primary: '"Nunito Sans", Arial, Helvetica, sans-serif'
  font-stack-mono: '"JetBrains Mono", monospace'
  display-lg:
    fontFamily: Nunito Sans
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Nunito Sans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Nunito Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Nunito Sans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Nunito Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Nunito Sans
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Nunito Sans
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  stat-lg:
    fontFamily: JetBrains Mono
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  stat-md:
    fontFamily: JetBrains Mono
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 24px
  label-caps:
    fontFamily: Nunito Sans
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  # Steam uses extremely flat radii: --gpCorner-Small:1px / Medium:2px / Large:3px
  sm: 2px       # --radius-sm (was 4px, now matches Steam --gpCorner-Medium)
  DEFAULT: 3px  # --radius-md (was 8px, now matches Steam --gpCorner-Large)
  md: 3px       # --radius-card (was 12px, Steam cards are near-square-cornered)
  full: 9999px  # --radius-pill — pills remain fully rounded
spacing:
  scale-4: 4px
  scale-8: 8px
  scale-12: 12px
  scale-16: 16px
  scale-24: 24px
  scale-32: 32px
  scale-48: 48px
  scale-64: 64px
  grid-max-width: 1440px
  gutter: 16px
  card-padding-desktop: 24px
  card-padding-mobile: 16px
---

## Brand & Style

The design system is engineered for technical precision and data density, catering to indie developers who require actionable, explainable insights. The visual style is **Steam-adjacent dark** — matching the aesthetic of the real Steam Store to feel native to the gaming context — while preserving the dashboard's analytical character. Surfaces are flat, hierarchy is driven by typography and contrast, and the color palette is sourced directly from Steam's live CSS.

## Colors

The palette matches Steam's actual dark store theme (sourced from `store.steampowered.com/public/shared/css/shared_global.css` and `store.css`).

- **Foundation:** `#1b2838` (`--gpStoreDarkerGrey`) as the page canvas. Layered surfaces step through `#1e2d3d` → `#2a475e` (`--gpStoreDarkGrey`) → `#3d4450` (`--gpSystemDarkGrey`).
- **Text:** `#c6d4df` for primary body (from Steam `body.v6` color), `#acb2b8` secondary, `#8f98a0` muted.
- **Accent (Steam Blue):** `#66c0f4` — Steam's `--gpColor-ChalkyBlue`, the link-hover and interactive highlight color. All buttons, active tab indicators, and key KPIs use this value.
- **Nav Header:** `#171a21` — Steam's actual `#global_header` background (darker than the page canvas), with `#000000` bottom border.
- **Review Sentiment Color Coding** (matching Steam's own Positive/Mixed/Negative badge treatment):
  - **Positive / Overwhelmingly Positive** → `#59bf40` (`--gpColor-GreenHi`)
  - **Mixed / Mostly Positive** → `#c7b24a` (amber-gold)
  - **Negative / Overwhelmingly Negative** → `#d94126` (`--gpColor-Red`)
- **Evidence Badges (SteamIQ-internal):** Mauve `#c084fc` (inferred), Teal `#2dd4bf` (NLP), Orange `#fb923c` (pricing) — not from Steam's own palette but used to distinguish AI evidence sources.

## Typography

Steam uses proprietary **Motiva Sans** (geometric humanist sans-serif), which is not available via Google Fonts or any free license.

- **Substitution:** SteamIQ uses **Nunito Sans** as the closest freely-licensed equivalent. Both typefaces share rounded strokes, similar x-height proportions, and a warm geometric character that matches the Steam store's visual identity.
- **JetBrains Mono** is retained for numeric KPIs, statistics, and data-dense table cells.
- **Label hierarchy:** `label-caps` (11px / 700 / 0.05em tracking) for secondary metadata and table headers.

## Layout & Spacing

The layout follows a **Fluid Grid** model with a hard constraint on maximum width to ensure readability on ultra-wide monitors.

- **Grid Model:** A 12-column system with a 16px gutter.
- **Responsive Behavior:**
  - **Desktop:** 24px margins, standard 24px card padding.
  - **Tablet:** Gutter reduces to 12px, content reflows to 6-column or 1-column stacks.
  - **Mobile:** 16px margins, card padding reduces to 16px to maximize data real-estate.
- **Rhythm:** All component heights and internal spacing must adhere to the 4px/8px baseline increment.

## Elevation & Depth

Tonal layering rather than shadows, matching Steam's flat, matte aesthetic.

- **Surface Tiers:**
  - Level 0: `#1b2838` — main canvas.
  - Level 1: `#1e2d3d` — primary cards and containers.
  - Level 2: `#2a475e` — hovered states or nested modules.
- **Outlines:** Every container at Level 1 and 2 carries a 1px solid `border-subtle (#2a475e)`.
- **Shadows:** Reserved only for dropdowns/modals: `0 8px 24px rgba(0,0,0,0.55)`.

## Shapes

Steam uses very flat, near-rectangular corner treatment across its store UI.

- **Steam's radii (source: `--gpCorner-*`):** 1px (small), 2px (medium), 3px (large).
- **Cards (`--radius-card`):** `3px` — matches Steam's store card treatment (was 12px before retheme).
- **Interactive Elements (`--radius-md`):** `3px` — buttons and inputs match Steam's compact button style.
- **Badges/Labels:** `pill-shaped (9999px)` — retained to distinguish categorical labels from structural containers.

## Components

### Buttons
- **Primary:** `#66c0f4` (Steam chalky blue) fill with `#0e141b` text. No shadow. Bold typography.
- **Secondary:** Transparent background with a `1px` border in `#2a475e`. Text is `#c6d4df`.
- **Ghost:** No border or fill. Text is `#acb2b8`.

### Cards
- **Standard:** `#1e2d3d` background with `#2a475e` outline. `3px` corner radius.
- **Emphasis:** `#2a475e` background. Use for primary data focus or active/hover states.

### Badges & Chips
- **Status Badges (Review Sentiment):** Pill shape. Fill = semantic color at 15% opacity. Text = semantic color at 100%. Steam-matched colors: green `#59bf40` / amber `#c7b24a` / red `#d94126`.

### Navigation
- **Top Navbar:** 64px height, `#171a21` background (Steam's actual header), `#000000` bottom border.
- **Sub-nav:** Horizontal tab bar. Active states: 2px bottom stroke in `#66c0f4` (Steam chalky blue).

### Form Fields
- **Inputs:** `#1b2838` fill, `#2a475e` stroke. On focus, border changes to `#66c0f4`.

### Charts & Visualization
- **Style:** Flat fills only. No gradients or 3D effects.
- **Gridlines:** Use `border/subtle (#2a475e)` at 50% opacity.
- **Icons:** Material Symbols Outlined style.