---
name: SteamIQ Intelligence System
colors:
  surface: '#021522'
  surface-dim: '#021522'
  surface-bright: '#293b4a'
  surface-container-lowest: '#00101c'
  surface-container-low: '#0a1d2b'
  surface-container: '#0e212f'
  surface-container-high: '#192c3a'
  surface-container-highest: '#243745'
  on-surface: '#d1e5f8'
  on-surface-variant: '#bec8cf'
  inverse-surface: '#d1e5f8'
  inverse-on-surface: '#203241'
  outline: '#889299'
  outline-variant: '#3e484e'
  surface-tint: '#73d1ff'
  primary: '#98dbff'
  on-primary: '#003548'
  primary-container: '#5ec2f0'
  on-primary-container: '#004e68'
  inverse-primary: '#006687'
  secondary: '#9bcbfb'
  on-secondary: '#003354'
  secondary-container: '#0f4a73'
  on-secondary-container: '#8ab9e9'
  tertiary: '#ffc891'
  on-tertiary: '#492900'
  tertiary-container: '#f5a64c'
  on-tertiary-container: '#6a3d00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#c1e8ff'
  primary-fixed-dim: '#73d1ff'
  on-primary-fixed: '#001e2b'
  on-primary-fixed-variant: '#004d67'
  secondary-fixed: '#cee5ff'
  secondary-fixed-dim: '#9bcbfb'
  on-secondary-fixed: '#001d33'
  on-secondary-fixed-variant: '#0f4a73'
  tertiary-fixed: '#ffdcbc'
  tertiary-fixed-dim: '#ffb86c'
  on-tertiary-fixed: '#2c1600'
  on-tertiary-fixed-variant: '#683c00'
  background: '#021522'
  on-background: '#d1e5f8'
  surface-variant: '#243745'
typography:
  display-lg:
    fontFamily: JetBrains Mono
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
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
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
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

The design system is engineered for technical precision and data density, catering to indie developers who require actionable, explainable insights. The brand personality is rooted in the aesthetic of high-stakes environments like trading terminals and engineering dashboards—where clarity and speed of information retrieval are paramount.

The visual style is **Modern/Corporate** with a lean toward **Minimalism**. It prioritizes functional utility over aesthetic decoration. Surfaces are kept flat to minimize visual noise, and the hierarchy is driven strictly by typography and contrast. The emotional response should be one of confidence and clinical objectivity; the UI acts as a transparent lens through which the data is viewed.

## Colors

The palette is optimized for long-duration focus in low-light environments. 

- **Foundation:** The system uses a deep navy base (`#0B1420`) to reduce eye strain, with layered surfaces creating depth through tonal shifts rather than shadows.
- **Accents:** The primary accent (`#5EC2F0`) is reserved for interactive elements and critical Key Performance Indicators (KPIs). 
- **Semantics:** Color is used exclusively as a data signal. `Success` and `Danger` values are applied to SHAP (Shapley Additive Explanations) values and sentiment analysis to provide immediate visual cues for positive or negative impact on game performance.
- **Data Series:** Secondary accents should use cool-toned blues and greys to maintain a professional, analytical tone.

## Typography

This design system utilizes a dual-font strategy to distinguish between UI narrative and raw data.

- **Inter:** The primary typeface for all UI labels, navigation, and body copy. It provides maximum legibility at small sizes common in data-dense dashboards.
- **JetBrains Mono:** Used exclusively for numeric values, statistics, and code-related technical data. The monospaced nature ensures that numbers align perfectly in tables and KPI stacks, allowing users to compare values by scanning vertically.
- **Hierarchy:** Use `label-caps` for secondary metadata and table headers to create a distinct structural break from data rows.

## Layout & Spacing

The layout follows a **Fluid Grid** model with a hard constraint on maximum width to ensure readability on ultra-wide monitors.

- **Grid Model:** A 12-column system with a 16px gutter.
- **Responsive Behavior:** 
  - **Desktop:** 24px margins, standard 24px card padding. 
  - **Tablet:** Gutter reduces to 12px, content reflows to 6-column or 1-column stacks.
  - **Mobile:** 16px margins, card padding reduces to 16px to maximize data real-estate.
- **Rhythm:** All component heights and internal spacing must adhere to the 4px/8px baseline increment to maintain a rigorous, "engineered" alignment.

## Elevation & Depth

This design system uses a **Tonal Layering** and **Low-Contrast Outline** approach rather than traditional shadows.

- **Surface Tiers:** 
  - Level 0: `bg/base` (The main application canvas).
  - Level 1: `bg/surface` (Primary cards and containers).
  - Level 2: `bg/surface-raised` (Hovered states or nested modules).
- **Outlines:** Every container at Level 1 and Level 2 must have a 1px solid border of `border/subtle`. 
- **Shadows:** Shadows are prohibited for standard UI elements. They are reserved exclusively for temporary "floating" components like dropdown menus, tooltips, and modals to provide a clear separation from the logic-heavy layers beneath. Use a concentrated dark shadow: `0 8px 24px rgba(0,0,0,0.35)`.

## Shapes

The shape language is controlled and geometric, reflecting the precision of the data it contains.

- **Cards:** Use `12px` (rounded-lg) for large containers to soften the technical edge slightly without appearing "playful."
- **Interactive Elements:** Buttons and Input fields use `8px` (rounded) for a more compact, focused appearance.
- **Badges/Labels:** Use `pill-shaped` (fully rounded) to immediately distinguish categorical labels or status indicators from buttons and structural containers.

## Components

### Buttons
- **Primary:** `accent/primary` fill with `bg/base` text. No shadow. Bold typography.
- **Secondary:** Transparent background with a `1px` border in `border/subtle`. Text is `text/primary`.
- **Ghost:** No border or fill. Text is `text/secondary`. Use for low-priority actions.
- **Danger:** `semantic/danger` fill with `bg/base` text.

### Cards
- **Standard:** `bg/surface` background with `border/subtle` outline. Use for secondary modules.
- **Emphasis:** `bg/surface-raised` background. Use for primary data focus or active states.

### Badges & Chips
- **Status Badges:** Use the pill shape. Fill color should be the semantic color (Success/Warning/Danger) set to 15% opacity. Text must be the same semantic color at 100% opacity for high legibility.

### Navigation
- **Top Navbar:** 64px height, `bg/base` background, and a 1px bottom border using `border/subtle`.
- **Sub-nav:** Horizontal tab bar. Active states are indicated by a 2px bottom stroke in `accent/primary`.

### Form Fields
- **Inputs:** `bg/base` fill, `border/subtle` stroke. On focus, the border changes to `accent/primary`.
- **Labels:** Always use `body-sm` in `text/muted` above the input.

### Charts & Visualization
- **Style:** Flat fills only. No gradients or 3D effects.
- **Gridlines:** Use `border/subtle` at 50% opacity. 
- **Icons:** Use Lucide (outline style). 20px for navigation, 24px for card headers, and 16px for inline text or small data rows.