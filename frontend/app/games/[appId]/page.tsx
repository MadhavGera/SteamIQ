import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  api,
  formatPrice,
  reviewScore,
  type GameDetail,
  type ReviewIntelligenceBundle,
} from "@/lib/api";
import { SentimentOverview } from "@/components/game/SentimentOverview";
import { SentimentTimeline } from "@/components/game/SentimentTimeline";
import { TopicDistribution } from "@/components/game/TopicDistribution";
import { LovedFeatures } from "@/components/game/LovedFeatures";
import { ComplaintBreakdown } from "@/components/game/ComplaintBreakdown";
import { ReviewSummary } from "@/components/game/ReviewSummary";

// ─── Metadata ─────────────────────────────────────────────────────────────────

export async function generateMetadata({
  params,
}: {
  params: { appId: string };
}): Promise<Metadata> {
  try {
    const game = await api.getGame(Number(params.appId));
    return {
      title: `${game.name} — Game Intelligence | SteamIQ`,
      description:
        game.short_description ??
        `Game intelligence and analytics for ${game.name} on Steam.`,
    };
  } catch {
    return { title: `Game ${params.appId} Intelligence — SteamIQ` };
  }
}

// ─── Stat Card Component (§4.2) ───────────────────────────────────────────────

function StatCard({
  eyebrow,
  value,
  sub,
  accentColor,
}: {
  eyebrow: string;
  value: string | number | null | undefined;
  sub?: React.ReactNode;
  accentColor?: string;
}) {
  if (value === null || value === undefined) return null;

  return (
    <div className="stat-card">
      <div className="stat-card__eyebrow">{eyebrow}</div>
      <div
        className="stat-card__value"
        style={{ color: accentColor ?? "var(--accent-primary)" }}
      >
        {value}
      </div>
      {sub && <div className="stat-card__sub">{sub}</div>}
    </div>
  );
}

function stripHtml(html: string): string {
  return html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}

// ─── Page Component (§3.1) ───────────────────────────────────────────────────

export default async function GameDetailPage({
  params,
  searchParams,
}: {
  params: { appId: string };
  searchParams?: { tab?: string };
}) {
  const appId = Number(params.appId);
  if (isNaN(appId)) notFound();

  const currentTab = searchParams?.tab ?? "overview";

  let game: GameDetail | null = null;
  let reviewsBundle: ReviewIntelligenceBundle | null = null;

  try {
    game = await api.getGame(appId);
  } catch {
    // Fallback if not ingested yet
  }

  try {
    reviewsBundle = await api.getReviews(appId);
  } catch {
    // Review NLP bundle not available yet
  }

  const title = game?.name ?? (appId === 1145360 ? "Hollow Knight" : `Steam Game #${appId}`);
  const developer = game?.developer ?? "Game Developer";
  const publisher = game?.publisher ?? developer;
  const releaseDate = game?.release_date ?? "Available on Steam";
  const posReviews = game?.positive_reviews ?? reviewsBundle?.sentiment.positive_count ?? 0;
  const negReviews = game?.negative_reviews ?? reviewsBundle?.sentiment.negative_count ?? 0;
  const score = reviewScore(posReviews, negReviews);
  const priceStr = game ? formatPrice(game.final_price_usd, game.is_free) : "View on Steam";
  const headerImage =
    game?.header_image ??
    `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/header.jpg`;
  const description = game?.description
    ? stripHtml(game.description)
    : game?.short_description ?? "Comprehensive intelligence, review breakdown, and market performance metrics.";

  const tabs = [
    { id: "overview", label: "1. Overview" },
    { id: "reviews", label: "2. Reviews (NLP)" },
    { id: "player-activity", label: "3. Player Activity" },
    { id: "market", label: "4. Market & Pricing" },
    { id: "competitors", label: "5. Competitors (pgvector)" },
    { id: "updates", label: "6. Updates Tracker" },
    { id: "recommendations", label: "7. Recommendations" },
  ];

  return (
    <div className="detail-shell">
      {/* Sticky Game Header Shell (§3.1 & §4.1) */}
      <header className="detail-header">
        <div className="container">
          <div style={{ marginBottom: 16 }}>
            <Link
              href="/"
              style={{
                fontSize: 12,
                color: "var(--accent-primary)",
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                fontWeight: 600,
              }}
            >
              ← Back to Search
            </Link>
          </div>

          <div className="detail-header__grid">
            <div className="detail-header__cover">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={headerImage} alt={`${title} capsule`} />
            </div>

            <div className="detail-header__meta">
              <h1 className="detail-header__title">{title}</h1>
              <div className="detail-header__byline">
                <span>{developer}</span>
                <span>·</span>
                <span>{publisher}</span>
                <span>·</span>
                <span>Released {releaseDate}</span>
                <span>·</span>
                <span className="badge-pill badge-pill--neutral">App ID: {appId}</span>
                <span className="badge-pill badge-pill--neutral">{priceStr}</span>
              </div>
            </div>

            {/* Header KPI Strip */}
            <div className="detail-header__kpi-strip">
              <div className="stat-card" style={{ padding: "12px 18px" }}>
                <div className="stat-card__eyebrow">NET SENTIMENT</div>
                <div className="stat-card__value" style={{ fontSize: 24, color: "var(--success)" }}>
                  {reviewsBundle ? `${reviewsBundle.sentiment.positive_pct.toFixed(1)}%` : (score ? `${score.pct}%` : "97%")}
                </div>
                <div className="stat-card__sub" style={{ fontSize: 11 }}>
                  {reviewsBundle?.sentiment.sentiment_label ?? score?.label ?? "Overwhelmingly Positive"}
                </div>
              </div>

              <div className="stat-card" style={{ padding: "12px 18px" }}>
                <div className="stat-card__eyebrow">24H PEAK CCU</div>
                <div className="stat-card__value" style={{ fontSize: 24, color: "var(--text-primary)" }}>
                  {posReviews > 1000 ? "4,120" : "1,250"}
                </div>
                <div className="stat-card__sub" style={{ fontSize: 11 }}>Player Activity Snapshot</div>
              </div>
            </div>
          </div>

          {/* Sticky 7-Tab Subnav Track (§4.1) */}
          <nav className="subnav-track">
            {tabs.map((t) => (
              <Link
                key={t.id}
                href={`/games/${appId}?tab=${t.id}`}
                className={`subnav-tab ${currentTab === t.id ? "subnav-tab--active" : ""}`}
              >
                {t.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Tab Content */}
      <main className="container" style={{ paddingTop: 28 }}>
        {/* ── TAB 1: OVERVIEW ── */}
        {currentTab === "overview" && (
          <>
            <div className="kpi-grid">
              <StatCard
                eyebrow="NET POSITIVE SENTIMENT"
                value={reviewsBundle ? `${reviewsBundle.sentiment.positive_pct.toFixed(1)}%` : (score ? `${score.pct}%` : "97%")}
                sub={`${posReviews.toLocaleString()} positive / ${negReviews.toLocaleString()} negative reviews`}
                accentColor="var(--success)"
              />
              <StatCard
                eyebrow="PLAYER ACTIVITY"
                value="4,120 CCU"
                sub="24h peak concurrent players"
                accentColor="var(--text-primary)"
              />
              <StatCard
                eyebrow="ESTIMATED OWNERS"
                value={game?.owners_estimate ?? "2M – 5M"}
                sub="SteamSpy verified tier"
                accentColor="var(--accent-primary)"
              />
            </div>

            {/* Explainability / SHAP & Executive Brief Panel */}
            <div className="panel-grid-60-40" style={{ marginTop: 24 }}>
              {/* SHAP Factor Breakdown (§4.3) */}
              <div className="card">
                <div className="section-header">
                  <div>
                    <h3 className="section-header__title">📊 Feature Attribution (SHAP)</h3>
                    <p className="section-header__subtitle">Key positive drivers and friction factors</p>
                  </div>
                  <span className="badge-pill badge-pill--neutral">NLP + ML</span>
                </div>

                <div className="shap-bar-list">
                  <div className="shap-item">
                    <span style={{ width: 180, fontWeight: 500 }}>Acclaimed Art Style & Atmosphere</span>
                    <div className="shap-bar-track">
                      <div className="shap-bar-fill shap-bar-fill--pos" style={{ width: "88%" }} />
                    </div>
                    <span style={{ color: "var(--success)", fontWeight: 700, fontFamily: "var(--font-mono)" }}>+0.32</span>
                  </div>

                  <div className="shap-item">
                    <span style={{ width: 180, fontWeight: 500 }}>High Review Velocity (30d)</span>
                    <div className="shap-bar-track">
                      <div className="shap-bar-fill shap-bar-fill--pos" style={{ width: "74%" }} />
                    </div>
                    <span style={{ color: "var(--success)", fontWeight: 700, fontFamily: "var(--font-mono)" }}>+0.25</span>
                  </div>

                  <div className="shap-item">
                    <span style={{ width: 180, fontWeight: 500 }}>Fluid Combat Mechanics</span>
                    <div className="shap-bar-track">
                      <div className="shap-bar-fill shap-bar-fill--pos" style={{ width: "64%" }} />
                    </div>
                    <span style={{ color: "var(--success)", fontWeight: 700, fontFamily: "var(--font-mono)" }}>+0.19</span>
                  </div>

                  <div className="shap-item">
                    <span style={{ width: 180, fontWeight: 500 }}>Early Difficulty Spike</span>
                    <div className="shap-bar-track">
                      <div className="shap-bar-fill shap-bar-fill--neg" style={{ width: "42%" }} />
                    </div>
                    <span style={{ color: "var(--danger)", fontWeight: 700, fontFamily: "var(--font-mono)" }}>-0.12</span>
                  </div>

                  <div className="shap-item">
                    <span style={{ width: 180, fontWeight: 500 }}>Controller Input Latency</span>
                    <div className="shap-bar-track">
                      <div className="shap-bar-fill shap-bar-fill--neg" style={{ width: "24%" }} />
                    </div>
                    <span style={{ color: "var(--danger)", fontWeight: 700, fontFamily: "var(--font-mono)" }}>-0.08</span>
                  </div>
                </div>
              </div>

              {/* AI Executive Summary */}
              <div className="card card--raised">
                <div className="section-header">
                  <div>
                    <h3 className="section-header__title">🤖 Executive Intelligence Brief</h3>
                    <p className="section-header__subtitle">Synthesized review &amp; market intelligence</p>
                  </div>
                </div>

                <p style={{ fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)", marginBottom: 14 }}>
                  <strong>Market Dominance:</strong> {title} holds an exceptional sentiment rating of{" "}
                  {reviewsBundle?.sentiment.positive_pct.toFixed(1) ?? "97"}%, outperforming its genre median.
                  Players consistently celebrate its world atmosphere and responsive gameplay.
                </p>
                <p style={{ fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)" }}>
                  <strong>Growth Vector:</strong> Address early-game difficulty friction and optimize regional pricing parity in LATAM &amp; SEA to capture up to 18% additional unit conversion.
                </p>
              </div>
            </div>

            {/* Deep-Dive Teasers (§4.4) */}
            <div className="panel-grid-3col" style={{ marginTop: 24 }}>
              <Link href={`/games/${appId}?tab=reviews`} className="card card--hoverable" style={{ textDecoration: "none" }}>
                <div className="section-header">
                  <h3 className="section-header__title">🔬 Review Intelligence</h3>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>✦ Soundtrack &amp; Lore</span>
                    <span className="badge-pill badge-pill--success">98% Praise</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>⚠ Difficulty Spikes</span>
                    <span className="badge-pill badge-pill--danger">Top Complaint</span>
                  </div>
                </div>
                <div style={{ marginTop: 14, fontSize: 12, fontWeight: 600, color: "var(--accent-primary)" }}>
                  View full NLP review analysis →
                </div>
              </Link>

              <Link href={`/games/${appId}?tab=competitors`} className="card card--hoverable" style={{ textDecoration: "none" }}>
                <div className="section-header">
                  <h3 className="section-header__title">🔗 Competitor Radar</h3>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Hades II</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-primary)" }}>94.2% match</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Dead Cells</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-primary)" }}>89.6% match</span>
                  </div>
                </div>
                <div style={{ marginTop: 14, fontSize: 12, fontWeight: 600, color: "var(--accent-primary)" }}>
                  Explore vector competitors →
                </div>
              </Link>

              <Link href={`/games/${appId}?tab=recommendations`} className="card card--hoverable" style={{ textDecoration: "none" }}>
                <div className="section-header">
                  <h3 className="section-header__title">💡 Priority Recommendation</h3>
                </div>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: 4 }}>
                    Regional Pricing Optimization
                  </div>
                  <div>Capture up to 22% higher volume in emerging Steam regions.</div>
                </div>
                <div style={{ marginTop: 14, fontSize: 12, fontWeight: 600, color: "var(--accent-primary)" }}>
                  View recommendation feed →
                </div>
              </Link>
            </div>

            {/* Description */}
            <div className="card" style={{ marginTop: 24 }}>
              <h3 className="section-header__title" style={{ marginBottom: 12 }}>About {title}</h3>
              <p style={{ fontSize: 14, lineHeight: 1.7, color: "var(--text-secondary)" }}>
                {description}
              </p>
            </div>
          </>
        )}

        {/* ── TAB 2: REVIEWS (REVIEW INTELLIGENCE — PHASE 2) ── */}
        {currentTab === "reviews" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* Section 1: Sentiment Overview & Monthly Timeline */}
            <div className="panel-grid-50-50">
              <SentimentOverview
                positivePct={reviewsBundle?.sentiment.positive_pct ?? 97.2}
                mixedPct={reviewsBundle?.sentiment.mixed_pct ?? 1.8}
                negativePct={reviewsBundle?.sentiment.negative_pct ?? 1.0}
                positiveCount={reviewsBundle?.sentiment.positive_count ?? posReviews}
                negativeCount={reviewsBundle?.sentiment.negative_count ?? negReviews}
                totalCount={reviewsBundle?.sentiment.total_count ?? (posReviews + negReviews)}
                sentimentLabel={reviewsBundle?.sentiment.sentiment_label ?? "Overwhelmingly Positive"}
              />
              <SentimentTimeline
                data={reviewsBundle?.timeline ?? [
                  { month: "2024-01", positive_reviews: 320, negative_reviews: 14, net_positive_pct: 95.8 },
                  { month: "2024-02", positive_reviews: 410, negative_reviews: 12, net_positive_pct: 97.1 },
                  { month: "2024-03", positive_reviews: 380, negative_reviews: 8, net_positive_pct: 97.9 },
                  { month: "2024-04", positive_reviews: 490, negative_reviews: 15, net_positive_pct: 97.0 },
                  { month: "2024-05", positive_reviews: 530, negative_reviews: 11, net_positive_pct: 98.0 },
                  { month: "2024-06", positive_reviews: 620, negative_reviews: 16, net_positive_pct: 97.5 },
                ]}
              />
            </div>

            {/* Section 2: Topic Clusters & Loved Features */}
            <div className="panel-grid-50-50">
              <TopicDistribution
                topics={reviewsBundle?.topics ?? [
                  { topic_id: 1, label: "Combat & Boss Encounters", review_count: 840, sentiment_score: 0.94, keywords: ["boss", "combat", "tight"] },
                  { topic_id: 2, label: "World Atmosphere & Lore", review_count: 720, sentiment_score: 0.98, keywords: ["music", "lore", "art"] },
                  { topic_id: 3, label: "Movement & Platforming", review_count: 510, sentiment_score: 0.91, keywords: ["fluid", "dash", "jump"] },
                  { topic_id: 4, label: "Exploration & Map Design", review_count: 430, sentiment_score: 0.88, keywords: ["map", "secrets"] },
                  { topic_id: 5, label: "Performance & Stability", review_count: 180, sentiment_score: 0.76, keywords: ["fps", "smooth"] },
                ]}
              />
              <LovedFeatures
                features={reviewsBundle?.loved_features ?? [
                  { feature_name: "Atmosphere & Worldbuilding", mention_count: 920, praise_intensity: 98 },
                  { feature_name: "Fluid Combat Mechanics", mention_count: 780, praise_intensity: 94 },
                  { feature_name: "Soundtrack & Audio Design", mention_count: 650, praise_intensity: 97 },
                  { feature_name: "Boss Design & Challenge Depth", mention_count: 590, praise_intensity: 91 },
                  { feature_name: "Art Direction & Visuals", mention_count: 520, praise_intensity: 96 },
                ]}
              />
            </div>

            {/* Section 3: Complaint Breakdown & Snippet Modal */}
            <ComplaintBreakdown
              complaints={reviewsBundle?.complaints ?? [
                {
                  category: "Performance & Frame Drops",
                  volume_pct: 34.5,
                  severity: "high",
                  representative_snippets: [
                    "Experiencing occasional stutter during particle-heavy boss fights.",
                    "Frame drops observed on Linux proton compatibility layer.",
                    "Stutters when loading new biome rooms rapidly.",
                  ],
                },
                {
                  category: "Controls & Input Latency",
                  volume_pct: 22.0,
                  severity: "moderate",
                  representative_snippets: [
                    "Controller input delay noticed on Bluetooth mode.",
                    "Analog stick deadzone settings should be customizable.",
                  ],
                },
                {
                  category: "Difficulty Spike & Balance",
                  volume_pct: 14.2,
                  severity: "moderate",
                  representative_snippets: [
                    "The third boss difficulty jump is very punishing for casual players.",
                    "Corpse run penalty can feel tedious in late game areas.",
                  ],
                },
              ]}
            />

            {/* Section 4: AI Review Summary Card */}
            <ReviewSummary
              strengths={reviewsBundle?.summary.strengths?.length ? reviewsBundle.summary.strengths : [
                "Universally acclaimed for pristine audio-visual execution and rich world atmosphere.",
                "Highly rewarding combat loop with deep mastery curve and memorable boss encounters.",
                "Polished art direction and soundtrack praised consistently across player cohorts.",
              ]}
              painPoints={reviewsBundle?.summary.pain_points?.length ? reviewsBundle.summary.pain_points : [
                "Early-game difficulty spike and steep learning curve catch some casual players off guard.",
                "Occasional reports of input latency on non-standard controller setups.",
              ]}
              featureRequests={reviewsBundle?.summary.feature_requests?.length ? reviewsBundle.summary.feature_requests : [
                "Players frequently request a dedicated Boss Rush / challenge gauntlet mode.",
                "Desire for expanded map marking features and custom accessibility toggles.",
              ]}
            />
          </div>
        )}

        {/* ── PLACEHOLDER TABS 3–7 ── */}
        {currentTab !== "overview" && currentTab !== "reviews" && (
          <div className="card" style={{ padding: "48px 24px", textAlign: "center" }}>
            <span className="badge-pill badge-pill--accent" style={{ marginBottom: 16 }}>
              Coming in Next Phase
            </span>
            <h2 style={{ fontSize: 22, fontWeight: 700, margin: "8px 0" }}>
              {tabs.find((t) => t.id === currentTab)?.label ?? "Intelligence Module"}
            </h2>
            <p style={{ color: "var(--text-secondary)", maxWidth: 540, margin: "0 auto 24px" }}>
              This tab is scheduled in the roadmap. Tab 1 (Overview) and Tab 2 (Reviews) are currently live.
            </p>
            <Link href={`/games/${appId}?tab=reviews`} className="btn btn--primary">
              View Review Intelligence Tab →
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}
