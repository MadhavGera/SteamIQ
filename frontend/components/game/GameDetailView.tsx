"use client";

import React, { useMemo, useState, useEffect } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  formatPrice,
  reviewScore,
  type CompetitorList,
  type GameDetail,
  type MarketIntelligence,
  type RecommendationBundle,
  type ReviewIntelligenceBundle,
  type UpdateImpactFeed,
} from "@/lib/api";
import { useUserMode } from "@/lib/UserModeContext";
import { SentimentOverview } from "@/components/game/SentimentOverview";
import { SentimentTimeline } from "@/components/game/SentimentTimeline";
import { TopicDistribution } from "@/components/game/TopicDistribution";
import { LovedFeatures } from "@/components/game/LovedFeatures";
import { ComplaintBreakdown } from "@/components/game/ComplaintBreakdown";
import { ReviewSummary } from "@/components/game/ReviewSummary";
import { CompetitorTable } from "@/components/game/CompetitorTable";
import { CompetitorSentimentMatrix } from "@/components/game/CompetitorSentimentMatrix";
import { CompetitorRadarTeaser } from "@/components/game/CompetitorRadarTeaser";
import { ShapAttributionCard } from "@/components/game/ShapAttributionCard";
import { ExecutiveBriefCard } from "@/components/game/ExecutiveBriefCard";
import { MarketTab } from "@/components/game/MarketTab";
import { UpdatesTab } from "@/components/game/UpdatesTab";
import { RecommendationsTab } from "@/components/game/RecommendationsTab";
import { MatchTab } from "@/components/game/MatchTab";

interface GameDetailViewProps {
  appId: number;
  initialTab?: string;
  game: GameDetail | null;
  reviewsBundle: ReviewIntelligenceBundle | null;
  competitorData: CompetitorList | null;
  marketData: MarketIntelligence | null;
  updatesData: UpdateImpactFeed | null;
  recommendationsData: RecommendationBundle | null;
}

export function GameDetailView({
  appId,
  initialTab = "overview",
  game,
  reviewsBundle,
  competitorData,
  marketData,
  updatesData,
  recommendationsData,
}: GameDetailViewProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { isPlayer, isDeveloper } = useUserMode();
  // HONEST-FALLBACK: Real game name or unindexed app placeholder
  const title = game?.name ?? `Steam App #${appId}`;
  // HONEST-FALLBACK: Real developer or pending indicator
  const developer = game?.developer ?? "--";
  // HONEST-FALLBACK: Real release date or pending indicator
  const releaseDate = game?.release_date ?? "--";
  // HONEST-FALLBACK: Real positive reviews count or 0
  const posReviews = game?.positive_reviews ?? reviewsBundle?.sentiment.positive_count ?? 0;
  // HONEST-FALLBACK: Real negative reviews count or 0
  const negReviews = game?.negative_reviews ?? reviewsBundle?.sentiment.negative_count ?? 0;
  const score = (posReviews + negReviews > 0) ? reviewScore(posReviews, negReviews) : null;
  const priceStr = game ? formatPrice(game.final_price_usd, game.is_free) : "--";
  // HONEST-FALLBACK: Header image CDN url or default steam capsule URL
  const headerImage =
    game?.header_image ??
    `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/header.jpg`;
  const rawDesc = game?.description ? game.description.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim() : game?.short_description;
  // HONEST-FALLBACK: Real description or pending indicator
  const description = rawDesc ?? "--";

  const hasData = Boolean(reviewsBundle?.is_processed || posReviews > 0 || game !== null);

  // Compute Mode-Specific Tabs per Amended IA (Roadmap v2 §2 & Blueprint §1)
  const availableTabs = useMemo(() => {
    if (isPlayer) {
      return [
        { id: "overview", label: "Overview" },
        { id: "match", label: "Match (Is It For Me?)" },
        { id: "reviews", label: "Reviews" },
        { id: "player-activity", label: "Player Activity" },
        { id: "market", label: "Deals & Pricing" },
        { id: "competitors", label: "Similar Games" },
      ];
    }
    return [
      { id: "overview", label: "Overview" },
      { id: "reviews", label: "Reviews" },
      { id: "player-activity", label: "Player Activity" },
      { id: "market", label: "Market" },
      { id: "competitors", label: "Competitors" },
      { id: "updates", label: "Updates" },
      { id: "recommendations", label: "Recommendations" },
    ];
  }, [isPlayer]);

  // Determine current active tab
  const urlTab = searchParams.get("tab") || initialTab;
  const [activeTab, setActiveTab] = useState(urlTab);

  useEffect(() => {
    const isTabValid = availableTabs.some((t) => t.id === urlTab);
    if (!isTabValid) {
      // If current tab is hidden in current mode (e.g. Updates in Player Mode), default to overview
      setActiveTab("overview");
    } else {
      setActiveTab(urlTab);
    }
  }, [urlTab, availableTabs]);

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    router.push(`/games/${appId}?tab=${tabId}`, { scroll: false });
  };

  // HONEST-FALLBACK: Real metric values or explicit pending state indicator
  const successScoreStr = game?.success_score != null
    ? `${Math.round(Number(game.success_score))}%`
    : "--";

  // HONEST-FALLBACK: Real net sentiment percentage or explicit pending state
  const netSentimentStr = game?.net_sentiment_pct != null
    ? `+${Number(game.net_sentiment_pct).toFixed(0)}%`
    : reviewsBundle?.sentiment?.positive_pct != null
    ? `+${reviewsBundle.sentiment.positive_pct.toFixed(0)}%`
    : score?.pct != null
    ? `+${score.pct}%`
    : "--";

  // HONEST-FALLBACK: Real 24h peak CCU count or explicit pending state
  const peakCcuStr = game?.peak_ccu_24h != null
    ? game.peak_ccu_24h.toLocaleString()
    : "--";

  return (
    <div className="detail-shell">
      {/* ── Page Header / Hero Area (Stitch Design) ── */}
      <header className="detail-header">
        <div className="container">
          {/* Breadcrumb row with Mode indicator */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div className="detail-breadcrumb">
              <Link href="/" className="detail-breadcrumb__link">
                Catalog
              </Link>
              <span className="detail-breadcrumb__sep">/</span>
              <span className="detail-breadcrumb__current">{title}</span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: 700,
                  color: "var(--text-muted)",
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                }}
              >
                VIEWING AS:
              </span>
              <span
                className="badge-pill badge-pill--neutral"
                style={{
                  color: isPlayer ? "var(--accent-light)" : "var(--accent-primary)",
                  borderColor: "var(--border-subtle)",
                }}
              >
                {isPlayer ? "🎮 PLAYER MODE" : "🛠️ DEVELOPER MODE"}
              </span>
            </div>
          </div>

          {/* Hero Row: Capsule Art + Game Title + Quick KPI Strip */}
          <div className="detail-header__hero-row">
            <div className="detail-header__game-info">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={headerImage}
                alt={`${title} capsule art`}
                className="detail-header__cover-thumb"
              />
              <div>
                <div className="detail-header__title-row">
                  <h1 className="detail-header__h1">{title}</h1>
                  <span className="detail-header__app-badge">APP ID {appId}</span>
                </div>
                <div className="detail-header__byline">
                  <span>Dev: <strong>{developer}</strong></span>
                  <span>•</span>
                  <span>Released: {releaseDate}</span>
                  <span>•</span>
                  <span style={{ color: "var(--accent-primary)", fontWeight: 600 }}>{priceStr}</span>
                </div>
              </div>
            </div>

            {/* Quick KPI Strip */}
            <div className="detail-header__kpi-strip">
              <div className="header-kpi-item">
                <span className="header-kpi-item__label">SUCCESS SCORE</span>
                <span className="header-kpi-item__val" style={{ color: "var(--accent-light)" }}>
                  {successScoreStr}
                </span>
              </div>
              <div className="header-kpi-divider" />
              <div className="header-kpi-item">
                <span className="header-kpi-item__label">NET SENTIMENT</span>
                <span className="header-kpi-item__val" style={{ color: "var(--success)" }}>
                  {netSentimentStr}
                </span>
              </div>
              <div className="header-kpi-divider" />
              <div className="header-kpi-item">
                <span className="header-kpi-item__label">24H PEAK CCU</span>
                <span className="header-kpi-item__val" style={{ color: "var(--text-primary)" }}>
                  {peakCcuStr}
                </span>
              </div>
            </div>
          </div>

          {/* Sub Navigation Bar (Mode-Aware) */}
          <nav className="subnav-track">
            {availableTabs.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => handleTabChange(t.id)}
                className={`subnav-tab ${activeTab === t.id ? "subnav-tab--active" : ""}`}
                style={{ background: "transparent", border: "none", cursor: "pointer" }}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* ── Main Tab Content Area ── */}
      <main className="container" style={{ paddingBottom: "64px" }}>
        {/* ── NO-DATA / PROCESSING STATE ── */}
        {!hasData && (
          <div style={{ padding: "80px 24px", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
            <div
              style={{
                width: "64px",
                height: "64px",
                borderRadius: "50%",
                border: "1px solid var(--border-subtle)",
                backgroundColor: "var(--bg-surface)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: "24px",
                boxShadow: "var(--shadow-dropdown)",
              }}
            >
              <div className="loading-bars">
                <span />
                <span />
                <span />
              </div>
            </div>
            <h2 style={{ fontSize: "24px", fontWeight: 700, marginBottom: "8px" }}>
              Analysis in Progress
            </h2>
            <p style={{ fontSize: "14px", color: "var(--text-secondary)", maxWidth: "420px", lineHeight: "1.6" }}>
              We&apos;re crunching the numbers. Check back shortly for full intelligence.
            </p>
          </div>
        )}

        {/* ── TAB: OVERVIEW ── */}
        {hasData && activeTab === "overview" && (
          <div>
            {/* Section Header: Intelligence Matrix */}
            <div className="matrix-header">
              <div>
                <h2 className="matrix-header__title">Intelligence Matrix</h2>
                <p className="matrix-header__desc">
                  Complete predictive modeling and attribution breakdown
                </p>
              </div>
              <button className="btn btn--secondary" style={{ fontSize: "12px", padding: "6px 14px" }}>
                <span>Export Full Report</span>
                <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                  arrow_outward
                </span>
              </button>
            </div>

            {/* 4-Card Hero KPI Strip */}
            <div className="kpi-matrix-grid">
              <div className="kpi-card">
                <div className="kpi-card__top">
                  <span className="kpi-card__eyebrow">SUCCESS SCORE</span>
                  <span className="badge-pill badge-pill--neutral" style={{ color: "var(--accent-light)" }}>
                    {game?.model_run_id ? "Optuna ML" : "Composite"}
                  </span>
                </div>
                <div>
                  <div className="kpi-card__num" style={{ color: "var(--accent-light)" }}>
                    {successScoreStr}
                  </div>
                  <div className="kpi-card__sub">Optuna ensemble champion</div>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-card__top">
                  <span className="kpi-card__eyebrow">NET SENTIMENT</span>
                  <span className={`badge-pill ${reviewsBundle?.sentiment.sentiment_label ? "badge-pill--success" : "badge-pill--neutral"}`}>
                    {/* HONEST-FALLBACK: Sentiment label from reviews or game overview or Unscored */}
                    {reviewsBundle?.sentiment.sentiment_label ?? (game?.review_score_desc ?? "Unscored")}
                  </span>
                </div>
                <div>
                  <div className="kpi-card__num" style={{ color: "var(--success)" }}>
                    {netSentimentStr}
                  </div>
                  <div className="kpi-card__sub">
                    {reviewsBundle
                      ? `${reviewsBundle.sentiment.total_count.toLocaleString()} reviews analyzed`
                      : (posReviews + negReviews > 0)
                      ? `${(posReviews + negReviews).toLocaleString()} reviews recorded`
                      : "No reviews recorded"}
                  </div>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-card__top">
                  <span className="kpi-card__eyebrow">PLAYER ACTIVITY</span>
                </div>
                <div>
                  <div className="kpi-card__num" style={{ color: "var(--accent-light)" }}>
                    {peakCcuStr}
                  </div>
                  <div className="kpi-card__sub">24h peak CCU</div>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-card__top">
                  <span className="kpi-card__eyebrow">COMMERCIAL BRACKET</span>
                </div>
                <div>
                  <div className="kpi-card__num" style={{ fontSize: "24px", color: "var(--accent-light)" }}>
                    {/* HONEST-FALLBACK: Real revenue tier or pending indicator */}
                    {game?.revenue_tier ?? "--"}
                  </div>
                  <div className="kpi-card__sub">
                    {game?.owners_estimate ? `Est. ${game.owners_estimate} owners` : "Market estimate pending"}
                  </div>
                </div>
              </div>
            </div>

            {/* Explainability / SHAP & Executive Brief Panel (60/40 Split) */}
            <div className="panel-grid-60-40" style={{ marginBottom: "28px" }}>
              <ShapAttributionCard
                // HONEST-FALLBACK: Real SHAP attributions or null
                shapValues={game?.shap_values ?? null}
                modelVersion={game?.model_run_id ? game.model_run_id.slice(0, 8) : null}
              />
              <ExecutiveBriefCard
                title={title}
                // HONEST-FALLBACK: Real executive brief or null
                brief={game?.executive_brief ?? null}
                // HONEST-FALLBACK: Real net sentiment percentage or null
                netSentimentPct={
                  reviewsBundle?.sentiment.positive_pct ??
                  (game?.net_sentiment_pct ? Number(game.net_sentiment_pct) : null)
                }
              />
            </div>

            {/* Deep-Dive Teasers */}
            <div className="panel-grid-3col">
              <button
                type="button"
                onClick={() => handleTabChange("reviews")}
                className="card card--hoverable"
                style={{ textAlign: "left", cursor: "pointer", background: "var(--bg-surface)" }}
              >
                <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "12px" }}>
                  Top Review Themes
                </h3>
                {reviewsBundle?.topics && reviewsBundle.topics.length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "13px" }}>
                    {reviewsBundle.topics.slice(0, 3).map((t) => (
                      <div key={t.topic_id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ color: "var(--text-primary)" }}>✦ {t.label}</span>
                        <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px" }}>
                          {t.review_count} reviews
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6", margin: "0" }}>
                    Review topic clustering pending. Run review NLP pipeline to extract topic clusters.
                  </p>
                )}
                <div style={{ marginTop: "16px", fontSize: "12px", fontWeight: 600, color: "var(--accent-primary)", display: "flex", alignItems: "center", gap: "4px" }}>
                  View full NLP analysis <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>arrow_forward</span>
                </div>
              </button>

              {competitorData && competitorData.competitors.length > 0 ? (
                <CompetitorRadarTeaser appId={appId} competitors={competitorData.competitors} />
              ) : (
                <button
                  type="button"
                  onClick={() => handleTabChange("competitors")}
                  className="card card--hoverable"
                  style={{ textAlign: "left", cursor: "pointer", background: "var(--bg-surface)" }}
                >
                  <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "12px" }}>
                    {isPlayer ? "Similar Games" : "Competitors"}
                  </h3>
                  <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
                    Discover closest semantic benchmark titles and shared gameplay tags.
                  </p>
                  <div style={{ marginTop: "16px", fontSize: "12px", fontWeight: 600, color: "var(--accent-primary)", display: "flex", alignItems: "center", gap: "4px" }}>
                    View titles <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>arrow_forward</span>
                  </div>
                </button>
              )}

              {isDeveloper ? (
                <button
                  type="button"
                  onClick={() => handleTabChange("recommendations")}
                  className="card card--hoverable"
                  style={{ textAlign: "left", cursor: "pointer", background: "var(--bg-surface)" }}
                >
                  <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "12px" }}>
                    Priority Recommendation
                  </h3>
                  {recommendationsData?.recommendations && recommendationsData.recommendations.length > 0 ? (
                    <div style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
                      <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
                        {recommendationsData.recommendations[0].title}
                      </div>
                      <div style={{ display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                        {recommendationsData.recommendations[0].rationale}
                      </div>
                    </div>
                  ) : (
                    <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6", margin: "0" }}>
                      Recommendation pipeline pending. Run recommendation job to generate strategic action items.
                    </p>
                  )}
                  <div style={{ marginTop: "16px", fontSize: "12px", fontWeight: 600, color: "var(--accent-primary)", display: "flex", alignItems: "center", gap: "4px" }}>
                    View recommendation plan <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>arrow_forward</span>
                  </div>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => handleTabChange("match")}
                  className="card card--hoverable"
                  style={{ textAlign: "left", cursor: "pointer", background: "var(--bg-surface)" }}
                >
                  <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "12px" }}>
                    Taste Alignment
                  </h3>
                  <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
                    Adjust gameplay intensity sliders to see how this game matches your playstyle.
                  </p>
                  <div style={{ marginTop: "16px", fontSize: "12px", fontWeight: 600, color: "var(--accent-primary)", display: "flex", alignItems: "center", gap: "4px" }}>
                    Check Match <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>arrow_forward</span>
                  </div>
                </button>
              )}
            </div>

            {/* About / Description */}
            <div className="card" style={{ marginTop: "24px" }}>
              <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "10px" }}>
                About {title}
              </h3>
              <p style={{ fontSize: "13px", lineHeight: "1.7", color: "var(--text-secondary)" }}>
                {description}
              </p>
            </div>
          </div>
        )}

        {/* ── TAB: MATCH (PLAYER MODE ONLY) ── */}
        {hasData && activeTab === "match" && isPlayer && (
          <MatchTab appId={appId} gameTitle={title} />
        )}

        {/* ── TAB: REVIEWS ── */}
        {hasData && activeTab === "reviews" && (
          !reviewsBundle || !reviewsBundle.is_processed ? (
            <div className="card" style={{ padding: "64px 24px", textAlign: "center", marginTop: "24px" }}>
              <div
                style={{
                  width: "56px",
                  height: "56px",
                  borderRadius: "50%",
                  backgroundColor: "var(--bg-surface)",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginBottom: "16px",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: "28px", color: "var(--text-secondary)" }}>
                  rate_review
                </span>
              </div>
              <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "8px" }}>
                Review Analysis Not Yet Available
              </h2>
              <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "520px", margin: "0 auto", lineHeight: "1.6" }}>
                Review analysis is not yet available for this game. Run `make process-reviews` and `make materialize-marts` to generate sentiment timelines, NLP topic clusters, and complaint breakdowns.
              </p>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "24px" }}>
              <div className="panel-grid-50-50">
                <SentimentOverview
                  positivePct={reviewsBundle.sentiment.positive_pct}
                  mixedPct={reviewsBundle.sentiment.mixed_pct}
                  negativePct={reviewsBundle.sentiment.negative_pct}
                  positiveCount={reviewsBundle.sentiment.positive_count}
                  negativeCount={reviewsBundle.sentiment.negative_count}
                  totalCount={reviewsBundle.sentiment.total_count}
                  sentimentLabel={reviewsBundle.sentiment.sentiment_label}
                />
                <SentimentTimeline data={reviewsBundle.timeline} />
              </div>

              <div className="panel-grid-50-50">
                <TopicDistribution topics={reviewsBundle.topics} />
                <LovedFeatures features={reviewsBundle.loved_features} />
              </div>

              <ComplaintBreakdown complaints={reviewsBundle.complaints} />

              <ReviewSummary
                // HONEST-FALLBACK: Safe structural empty array for unpopulated summary items
                strengths={reviewsBundle.summary.strengths ?? []}
                // HONEST-FALLBACK: Safe structural empty array for unpopulated summary items
                painPoints={reviewsBundle.summary.pain_points ?? []}
                // HONEST-FALLBACK: Safe structural empty array for unpopulated summary items
                featureRequests={reviewsBundle.summary.feature_requests ?? []}
              />
            </div>
          )
        )}

        {/* ── TAB: MARKET ── */}
        {hasData && activeTab === "market" && (
          <MarketTab market={marketData} gameTitle={title} />
        )}

        {/* ── TAB: COMPETITORS / SIMILAR GAMES ── */}
        {hasData && activeTab === "competitors" && (
          <div className="tab-pane space-y-6 animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "24px" }}>
            {competitorData?.source_game && (
              <CompetitorSentimentMatrix
                sourceGame={competitorData.source_game}
                competitors={competitorData.competitors}
              />
            )}
            <CompetitorTable
              // HONEST-FALLBACK: Safe structural empty array for unpopulated competitors
              competitors={competitorData?.competitors ?? []}
              sourceName={title}
              // HONEST-FALLBACK: Boolean flag default
              isProcessed={competitorData?.is_processed ?? false}
            />
          </div>
        )}

        {/* ── TAB: UPDATES (DEV MODE ONLY) ── */}
        {hasData && activeTab === "updates" && isDeveloper && (
          <UpdatesTab updatesFeed={updatesData} gameTitle={title} />
        )}

        {/* ── TAB: RECOMMENDATIONS (DEV MODE ONLY) ── */}
        {hasData && activeTab === "recommendations" && isDeveloper && (
          <RecommendationsTab bundle={recommendationsData} gameTitle={title} />
        )}

        {/* ── TAB: PLAYER ACTIVITY ── */}
        {hasData && activeTab === "player-activity" && (
          <div className="card" style={{ padding: "64px 24px", textAlign: "center", marginTop: "24px" }}>
            <h2 style={{ fontSize: "22px", fontWeight: 700, margin: "8px 0" }}>
              Player Activity Telemetry
            </h2>
            <p style={{ color: "var(--text-secondary)", maxWidth: "520px", margin: "0 auto 24px", lineHeight: "1.6", fontSize: "14px" }}>
              High-frequency minute-level player CCU tracking and retention cohort charts are scheduled in Phase 7. Current 24h Peak ({peakCcuStr}) and lifetime playtime metrics ({game?.average_playtime_forever ? `${Math.round(game.average_playtime_forever / 60)}h avg` : "N/A"}) are active.
            </p>
            <div style={{ display: "flex", gap: "12px", justifyContent: "center" }}>
              <button type="button" onClick={() => handleTabChange("overview")} className="btn btn--secondary">
                Overview
              </button>
              <button type="button" onClick={() => handleTabChange("market")} className="btn btn--primary">
                {isPlayer ? "Deals & Pricing" : "Market Intelligence"}
                <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                  arrow_forward
                </span>
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
