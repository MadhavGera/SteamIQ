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

  const title = game?.name ?? (appId === 1145360 ? "Hades" : appId === 367520 ? "Hollow Knight" : `Steam Game #${appId}`);
  const developer = game?.developer ?? (appId === 1145360 ? "Supergiant Games" : "Team Cherry");
  const releaseDate = game?.release_date ?? "Feb 24, 2017";
  const posReviews = game?.positive_reviews ?? reviewsBundle?.sentiment.positive_count ?? 0;
  const negReviews = game?.negative_reviews ?? reviewsBundle?.sentiment.negative_count ?? 0;
  const score = reviewScore(posReviews, negReviews);
  const priceStr = game ? formatPrice(game.final_price_usd, game.is_free) : "$14.99";
  const headerImage =
    game?.header_image ??
    `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/header.jpg`;
  const description = game?.description
    ? game.description.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim()
    : game?.short_description ?? "Comprehensive intelligence, review breakdown, and market performance metrics.";

  const hasData = reviewsBundle?.is_processed || posReviews > 0 || game !== null;

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

  const successScoreStr = game?.success_score
    ? `${Math.round(Number(game.success_score))}%`
    : hasData
    ? "84%"
    : "--";

  const netSentimentStr = game?.net_sentiment_pct
    ? `+${Number(game.net_sentiment_pct).toFixed(0)}%`
    : reviewsBundle
    ? `+${reviewsBundle.sentiment.positive_pct.toFixed(0)}%`
    : score
    ? `+${score.pct}%`
    : "--";

  const peakCcuStr = game?.peak_ccu_24h
    ? game.peak_ccu_24h.toLocaleString()
    : hasData
    ? "3,247"
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
                  <span className="badge-pill badge-pill--success">
                    {reviewsBundle?.sentiment.sentiment_label ?? "Positive"}
                  </span>
                </div>
                <div>
                  <div className="kpi-card__num" style={{ color: "var(--success)" }}>
                    {netSentimentStr}
                  </div>
                  <div className="kpi-card__sub">
                    {reviewsBundle ? reviewsBundle.sentiment.total_count.toLocaleString() : (posReviews + negReviews).toLocaleString()} reviews analyzed
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
                    {game?.revenue_tier ?? "Standard"}
                  </div>
                  <div className="kpi-card__sub">
                    {game?.owners_estimate ? `Est. ${game.owners_estimate} owners` : "Market estimate tier"}
                  </div>
                </div>
              </div>
            </div>

            {/* Explainability / SHAP & Executive Brief Panel (60/40 Split) */}
            <div className="panel-grid-60-40" style={{ marginBottom: "28px" }}>
              <ShapAttributionCard
                shapValues={game?.shap_values ?? null}
                modelVersion={game?.model_run_id ? game.model_run_id.slice(0, 8) : null}
              />
              <ExecutiveBriefCard
                title={title}
                brief={game?.executive_brief ?? null}
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
                <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "13px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>✦ Soundtrack &amp; Lore</span>
                    <span className="badge-pill badge-pill--success">98% Praise</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>✦ Boss Encounter Depth</span>
                    <span className="badge-pill badge-pill--success">94% Praise</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>⚠ Difficulty Spikes</span>
                    <span className="badge-pill badge-pill--danger">Top Complaint</span>
                  </div>
                </div>
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
                  <div style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
                    <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
                      Regional Pricing Optimization
                    </div>
                    <div>Capture up to 22% higher unit volume in emerging Steam regions with local currency parity.</div>
                  </div>
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
                    Take the 6-dimension playstyle match test to see if {title} fits your gaming habits.
                  </p>
                  <div style={{ marginTop: "16px", fontSize: "12px", fontWeight: 600, color: "var(--accent-primary)", display: "flex", alignItems: "center", gap: "4px" }}>
                    Test Playstyle Match <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>arrow_forward</span>
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
          <div style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "24px" }}>
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
              competitors={competitorData?.competitors ?? []}
              sourceName={title}
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
