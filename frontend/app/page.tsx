"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import {
  api,
  formatPrice,
  reviewScore,
  type GameSummary,
  type GameSearchResult,
} from "@/lib/api";

// Sample games with actual Steam metadata to showcase the UI immediately
const CURATED_SHOWCASE: GameSummary[] = [
  {
    app_id: 1145360,
    name: "Hollow Knight: Silksong",
    short_description: "Explore a vast, haunted kingdom in Hollow Knight: Silksong! The sequel to the award-winning action-adventure.",
    header_image: "https://cdn.cloudflare.steamstatic.com/steam/apps/1145360/header.jpg",
    developer: "Team Cherry",
    publisher: "Team Cherry",
    release_date: "To be announced",
    is_free: false,
    final_price_usd: "19.99",
    discount_pct: 0,
    positive_reviews: 48500,
    negative_reviews: 1200,
    owners_estimate: "2,000,000 .. 5,000,000",
    genres: [
      { id: "1", description: "Action" },
      { id: "25", description: "Adventure" },
      { id: "23", description: "Indie" },
    ],
  },
  {
    app_id: 1145350,
    name: "Hades II",
    short_description: "Battle beyond the Underworld using dark sorcery to take on the Titan of Time in this god-like rogue-like dungeon crawler.",
    header_image: "https://cdn.cloudflare.steamstatic.com/steam/apps/1145350/header.jpg",
    developer: "Supergiant Games",
    publisher: "Supergiant Games",
    release_date: "May 6, 2024",
    is_free: false,
    final_price_usd: "29.99",
    discount_pct: 0,
    positive_reviews: 54200,
    negative_reviews: 1800,
    owners_estimate: "1,000,000 .. 2,000,000",
    genres: [
      { id: "1", description: "Action" },
      { id: "3", description: "RPG" },
      { id: "23", description: "Indie" },
    ],
  },
  {
    app_id: 413150,
    name: "Stardew Valley",
    short_description: "You've inherited your grandfather's old farm plot in Stardew Valley. Armed with hand-me-down tools and a few coins, you set out to begin your new life.",
    header_image: "https://cdn.cloudflare.steamstatic.com/steam/apps/413150/header.jpg",
    developer: "ConcernedApe",
    publisher: "ConcernedApe",
    release_date: "Feb 26, 2016",
    is_free: false,
    final_price_usd: "14.99",
    discount_pct: 0,
    positive_reviews: 620000,
    negative_reviews: 11000,
    owners_estimate: "20,000,000 .. 50,000,000",
    genres: [
      { id: "23", description: "Indie" },
      { id: "3", description: "RPG" },
      { id: "28", description: "Simulation" },
    ],
  },
  {
    app_id: 1091500,
    name: "Cyberpunk 2077",
    short_description: "Cyberpunk 2077 is an open-world, action-adventure RPG set in the megalopolis of Night City, where you play as a cyberpunk mercenary.",
    header_image: "https://cdn.cloudflare.steamstatic.com/steam/apps/1091500/header.jpg",
    developer: "CD PROJEKT RED",
    publisher: "CD PROJEKT RED",
    release_date: "Dec 10, 2020",
    is_free: false,
    final_price_usd: "59.99",
    discount_pct: 0,
    positive_reviews: 680000,
    negative_reviews: 140000,
    owners_estimate: "10,000,000 .. 20,000,000",
    genres: [
      { id: "3", description: "RPG" },
      { id: "1", description: "Action" },
      { id: "25", description: "Open World" },
    ],
  },
];

// ─── Game Card Component (§2.7, §2.10) ────────────────────────────────────────

function GameCard({ game }: { game: GameSummary }) {
  const score = reviewScore(game.positive_reviews, game.negative_reviews);
  const priceStr = formatPrice(game.final_price_usd, game.is_free);
  const genres = game.genres?.slice(0, 3) ?? [];

  let badgeClass = "badge-pill--neutral";
  if (score) {
    if (score.pct >= 80) badgeClass = "badge-pill--success";
    else if (score.pct < 60) badgeClass = "badge-pill--danger";
    else badgeClass = "badge-pill--warning";
  }

  return (
    <Link href={`/games/${game.app_id}`} className="game-card">
      <div className="game-card__media">
        {game.header_image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={game.header_image}
            alt={`${game.name} header`}
            className="game-card__image"
            loading="lazy"
          />
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: "2rem" }}>
            🎮
          </div>
        )}
        <div className="game-card__id-badge">
          ID: {game.app_id}
        </div>
      </div>

      <div className="game-card__content">
        <h3 className="game-card__title">{game.name}</h3>

        <div className="game-card__byline">
          {[game.developer, game.release_date].filter(Boolean).join(" · ")}
        </div>

        {genres.length > 0 && (
          <div className="game-card__tags">
            {genres.map((g) => (
              <span key={g.id} className="genre-chip">{g.description}</span>
            ))}
          </div>
        )}

        <div className="game-card__footer">
          <span className={`game-card__price ${game.is_free ? "free" : ""}`}>
            {priceStr}
          </span>

          {score && (
            <span className={`badge-pill ${badgeClass}`}>
              👍 {score.pct}% · {score.label}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

// ─── Main Landing Page ────────────────────────────────────────────────────────

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<GameSearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const doSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResult(null);
      return;
    }
    setLoading(true);
    try {
      const data = await api.searchGames(q, 1, 20);
      setResult(data);
    } catch {
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleTagClick = (name: string) => {
    setQuery(name);
    doSearch(name);
  };

  const copyCommand = () => {
    navigator.clipboard.writeText("make ingest APPID=1145360");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="container">
      {/* Hero Section */}
      <section className="hero">
        <h1 className="hero__title">
          Understand any Steam game<br />
          in seconds
        </h1>

        <p className="hero__subtitle">
          Sentiment analysis, competitor discovery, and ML-powered success prediction
          — pre-computed, explainable, and built for developers and publishers.
        </p>

        {/* Hero Search Box */}
        <div className="hero-search-wrapper">
          <form
            className="hero-search"
            onSubmit={(e) => {
              e.preventDefault();
              doSearch(query);
            }}
          >
            <div className="hero-search__icon">⌕</div>
            <input
              type="text"
              className="hero-search__input"
              placeholder="Search any game (e.g. Hollow Knight, Hades, Stardew Valley...)"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
            <button type="submit" className="hero-search__btn">
              {loading ? "Searching..." : "Analyze Game →"}
            </button>
          </form>

          {/* Quick-Search Chips */}
          <div className="quick-tags">
            <span className="quick-tags__label">Popular Games:</span>
            {[
              { name: "Hollow Knight", id: 1145360 },
              { name: "Hades II", id: 1145350 },
              { name: "Stardew Valley", id: 413150 },
              { name: "Cyberpunk 2077", id: 1091500 },
              { name: "Baldur's Gate 3", id: 1086940 },
            ].map((tag) => (
              <button
                key={tag.id}
                type="button"
                className="quick-tag-chip"
                onClick={() => handleTagClick(tag.name)}
              >
                <span>{tag.name}</span>
                <span className="quick-tag-chip__id">#{tag.id}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Live System Architecture Ticker Strip */}
      <div className="ticker-strip">
        <div className="ticker-item">
          <span className="ticker-item__label">Data Layer</span>
          <div className="ticker-item__value">
            <span style={{ color: "var(--success)" }}>●</span> PostgreSQL 16
          </div>
          <span className="ticker-item__sub">Zone-prefix schema (ADR 0001)</span>
        </div>

        <div className="ticker-item">
          <span className="ticker-item__label">Inference Engine</span>
          <div className="ticker-item__value">
            <span style={{ color: "var(--accent-primary)" }}>⚡</span> FastAPI Async
          </div>
          <span className="ticker-item__sub">Serving layer materialized reads</span>
        </div>

        <div className="ticker-item">
          <span className="ticker-item__label">Predictive Core</span>
          <div className="ticker-item__value">
            <span style={{ color: "var(--accent-primary)" }}>📈</span> XGBoost + LGBM
          </div>
          <span className="ticker-item__sub">Optuna tuned with SHAP explainability</span>
        </div>

        <div className="ticker-item">
          <span className="ticker-item__label">NLP Intelligence</span>
          <div className="ticker-item__value">
            <span style={{ color: "var(--warning)" }}>🔬</span> BERTopic + RoBERTa
          </div>
          <span className="ticker-item__sub">Hierarchical sentiment &amp; complaints</span>
        </div>
      </div>

      {/* Search Results OR Featured Showcase */}
      <section style={{ marginBottom: 48 }}>
        <div className="section-header">
          <div>
            <h2 className="section-header__title">
              {result && result.games.length > 0 ? (
                <>Database Search Results ({result.total} games found)</>
              ) : (
                <>Featured &amp; Ingested Games</>
              )}
            </h2>
            <p className="section-header__subtitle">
              {result && result.games.length > 0
                ? `Showing matches for query "${result.query}"`
                : "Click any game card to explore the full Game Intelligence dashboard"}
            </p>
          </div>
        </div>

        <div className="games-grid">
          {result && result.games.length > 0 ? (
            result.games.map((g) => <GameCard key={g.app_id} game={g} />)
          ) : (
            CURATED_SHOWCASE.map((g) => <GameCard key={g.app_id} game={g} />)
          )}
        </div>
      </section>

      {/* CLI Quick Ingestion Box */}
      <div className="cli-box">
        <div className="cli-box__content">
          <h3 className="cli-box__title">Ingest Any Steam Game Directly</h3>
          <p className="cli-box__subtitle">
            Want to analyze your own game or a specific competitor? Run the ingestion CLI command in your terminal:
          </p>
        </div>

        <div className="cli-code">
          <code>make ingest APPID=1145360</code>
          <button
            type="button"
            onClick={copyCommand}
            style={{
              background: "var(--bg-surface-raised)",
              color: copied ? "var(--success)" : "var(--text-secondary)",
              border: "1px solid var(--border-subtle)",
              padding: "4px 10px",
              borderRadius: "var(--radius-sm)",
              cursor: "pointer",
              fontSize: "11px",
              fontFamily: "var(--font-sans)",
              fontWeight: 600,
            }}
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
      </div>

      {/* 4-Module Platform Architecture Grid */}
      <section style={{ marginBottom: 64 }}>
        <div className="section-header">
          <div>
            <h2 className="section-header__title">Core Intelligence Modules</h2>
            <p className="section-header__subtitle">
              End-to-end data pipeline transforming raw Steam reviews and market metrics into actionable decisions
            </p>
          </div>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-card__phase">Phase 2 · NLP Core</div>
            <h3 className="feature-card__title">Review Intelligence</h3>
            <p className="feature-card__desc">
              Transformer-based sentiment scoring, unsupervised BERTopic discovery, loved features detection, and complaint clustering.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-card__phase">Phase 3 · Vector Search</div>
            <h3 className="feature-card__title">Competitor Radar</h3>
            <p className="feature-card__desc">
              pgvector sentence-transformer similarity engine calculating nearest game neighbors across tags, descriptions, and mechanics.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-card__phase">Phase 4 · Predictive ML</div>
            <h3 className="feature-card__title">Success Prediction</h3>
            <p className="feature-card__desc">
              XGBoost + LightGBM ensemble forecasting commercial success, complete with SHAP factor attributions and revenue tiering.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-card__phase">Phase 5 · Decision Engine</div>
            <h3 className="feature-card__title">Decision Intelligence</h3>
            <p className="feature-card__desc">
              Automated recommendation engine, update impact tracking, and comparable price spectrum analysis for commercial optimization.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
