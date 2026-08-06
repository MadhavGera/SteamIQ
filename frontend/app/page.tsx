"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import {
  api,
  formatPrice,
  reviewScore,
  type GameSummary,
  type GameSearchResult,
  SteamIQApiError,
} from "@/lib/api";

// ─── Skeleton cards ───────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <div className="skeleton-card__image" />
      <div className="skeleton-card__body">
        <div className="skeleton" style={{ height: 18, width: "75%" }} />
        <div className="skeleton" style={{ height: 13, width: "50%" }} />
        <div className="skeleton" style={{ height: 32, width: "100%" }} />
        <div className="skeleton" style={{ height: 13, width: "30%" }} />
      </div>
    </div>
  );
}

// ─── Game card ────────────────────────────────────────────────────────────────

function GameCard({ game }: { game: GameSummary }) {
  const score = reviewScore(game.positive_reviews, game.negative_reviews);
  const priceStr = formatPrice(game.final_price_usd, game.is_free);
  const genres = game.genres?.slice(0, 3) ?? [];

  let badgeClass = "review-badge--mixed";
  if (score) {
    if (score.pct >= 80) badgeClass = "review-badge--positive";
    else if (score.pct < 50) badgeClass = "review-badge--negative";
  }

  return (
    <Link href={`/games/${game.app_id}`} className="game-card">
      {game.header_image ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={game.header_image}
          alt={`${game.name} header`}
          className="game-card__image"
          loading="lazy"
        />
      ) : (
        <div className="game-card__image-placeholder">🎮</div>
      )}

      <div className="game-card__body">
        <h3 className="game-card__title">{game.name}</h3>

        <div className="game-card__meta">
          {game.developer && <span>{game.developer}</span>}
          {game.release_date && (
            <>
              <span style={{ opacity: 0.4 }}>·</span>
              <span>{game.release_date}</span>
            </>
          )}
        </div>

        {game.short_description && (
          <p className="game-card__desc">{game.short_description}</p>
        )}

        {genres.length > 0 && (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
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
            <span className={`review-badge ${badgeClass}`}>
              👍 {score.pct}%
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

// ─── Search form ──────────────────────────────────────────────────────────────

function HeroSearch({ onSearch }: { onSearch: (q: string) => void }) {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) onSearch(input.trim());
  };

  return (
    <form className="hero__search" onSubmit={handleSubmit} role="search">
      <input
        id="hero-search-input"
        type="search"
        className="hero__search-input"
        placeholder="Search any game — Hollow Knight, Hades, Stardew Valley..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        aria-label="Search games"
        autoComplete="off"
        autoFocus
      />
      <button
        id="hero-search-btn"
        type="submit"
        className="hero__search-btn"
        aria-label="Search"
      >
        ⌕
      </button>
    </form>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<GameSearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  const PAGE_SIZE = 20;

  const doSearch = useCallback(async (q: string, p = 1) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.searchGames(q, p, PAGE_SIZE);
      setResult(data);
      setPage(p);
    } catch (err) {
      if (err instanceof SteamIQApiError) {
        setError(err.message);
      } else {
        setError("Could not reach the SteamIQ API. Is the backend running?");
      }
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSearch = (q: string) => {
    setQuery(q);
    doSearch(q, 1);
  };

  const totalPages = result ? Math.ceil(result.total / PAGE_SIZE) : 0;

  return (
    <>
      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero__eyebrow">
            ⚡ Game Intelligence Platform
          </div>
          <h1 className="hero__title">
            Understand any <span className="gradient">Steam game</span>
            <br />in seconds
          </h1>
          <p className="hero__subtitle">
            Sentiment analysis, competitor discovery, and ML-powered success prediction
            — all pre-computed, all instant.
          </p>
          <HeroSearch onSearch={handleSearch} />
        </div>
      </section>

      {/* Results */}
      <section className="section">
        <div className="container">
          {/* Error */}
          {error && (
            <div className="error-banner" role="alert">
              ⚠️ {error}
            </div>
          )}

          {/* Loading skeletons */}
          {loading && (
            <>
              <div className="section__header">
                <div className="skeleton" style={{ height: 20, width: 180 }} />
              </div>
              <div className="game-grid">
                {Array.from({ length: 8 }).map((_, i) => (
                  <SkeletonCard key={i} />
                ))}
              </div>
            </>
          )}

          {/* Results */}
          {!loading && result && (
            <>
              <div className="section__header">
                <h2 className="section__title">
                  Results for &ldquo;{result.query}&rdquo;
                </h2>
                <span className="section__meta">
                  {result.total.toLocaleString()} game{result.total !== 1 ? "s" : ""} found
                </span>
              </div>

              {result.games.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state__icon">🔍</div>
                  <h3 className="empty-state__title">No games found</h3>
                  <p className="empty-state__body">
                    No games match &ldquo;{result.query}&rdquo; in the database.
                    Run <code style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-primary)" }}>make ingest APPID=&lt;id&gt;</code> to ingest a game.
                  </p>
                </div>
              ) : (
                <>
                  <div className="game-grid" role="list" aria-label="Search results">
                    {result.games.map((game) => (
                      <GameCard key={game.app_id} game={game} />
                    ))}
                  </div>

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className="pagination" aria-label="Pagination">
                      <button
                        id="pagination-prev"
                        className="pagination__btn"
                        onClick={() => doSearch(query, page - 1)}
                        disabled={page <= 1}
                        aria-label="Previous page"
                      >
                        ← Prev
                      </button>
                      <span className="pagination__info">
                        Page {page} of {totalPages}
                      </span>
                      <button
                        id="pagination-next"
                        className="pagination__btn"
                        onClick={() => doSearch(query, page + 1)}
                        disabled={page >= totalPages}
                        aria-label="Next page"
                      >
                        Next →
                      </button>
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {/* Initial state — no search yet */}
          {!loading && !result && !error && (
            <div className="empty-state">
              <div className="empty-state__icon">🎮</div>
              <h2 className="empty-state__title">Search a game to get started</h2>
              <p className="empty-state__body">
                Enter a game name above. First run{" "}
                <code style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-primary)" }}>
                  make seed
                </code>{" "}
                to ingest a set of well-known games.
              </p>
            </div>
          )}
        </div>
      </section>
    </>
  );
}
