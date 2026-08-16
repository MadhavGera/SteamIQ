"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  api,
  type GameSummary,
  type GameSearchResult,
} from "@/lib/api";

const FEATURED_GAMES = [
  {
    app_id: 367520,
    name: "Hollow Knight",
    genre: "Metroidvania · Action-Adventure",
    gradient: "linear-gradient(135deg, #0f4a73, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/367520/header.jpg",
  },
  {
    app_id: 1145360,
    name: "Hades",
    genre: "Rogue-like · Action RPG",
    gradient: "linear-gradient(135deg, #243745, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/1145360/header.jpg",
  },
  {
    app_id: 504230,
    name: "Celeste",
    genre: "Precision Platformer · Indie",
    gradient: "linear-gradient(135deg, #93000a, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/504230/header.jpg",
  },
  {
    app_id: 413150,
    name: "Stardew Valley",
    genre: "Farming Sim · Cozy RPG",
    gradient: "linear-gradient(135deg, #4c1d95, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/413150/header.jpg",
  },
  {
    app_id: 588650,
    name: "Dead Cells",
    genre: "Rogue-lite · Metroidvania",
    gradient: "linear-gradient(135deg, #0f4a73, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/588650/header.jpg",
  },
  {
    app_id: 105600,
    name: "Terraria",
    genre: "Sandbox · Crafting · Survival",
    gradient: "linear-gradient(135deg, #243745, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/105600/header.jpg",
  },
  {
    app_id: 646570,
    name: "Slay the Spire",
    genre: "Roguelike Deckbuilder · Strategy",
    gradient: "linear-gradient(135deg, #93000a, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/646570/header.jpg",
  },
  {
    app_id: 1091500,
    name: "Cyberpunk 2077",
    genre: "Action RPG · Open World",
    gradient: "linear-gradient(135deg, #4c1d95, #0e212f)",
    image: "https://cdn.cloudflare.steamstatic.com/steam/apps/1091500/header.jpg",
  },
];

const POPULAR_QUERIES = [
  { name: "Hollow Knight", id: "367520" },
  { name: "Hades", id: "1145360" },
  { name: "Celeste", id: "504230" },
  { name: "Stardew Valley", id: "413150" },
  { name: "Dead Cells", id: "588650" },
];

export default function LandingPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<GameSummary[] | null>(null);
  const [totalMatches, setTotalMatches] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [ingestedGames, setIngestedGames] = useState<Record<number, GameSummary>>({});
  
  const [debouncedQuery, setDebouncedQuery] = useState("");

  // Fetch real ingested game records to replace hardcoded strings with API data
  useEffect(() => {
    let isMounted = true;
    api.searchGames("", 1, 50).then((res) => {
      if (!isMounted || !res?.games) return;
      const map: Record<number, GameSummary> = {};
      for (const g of res.games) {
        map[g.app_id] = g;
      }
      setIngestedGames(map);
    }).catch(() => {
      // Ignore initial load error if backend is cold
    });
    return () => { isMounted = false; };
  }, []);

  const doSearch = useCallback(async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) {
      setResults(null);
      setTotalMatches(null);
      setHasSearched(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      // If query is numeric App ID, direct lookup or search
      const res: GameSearchResult = await api.searchGames(trimmed, 1, 12);
      setResults(res.games);
      setTotalMatches(res.total);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Search failed";
      setError(msg);
      setResults([]);
      setTotalMatches(0);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounce the query for live search
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedQuery(query);
    }, 500);
    return () => clearTimeout(handler);
  }, [query]);

  // Trigger search when debounced query changes
  useEffect(() => {
    if (debouncedQuery) {
      doSearch(debouncedQuery);
    }
  }, [debouncedQuery, doSearch]);

  return (
    <div className="container" style={{ paddingTop: "24px", paddingBottom: "64px" }}>
      {/* ── Hero Section (Stitch Design) ── */}
      <section className="hero-wrapper">
        <h1 className="hero-title">
          Game Intelligence, Powered by AI
        </h1>

        <p className="hero-subtitle">
          Turn Steam reviews, player data, and market signals into explainable decisions.
          <br />
          Discover what gamers actually want before you write a single line of code.
        </p>

        {/* Search Bar */}
        <form
          className="stitch-search-container"
          onSubmit={(e) => {
            e.preventDefault();
            doSearch(query);
          }}
        >
          <span
            className="material-symbols-outlined"
            style={{ color: "var(--text-muted)", fontSize: "20px" }}
          >
            search
          </span>
          <input
            type="text"
            className="stitch-search-input"
            placeholder="Search by game name or Steam App ID..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          <button type="submit" className="stitch-search-btn">
            {loading ? "Analyzing..." : "Analyze Game"}
            <span className="material-symbols-outlined" style={{ fontSize: "16px", fontWeight: 700 }}>
              arrow_forward
            </span>
          </button>
        </form>

        {/* Popular Queries */}
        <div className="popular-queries">
          <span className="popular-queries__label">Popular Queries</span>
          <div className="popular-queries__list">
            {POPULAR_QUERIES.map((item) => (
              <button
                key={item.id}
                type="button"
                className="popular-chip"
                onClick={() => {
                  setQuery(item.name);
                  doSearch(item.name);
                }}
              >
                <span>{item.name}</span>
                <span className="popular-chip__id">{item.id}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Search Results View ── */}
      {hasSearched && (
        <section className="featured-section">
          <div className="featured-section__header">
            <span className="featured-section__eyebrow">Search Results</span>
            <h2 className="featured-section__title">
              {loading
                ? "Analyzing Steam Database..."
                : results && results.length > 0
                ? `${totalMatches} ${totalMatches === 1 ? "Game" : "Games"} Found`
                : "No Games Found"}
            </h2>
            <p className="featured-section__subtitle">
              {results && results.length > 0
                ? "Select a game to view deep sentiment, telemetry, and success prediction."
                : `No ingested games matched "${query}".`}
            </p>
          </div>

          {/* Error Banner */}
          {error && (
            <div
              className="badge-pill badge-pill--danger"
              style={{ padding: "8px 16px", marginBottom: "16px", display: "inline-flex" }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                warning
              </span>
              <span>{error}</span>
            </div>
          )}

          {/* Loading Skeleton */}
          {loading && (
            <div className="featured-grid">
              {[1, 2, 3, 4].map((n) => (
                <div key={n} className="card" style={{ padding: "16px" }}>
                  <div className="skeleton" style={{ height: "130px", marginBottom: "12px" }} />
                  <div className="skeleton" style={{ height: "20px", width: "70%", marginBottom: "8px" }} />
                  <div className="skeleton" style={{ height: "14px", width: "40%" }} />
                </div>
              ))}
            </div>
          )}

          {/* Empty State */}
          {!loading && results && results.length === 0 && (
            <div className="card" style={{ padding: "48px 24px", textAlign: "center", maxWidth: "600px", margin: "0 auto" }}>
              <div
                style={{
                  width: "56px",
                  height: "56px",
                  borderRadius: "50%",
                  backgroundColor: "var(--bg-surface-raised)",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto 16px",
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: "28px", color: "var(--text-muted)" }}>
                  search_off
                </span>
              </div>
              <h3 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "8px" }}>
                No Games Found
              </h3>
              <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6", marginBottom: "20px" }}>
                We couldn&apos;t find any games matching &ldquo;{query}&rdquo; on Steam.
              </p>
              <button
                type="button"
                onClick={() => {
                  setQuery("");
                  setHasSearched(false);
                }}
                className="btn btn--secondary"
              >
                Clear Search
              </button>
            </div>
          )}

          {/* Results Grid */}
          {!loading && results && results.length > 0 && (
            <div className="featured-grid">
              {results.map((g) => (
                <Link
                  key={g.app_id}
                  href={`/games/${g.app_id}?tab=overview`}
                  className="featured-card"
                >
                  <div className="featured-card__cover">
                    {g.header_image ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={g.header_image} alt={g.name} />
                    ) : null}
                    <div className="featured-card__badge" style={{ display: "flex", gap: "6px" }}>
                      {g.is_ingested ? (
                        <span className="badge-pill badge-pill--success" style={{ fontSize: "10px", padding: "1px 6px" }}>Indexed</span>
                      ) : (
                        <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px", padding: "1px 6px" }}>Unindexed</span>
                      )}
                      <div>
                        <span>ID:</span>
                        <strong>{g.app_id}</strong>
                      </div>
                    </div>
                  </div>
                  <div className="featured-card__body">
                    <h3 className="featured-card__title">{g.name}</h3>
                    <p className="featured-card__meta">
                      {[g.developer, g.genres?.[0]?.description].filter(Boolean).join(" · ") || "Steam Game"}
                    </p>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ── Featured & Ingested Games Grid (Honest Showcase with Live API Data) ── */}
      {!hasSearched && (
        <section className="featured-section">
          <div className="featured-section__header">
            <span className="featured-section__eyebrow">Curated Showcase</span>
            <h2 className="featured-section__title">Ready to Analyze</h2>
            <p className="featured-section__subtitle">
              Explore decision intelligence for indexed catalog titles, or search any Steam App ID to ingest live store telemetry.
            </p>
          </div>

          <div className="featured-grid">
            {FEATURED_GAMES.map((g) => {
              const live = ingestedGames[g.app_id];
              const liveGenre = live?.genres?.[0]?.description;
              const liveDev = live?.developer;

              return (
                <Link
                  key={g.app_id}
                  href={`/games/${g.app_id}?tab=overview`}
                  className="featured-card"
                >
                  <div
                    className="featured-card__cover"
                    style={{ background: g.gradient }}
                  >
                    {live?.header_image || g.image ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={live?.header_image || g.image} alt={live?.name || g.name} />
                    ) : null}
                    <div className="featured-card__badge">
                      <span>ID:</span>
                      <strong>{g.app_id}</strong>
                    </div>
                  </div>

                  <div className="featured-card__body">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
                      <h3 className="featured-card__title">{live?.name || g.name}</h3>
                      {live && (
                        <span className="badge-pill badge-pill--success" style={{ fontSize: "10px", padding: "1px 6px" }}>
                          Indexed
                        </span>
                      )}
                    </div>
                    <p className="featured-card__meta">
                      {liveGenre
                        ? (liveDev ? `${liveDev} · ${liveGenre}` : liveGenre)
                        : "Catalog Showcase"}
                    </p>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
