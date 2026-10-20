"use client";

import React from "react";
import { MarketOverviewMetrics } from "@/lib/api";

interface AnalysisOverviewGridProps {
  marketSummary?: MarketOverviewMetrics | null;
  intent: string;
  confidence: number;
  tau: number;
}

export function AnalysisOverviewGrid({
  marketSummary,
  intent,
  confidence,
  tau,
}: AnalysisOverviewGridProps) {
  // HONEST-FALLBACK: Default to zero when games analyzed is undefined
  const gamesCount = marketSummary?.games_analyzed ?? 0;
  // HONEST-FALLBACK: Default price range to unrecorded placeholder
  const priceRange = marketSummary?.price_range_usd ?? "--";
  // HONEST-FALLBACK: Default median price to unrecorded placeholder
  const medianPrice = marketSummary?.median_price_usd ?? "--";
  // HONEST-FALLBACK: Default sentiment to null if unrecorded
  const sentimentPct = marketSummary?.sentiment_pct ?? null;
  // HONEST-FALLBACK: Default review count to zero when unrecorded
  const reviewCount = marketSummary?.review_count ?? 0;

  const isMarketIntent = intent === "market_opportunity";
  const isSimilarIntent = intent === "similar_games";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
        gap: "14px",
        marginBottom: "20px",
      }}
    >
      {/* 1. DATA COVERAGE */}
      <div
        className="card"
        style={{
          background: "var(--bg-surface-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "10px",
          padding: "16px 18px",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
          <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.06em", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Data Coverage
          </span>
          <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "var(--accent-primary)" }}>
            dataset
          </span>
        </div>
        <div style={{ fontSize: "22px", fontWeight: 700, color: "var(--text-primary)" }}>
          {gamesCount > 0 ? `${gamesCount} Peer Titles` : "Catalog Scope"}
        </div>
        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
          {reviewCount > 0
            ? `${reviewCount.toLocaleString()} verified reviews`
            : "Steam Store & Community Data"}
        </div>
      </div>

      {/* 2. PRICE POSITIONING */}
      <div
        className="card"
        style={{
          background: "var(--bg-surface-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "10px",
          padding: "16px 18px",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
          <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.06em", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Price Positioning
          </span>
          <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "#34d399" }}>
            payments
          </span>
        </div>
        <div style={{ fontSize: "22px", fontWeight: 700, color: "var(--text-primary)" }}>
          {medianPrice !== "--" ? `${medianPrice} Median` : "--"}
        </div>
        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
          Range: {priceRange}
        </div>
      </div>

      {/* 3. PLAYER SENTIMENT */}
      <div
        className="card"
        style={{
          background: "var(--bg-surface-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "10px",
          padding: "16px 18px",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
          <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.06em", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Player Sentiment
          </span>
          <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "#818cf8" }}>
            sentiment_very_satisfied
          </span>
        </div>
        <div style={{ fontSize: "22px", fontWeight: 700, color: "var(--text-primary)" }}>
          {sentimentPct !== null ? `${sentimentPct.toFixed(1)}% Positive` : "--"}
        </div>
        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
          {reviewCount > 0 ? `Consensus over ${reviewCount.toLocaleString()} reviews` : "Review Intelligence Aggregated"}
        </div>
      </div>

      {/* 4. MARKET EVIDENCE */}
      <div
        className="card"
        style={{
          background: "var(--bg-surface-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "10px",
          padding: "16px 18px",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
          <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.06em", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Market Evidence
          </span>
          <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "#fbbf24" }}>
            trending_up
          </span>
        </div>
        <div style={{ fontSize: "22px", fontWeight: 700, color: "var(--text-primary)" }}>
          {isMarketIntent
            ? "Opportunity Mart"
            : isSimilarIntent
            ? "Two-Tower CF"
            : "Analytical Mart"}
        </div>
        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
          {isMarketIntent
            ? `Demand & Saturation · ${(confidence * 100).toFixed(0)}% Conf (τ = ${(tau * 100).toFixed(0)}%)`
            : isSimilarIntent
            ? `Neural R^64 ANN · ${(confidence * 100).toFixed(0)}% Conf (τ = ${(tau * 100).toFixed(0)}%)`
            : `Grounded Marts · ${(confidence * 100).toFixed(0)}% Conf (τ = ${(tau * 100).toFixed(0)}%)`}
        </div>
      </div>
    </div>
  );
}
