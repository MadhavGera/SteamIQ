"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  api,
  type MarketOpportunityScore,
  type MarketTrendPoint,
} from "@/lib/api";
import { useUserMode } from "@/lib/UserModeContext";

export default function MarketExplorerPage() {
  const { isDeveloper, setMode } = useUserMode();

  const [opportunities, setOpportunities] = useState<MarketOpportunityScore[]>([]);
  const [trends, setTrends] = useState<MarketTrendPoint[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [filterType, setFilterType] = useState<"all" | "genre" | "tag">("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"opportunities" | "trends">("opportunities");

  useEffect(() => {
    let isMounted = true;
    async function fetchData() {
      setIsLoading(true);
      setError(null);
      try {
        const [oppsRes, trendsRes] = await Promise.all([
          api.getMarketOpportunities(undefined, 100),
          api.getMarketTrends(undefined, undefined, 50),
        ]);
        if (isMounted) {
          // HONEST-FALLBACK: Empty array if response returns null or undefined
          setOpportunities(oppsRes ?? []);
          // HONEST-FALLBACK: Empty array if response returns null or undefined
          setTrends(trendsRes ?? []);
          setIsLoading(false);
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : "Failed to load market intelligence";
          setError(msg);
          setIsLoading(false);
        }
      }
    }
    fetchData();
    return () => {
      isMounted = false;
    };
  }, []);

  const filteredOpportunities = useMemo(() => {
    return opportunities.filter((item) => {
      if (filterType !== "all" && item.entity_type !== filterType) {
        return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        return item.genre_or_tag.toLowerCase().includes(q);
      }
      return true;
    });
  }, [opportunities, filterType, searchQuery]);

  // Quadrant classification based on median demand and saturation scores
  const quadrants = useMemo(() => {
    const q1: MarketOpportunityScore[] = []; // High Demand, Low Saturation (Blue Ocean)
    const q2: MarketOpportunityScore[] = []; // High Demand, High Saturation (Crowded)
    const q3: MarketOpportunityScore[] = []; // Low Demand, Low Saturation (Niche)
    const q4: MarketOpportunityScore[] = []; // Low Demand, High Saturation (Danger Zone)

    for (const item of opportunities) {
      const highDemand = item.demand_score >= 50;
      const lowSaturation = item.saturation_score < 50;

      if (highDemand && lowSaturation) {
        q1.push(item);
      } else if (highDemand && !lowSaturation) {
        q2.push(item);
      } else if (!highDemand && lowSaturation) {
        q3.push(item);
      } else {
        q4.push(item);
      }
    }
    return { q1, q2, q3, q4 };
  }, [opportunities]);

  // Top opportunity metric
  const topNiche = useMemo(() => {
    if (opportunities.length === 0) return null;
    return [...opportunities].sort((a, b) => b.opportunity_score - a.opportunity_score)[0];
  }, [opportunities]);

  const avgSaturation = useMemo(() => {
    if (opportunities.length === 0) return 0;
    const sum = opportunities.reduce((acc, curr) => acc + curr.saturation_score, 0);
    return Math.round(sum / opportunities.length);
  }, [opportunities]);

  return (
    <div className="container" style={{ paddingBottom: "80px", maxWidth: "1280px", margin: "0 auto", paddingLeft: "24px", paddingRight: "24px" }}>
      {/* Page Header */}
      <div style={{ marginTop: "32px", marginBottom: "28px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "28px", color: "var(--accent-cyan)", fontVariationSettings: "'FILL' 1" }}
              >
                query_stats
              </span>
              <h1 style={{ fontSize: "28px", fontWeight: "800", letterSpacing: "-0.02em", color: "var(--text-high)" }}>
                Market Explorer
              </h1>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "700",
                  textTransform: "uppercase",
                  padding: "2px 8px",
                  borderRadius: "999px",
                  background: "rgba(34, 211, 238, 0.15)",
                  color: "var(--accent-cyan)",
                  border: "1px solid rgba(34, 211, 238, 0.3)",
                }}
              >
                Developer Intelligence
              </span>
            </div>
            <p style={{ color: "var(--text-medium)", fontSize: "14px", maxWidth: "720px", margin: 0 }}>
              Pre-materialized genre &amp; tag opportunity scoring, market saturation matrices, and feature gap analytics. Strictly zero live aggregation (ADR 0001, Decision 2).
            </p>
          </div>

          {/* Navigation Controls */}
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button
              onClick={() => setActiveTab("opportunities")}
              style={{
                padding: "8px 16px",
                borderRadius: "8px",
                fontSize: "13px",
                fontWeight: "600",
                cursor: "pointer",
                background: activeTab === "opportunities" ? "var(--bg-surface-elevated)" : "transparent",
                color: activeTab === "opportunities" ? "var(--text-high)" : "var(--text-medium)",
                border: activeTab === "opportunities" ? "1px solid var(--border-medium)" : "1px solid transparent",
              }}
            >
              Opportunity Index ({opportunities.length})
            </button>
            <button
              onClick={() => setActiveTab("trends")}
              style={{
                padding: "8px 16px",
                borderRadius: "8px",
                fontSize: "13px",
                fontWeight: "600",
                cursor: "pointer",
                background: activeTab === "trends" ? "var(--bg-surface-elevated)" : "transparent",
                color: activeTab === "trends" ? "var(--text-high)" : "var(--text-medium)",
                border: activeTab === "trends" ? "1px solid var(--border-medium)" : "1px solid transparent",
              }}
            >
              Market Trends ({trends.length})
            </button>
          </div>
        </div>
      </div>

      {/* Developer Mode Banner (if currently in Player Mode) */}
      {!isDeveloper && (
        <div
          style={{
            background: "rgba(245, 158, 11, 0.1)",
            border: "1px solid rgba(245, 158, 11, 0.3)",
            borderRadius: "12px",
            padding: "16px 20px",
            marginBottom: "28px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span className="material-symbols-outlined" style={{ color: "#f59e0b", fontSize: "24px" }}>
              lock
            </span>
            <div>
              <div style={{ fontWeight: "700", color: "#f59e0b", fontSize: "14px" }}>
                Player Mode Active
              </div>
              <div style={{ color: "var(--text-medium)", fontSize: "13px" }}>
                Market Explorer is designed for developers &amp; publishers evaluating commercial potential. Switch to Developer Mode to unlock full predictive breakdowns.
              </div>
            </div>
          </div>
          <button
            onClick={() => setMode("developer")}
            style={{
              background: "#f59e0b",
              color: "#000",
              fontWeight: "700",
              fontSize: "13px",
              padding: "8px 16px",
              borderRadius: "8px",
              border: "none",
              cursor: "pointer",
            }}
          >
            Switch to Developer Mode
          </button>
        </div>
      )}

      {/* KPI Overview Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "16px",
          marginBottom: "28px",
        }}
      >
        <div className="card" style={{ padding: "18px 20px", background: "var(--bg-surface-lowest)", border: "1px solid var(--border-subtle)", borderRadius: "12px" }}>
          <div style={{ fontSize: "12px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>
            Tracked Niches
          </div>
          <div style={{ fontSize: "24px", fontWeight: "800", color: "var(--text-high)" }}>
            {opportunities.length}
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-medium)", marginTop: "4px" }}>
            Pre-materialized genre &amp; tag clusters
          </div>
        </div>

        <div className="card" style={{ padding: "18px 20px", background: "var(--bg-surface-lowest)", border: "1px solid var(--border-subtle)", borderRadius: "12px" }}>
          <div style={{ fontSize: "12px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>
            Top Opportunity Niche
          </div>
          <div style={{ fontSize: "24px", fontWeight: "800", color: "var(--accent-cyan)" }}>
            {/* HONEST-FALLBACK: Safe label if top niche is uncalculated */}
            {topNiche?.genre_or_tag ?? "—"}
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-medium)", marginTop: "4px" }}>
            {/* HONEST-FALLBACK: Fallback display if score is unavailable */}
            Score: {topNiche ? `${topNiche.opportunity_score} / 100` : "—"}
          </div>
        </div>

        <div className="card" style={{ padding: "18px 20px", background: "var(--bg-surface-lowest)", border: "1px solid var(--border-subtle)", borderRadius: "12px" }}>
          <div style={{ fontSize: "12px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>
            Avg Market Saturation
          </div>
          <div style={{ fontSize: "24px", fontWeight: "800", color: avgSaturation > 60 ? "#ef4444" : "#10b981" }}>
            {avgSaturation}%
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-medium)", marginTop: "4px" }}>
            {avgSaturation > 60 ? "High competition baseline" : "Moderate competition baseline"}
          </div>
        </div>

        <div className="card" style={{ padding: "18px 20px", background: "var(--bg-surface-lowest)", border: "1px solid var(--border-subtle)", borderRadius: "12px" }}>
          <div style={{ fontSize: "12px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>
            Data Pipeline Source
          </div>
          <div style={{ fontSize: "16px", fontWeight: "700", color: "var(--text-high)", display: "flex", alignItems: "center", gap: "6px" }}>
            <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "#10b981" }}>
              check_circle
            </span>
            mart_opportunity_scores
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-medium)", marginTop: "4px" }}>
            Zero runtime aggregation
          </div>
        </div>
      </div>

      {isLoading && (
        <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-medium)" }}>
          <div className="material-symbols-outlined" style={{ fontSize: "36px", animation: "spin 1s linear infinite", marginBottom: "12px" }}>
            refresh
          </div>
          <div>Loading market intelligence marts...</div>
        </div>
      )}

      {error && (
        <div style={{ padding: "24px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "12px", color: "#ef4444", marginBottom: "28px" }}>
          <strong>Error loading market data:</strong> {error}
        </div>
      )}

      {!isLoading && !error && activeTab === "opportunities" && (
        <>
          {/* 4-Quadrant Strategic Matrix */}
          <div style={{ marginBottom: "36px" }}>
            <div style={{ marginBottom: "14px" }}>
              <h2 style={{ fontSize: "18px", fontWeight: "700", color: "var(--text-high)", margin: 0 }}>
                Demand vs. Saturation Matrix
              </h2>
              <p style={{ fontSize: "13px", color: "var(--text-medium)", margin: "4px 0 0 0" }}>
                Identifies high-opportunity white spaces (high demand, low saturation) versus crowded red oceans.
              </p>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "16px",
              }}
            >
              {/* Q1: Blue Ocean */}
              <div
                style={{
                  background: "rgba(16, 185, 129, 0.05)",
                  border: "1px solid rgba(16, 185, 129, 0.25)",
                  borderRadius: "12px",
                  padding: "18px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <div style={{ fontWeight: "700", color: "#10b981", fontSize: "14px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>trending_up</span>
                    High Opportunity (Blue Ocean)
                  </div>
                  <span style={{ fontSize: "12px", color: "#10b981", fontWeight: "700" }}>
                    High Demand · Low Saturation
                  </span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                  {quadrants.q1.length > 0 ? (
                    quadrants.q1.map((item) => (
                      <span
                        key={item.id}
                        style={{
                          fontSize: "12px",
                          padding: "4px 10px",
                          borderRadius: "6px",
                          background: "rgba(16, 185, 129, 0.15)",
                          color: "#10b981",
                          fontWeight: "600",
                          border: "1px solid rgba(16, 185, 129, 0.3)",
                        }}
                      >
                        {item.genre_or_tag} ({item.opportunity_score})
                      </span>
                    ))
                  ) : (
                    <span style={{ fontSize: "12px", color: "var(--text-low)" }}>No clusters currently in this quadrant</span>
                  )}
                </div>
              </div>

              {/* Q2: Crowded Giants */}
              <div
                style={{
                  background: "rgba(245, 158, 11, 0.05)",
                  border: "1px solid rgba(245, 158, 11, 0.25)",
                  borderRadius: "12px",
                  padding: "18px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <div style={{ fontWeight: "700", color: "#f59e0b", fontSize: "14px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>groups</span>
                    Crowded Giants (High Demand)
                  </div>
                  <span style={{ fontSize: "12px", color: "#f59e0b", fontWeight: "700" }}>
                    High Demand · High Saturation
                  </span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                  {quadrants.q2.length > 0 ? (
                    quadrants.q2.map((item) => (
                      <span
                        key={item.id}
                        style={{
                          fontSize: "12px",
                          padding: "4px 10px",
                          borderRadius: "6px",
                          background: "rgba(245, 158, 11, 0.15)",
                          color: "#f59e0b",
                          fontWeight: "600",
                          border: "1px solid rgba(245, 158, 11, 0.3)",
                        }}
                      >
                        {item.genre_or_tag} ({item.opportunity_score})
                      </span>
                    ))
                  ) : (
                    <span style={{ fontSize: "12px", color: "var(--text-low)" }}>No clusters currently in this quadrant</span>
                  )}
                </div>
              </div>

              {/* Q3: Niche Underdogs */}
              <div
                style={{
                  background: "rgba(59, 130, 246, 0.05)",
                  border: "1px solid rgba(59, 130, 246, 0.25)",
                  borderRadius: "12px",
                  padding: "18px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <div style={{ fontWeight: "700", color: "#3b82f6", fontSize: "14px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>filter_vintage</span>
                    Niche Underdogs
                  </div>
                  <span style={{ fontSize: "12px", color: "#3b82f6", fontWeight: "700" }}>
                    Moderate Demand · Low Saturation
                  </span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                  {quadrants.q3.length > 0 ? (
                    quadrants.q3.map((item) => (
                      <span
                        key={item.id}
                        style={{
                          fontSize: "12px",
                          padding: "4px 10px",
                          borderRadius: "6px",
                          background: "rgba(59, 130, 246, 0.15)",
                          color: "#3b82f6",
                          fontWeight: "600",
                          border: "1px solid rgba(59, 130, 246, 0.3)",
                        }}
                      >
                        {item.genre_or_tag} ({item.opportunity_score})
                      </span>
                    ))
                  ) : (
                    <span style={{ fontSize: "12px", color: "var(--text-low)" }}>No clusters currently in this quadrant</span>
                  )}
                </div>
              </div>

              {/* Q4: Saturated / Danger */}
              <div
                style={{
                  background: "rgba(239, 68, 68, 0.05)",
                  border: "1px solid rgba(239, 68, 68, 0.25)",
                  borderRadius: "12px",
                  padding: "18px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <div style={{ fontWeight: "700", color: "#ef4444", fontSize: "14px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>warning</span>
                    High Saturation / Competitive Risk
                  </div>
                  <span style={{ fontSize: "12px", color: "#ef4444", fontWeight: "700" }}>
                    Lower Demand · High Saturation
                  </span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                  {quadrants.q4.length > 0 ? (
                    quadrants.q4.map((item) => (
                      <span
                        key={item.id}
                        style={{
                          fontSize: "12px",
                          padding: "4px 10px",
                          borderRadius: "6px",
                          background: "rgba(239, 68, 68, 0.15)",
                          color: "#ef4444",
                          fontWeight: "600",
                          border: "1px solid rgba(239, 68, 68, 0.3)",
                        }}
                      >
                        {item.genre_or_tag} ({item.opportunity_score})
                      </span>
                    ))
                  ) : (
                    <span style={{ fontSize: "12px", color: "var(--text-low)" }}>No clusters currently in this quadrant</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Table Toolbar */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "12px",
              marginBottom: "16px",
            }}
          >
            {/* Search */}
            <div style={{ position: "relative", minWidth: "260px" }}>
              <input
                type="text"
                placeholder="Search niche (e.g. Action, RPG, Rogue)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px 12px 8px 36px",
                  borderRadius: "8px",
                  background: "var(--bg-surface-elevated)",
                  border: "1px solid var(--border-medium)",
                  color: "var(--text-high)",
                  fontSize: "13px",
                  outline: "none",
                }}
              />
              <span
                className="material-symbols-outlined"
                style={{
                  position: "absolute",
                  left: "10px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  fontSize: "18px",
                  color: "var(--text-low)",
                }}
              >
                search
              </span>
            </div>

            {/* Entity Type Filter */}
            <div style={{ display: "flex", gap: "6px" }}>
              {(["all", "genre", "tag"] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => setFilterType(type)}
                  style={{
                    padding: "6px 12px",
                    borderRadius: "6px",
                    fontSize: "12px",
                    fontWeight: "600",
                    cursor: "pointer",
                    textTransform: "capitalize",
                    background: filterType === type ? "var(--accent-cyan)" : "var(--bg-surface-elevated)",
                    color: filterType === type ? "#000" : "var(--text-medium)",
                    border: "1px solid var(--border-medium)",
                  }}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Opportunities Table */}
          <div
            className="card"
            style={{
              overflowX: "auto",
              background: "var(--bg-surface-lowest)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "12px",
            }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "13px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-subtle)", color: "var(--text-low)", textTransform: "uppercase", fontSize: "11px", letterSpacing: "0.05em" }}>
                  <th style={{ padding: "14px 16px" }}>Niche &amp; Type</th>
                  <th style={{ padding: "14px 16px" }}>Opportunity Score</th>
                  <th style={{ padding: "14px 16px" }}>Demand</th>
                  <th style={{ padding: "14px 16px" }}>Saturation</th>
                  <th style={{ padding: "14px 16px" }}>Sentiment Gap</th>
                  <th style={{ padding: "14px 16px" }}>Monetization</th>
                  <th style={{ padding: "14px 16px" }}>Catalog Games</th>
                  <th style={{ padding: "14px 16px" }}>Median Price</th>
                  <th style={{ padding: "14px 16px" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredOpportunities.length === 0 ? (
                  <tr>
                    <td colSpan={9} style={{ padding: "40px", textAlign: "center", color: "var(--text-medium)" }}>
                      No opportunity records matched your search filters.
                    </td>
                  </tr>
                ) : (
                  filteredOpportunities.map((row) => {
                    const isExpanded = expandedId === row.id;
                    const scoreColor =
                      row.opportunity_score >= 70
                        ? "#10b981"
                        : row.opportunity_score >= 45
                        ? "#f59e0b"
                        : "#94a3b8";

                    return (
                      <React.Fragment key={row.id}>
                        <tr
                          style={{
                            borderBottom: "1px solid var(--border-subtle)",
                            background: isExpanded ? "rgba(255, 255, 255, 0.02)" : "transparent",
                            transition: "background 0.15s ease",
                          }}
                        >
                          <td style={{ padding: "14px 16px" }}>
                            <div style={{ fontWeight: "700", color: "var(--text-high)", fontSize: "14px" }}>
                              {row.genre_or_tag}
                            </div>
                            <span
                              style={{
                                fontSize: "10px",
                                textTransform: "uppercase",
                                fontWeight: "700",
                                color: "var(--text-low)",
                              }}
                            >
                              {row.entity_type}
                            </span>
                          </td>

                          <td style={{ padding: "14px 16px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <span style={{ fontWeight: "800", color: scoreColor, fontSize: "15px" }}>
                                {row.opportunity_score}
                              </span>
                              <div style={{ width: "60px", height: "6px", background: "var(--bg-surface-elevated)", borderRadius: "3px", overflow: "hidden" }}>
                                <div
                                  style={{
                                    width: `${Math.min(100, row.opportunity_score)}%`,
                                    height: "100%",
                                    background: scoreColor,
                                  }}
                                />
                              </div>
                            </div>
                          </td>

                          <td style={{ padding: "14px 16px", color: "var(--text-medium)", fontWeight: "600" }}>
                            {row.demand_score}
                          </td>

                          <td style={{ padding: "14px 16px", color: "var(--text-medium)", fontWeight: "600" }}>
                            {row.saturation_score}
                          </td>

                          <td style={{ padding: "14px 16px", color: "var(--text-medium)", fontWeight: "600" }}>
                            {row.sentiment_gap_score}
                          </td>

                          <td style={{ padding: "14px 16px", color: "var(--text-medium)", fontWeight: "600" }}>
                            {row.monetization_score}
                          </td>

                          <td style={{ padding: "14px 16px", color: "var(--text-high)", fontWeight: "600" }}>
                            {row.game_count}
                          </td>

                          <td style={{ padding: "14px 16px", color: "var(--text-high)", fontWeight: "600" }}>
                            {/* HONEST-FALLBACK: Safe currency string if median price is uncalculated */}
                            {row.median_price_usd !== null ? `$${row.median_price_usd.toFixed(2)}` : "—"}
                          </td>

                          <td style={{ padding: "14px 16px" }}>
                            <button
                              onClick={() => setExpandedId(isExpanded ? null : row.id)}
                              style={{
                                background: "var(--bg-surface-elevated)",
                                border: "1px solid var(--border-medium)",
                                color: "var(--text-high)",
                                fontSize: "12px",
                                fontWeight: "600",
                                padding: "4px 10px",
                                borderRadius: "6px",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: "4px",
                              }}
                            >
                              <span>{isExpanded ? "Hide" : "Details"}</span>
                              <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>
                                {isExpanded ? "expand_less" : "expand_more"}
                              </span>
                            </button>
                          </td>
                        </tr>

                        {/* Expanded details row */}
                        {isExpanded && (
                          <tr style={{ background: "rgba(0, 0, 0, 0.2)", borderBottom: "1px solid var(--border-subtle)" }}>
                            <td colSpan={9} style={{ padding: "16px 20px" }}>
                              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "20px" }}>
                                {/* Recommended Features */}
                                <div>
                                  <div style={{ fontSize: "12px", textTransform: "uppercase", fontWeight: "700", color: "var(--accent-cyan)", marginBottom: "8px" }}>
                                    Recommended Feature Signals
                                  </div>
                                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                                    {Array.isArray(row.recommended_features) && row.recommended_features.length > 0 ? (
                                      row.recommended_features.map((feat, idx) => (
                                        <span
                                          key={idx}
                                          style={{
                                            fontSize: "12px",
                                            padding: "3px 8px",
                                            borderRadius: "6px",
                                            background: "rgba(34, 211, 238, 0.1)",
                                            color: "var(--accent-cyan)",
                                            border: "1px solid rgba(34, 211, 238, 0.25)",
                                          }}
                                        >
                                          {String(feat)}
                                        </span>
                                      ))
                                    ) : (
                                      <span style={{ fontSize: "12px", color: "var(--text-low)" }}>No specific feature signals recorded</span>
                                    )}
                                  </div>
                                </div>

                                {/* Top Loved Themes */}
                                <div>
                                  <div style={{ fontSize: "12px", textTransform: "uppercase", fontWeight: "700", color: "#10b981", marginBottom: "8px" }}>
                                    Top Praised Themes
                                  </div>
                                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                                    {Array.isArray(row.top_loved_themes) && row.top_loved_themes.length > 0 ? (
                                      row.top_loved_themes.map((theme, idx) => (
                                        <span
                                          key={idx}
                                          style={{
                                            fontSize: "12px",
                                            padding: "3px 8px",
                                            borderRadius: "6px",
                                            background: "rgba(16, 185, 129, 0.1)",
                                            color: "#10b981",
                                            border: "1px solid rgba(16, 185, 129, 0.25)",
                                          }}
                                        >
                                          {String(theme)}
                                        </span>
                                      ))
                                    ) : (
                                      <span style={{ fontSize: "12px", color: "var(--text-low)" }}>No praised themes recorded</span>
                                    )}
                                  </div>
                                </div>

                                {/* Top Complaints */}
                                <div>
                                  <div style={{ fontSize: "12px", textTransform: "uppercase", fontWeight: "700", color: "#ef4444", marginBottom: "8px" }}>
                                    Headroom / Top Complaints
                                  </div>
                                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                                    {Array.isArray(row.top_complaint_themes) && row.top_complaint_themes.length > 0 ? (
                                      row.top_complaint_themes.map((theme, idx) => (
                                        <span
                                          key={idx}
                                          style={{
                                            fontSize: "12px",
                                            padding: "3px 8px",
                                            borderRadius: "6px",
                                            background: "rgba(239, 68, 68, 0.1)",
                                            color: "#ef4444",
                                            border: "1px solid rgba(239, 68, 68, 0.25)",
                                          }}
                                        >
                                          {String(theme)}
                                        </span>
                                      ))
                                    ) : (
                                      <span style={{ fontSize: "12px", color: "var(--text-low)" }}>No major complaint spikes recorded</span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Market Trends View */}
      {!isLoading && !error && activeTab === "trends" && (
        <div className="card" style={{ padding: "20px", background: "var(--bg-surface-lowest)", border: "1px solid var(--border-subtle)", borderRadius: "12px" }}>
          <div style={{ marginBottom: "16px" }}>
            <h2 style={{ fontSize: "18px", fontWeight: "700", color: "var(--text-high)", margin: 0 }}>
              Historical Market &amp; Platform Trends
            </h2>
            <p style={{ fontSize: "13px", color: "var(--text-medium)", margin: "4px 0 0 0" }}>
              Pre-materialized 7-day and 30-day movement indicators from mart_trends.
            </p>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "13px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-subtle)", color: "var(--text-low)", textTransform: "uppercase", fontSize: "11px", letterSpacing: "0.05em" }}>
                  <th style={{ padding: "12px 14px" }}>Date</th>
                  <th style={{ padding: "12px 14px" }}>Trend Type</th>
                  <th style={{ padding: "12px 14px" }}>Category / Niche</th>
                  <th style={{ padding: "12px 14px" }}>Metric Name</th>
                  <th style={{ padding: "12px 14px" }}>Value</th>
                  <th style={{ padding: "12px 14px" }}>7d Delta</th>
                  <th style={{ padding: "12px 14px" }}>30d Delta</th>
                </tr>
              </thead>
              <tbody>
                {trends.length === 0 ? (
                  <tr>
                    <td colSpan={7} style={{ padding: "30px", textAlign: "center", color: "var(--text-medium)" }}>
                      No trends currently materialized in mart_trends.
                    </td>
                  </tr>
                ) : (
                  trends.map((t) => (
                    <tr key={t.id} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                      <td style={{ padding: "12px 14px", color: "var(--text-medium)" }}>
                        {new Date(t.recorded_date).toLocaleDateString()}
                      </td>
                      <td style={{ padding: "12px 14px" }}>
                        <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", padding: "2px 6px", borderRadius: "4px", background: "var(--bg-surface-elevated)", color: "var(--text-high)" }}>
                          {t.trend_type}
                        </span>
                      </td>
                      <td style={{ padding: "12px 14px", fontWeight: "600", color: "var(--text-high)" }}>
                        {/* HONEST-FALLBACK: Safe placeholder if category is null */}
                        {t.category ?? "General"}
                      </td>
                      <td style={{ padding: "12px 14px", color: "var(--text-medium)" }}>
                        {t.metric_name}
                      </td>
                      <td style={{ padding: "12px 14px", fontWeight: "700", color: "var(--text-high)" }}>
                        {t.metric_value}
                      </td>
                      <td style={{ padding: "12px 14px", color: t.change_pct_7d && t.change_pct_7d > 0 ? "#10b981" : t.change_pct_7d && t.change_pct_7d < 0 ? "#ef4444" : "var(--text-medium)" }}>
                        {/* HONEST-FALLBACK: Safe placeholder if 7d delta is null */}
                        {t.change_pct_7d !== null ? `${t.change_pct_7d > 0 ? "+" : ""}${t.change_pct_7d}%` : "—"}
                      </td>
                      <td style={{ padding: "12px 14px", color: t.change_pct_30d && t.change_pct_30d > 0 ? "#10b981" : t.change_pct_30d && t.change_pct_30d < 0 ? "#ef4444" : "var(--text-medium)" }}>
                        {/* HONEST-FALLBACK: Safe placeholder if 30d delta is null */}
                        {t.change_pct_30d !== null ? `${t.change_pct_30d > 0 ? "+" : ""}${t.change_pct_30d}%` : "—"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
