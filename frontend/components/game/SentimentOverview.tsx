"use client";

import React from "react";

export interface SentimentOverviewProps {
  positivePct: number;
  mixedPct: number;
  negativePct: number;
  positiveCount: number;
  negativeCount: number;
  totalCount?: number;
  sentimentLabel: string;
  loading?: boolean;
}

export function SentimentOverview({
  positivePct,
  mixedPct,
  negativePct,
  positiveCount,
  negativeCount,
  totalCount,
  sentimentLabel,
  loading = false,
}: SentimentOverviewProps) {
  if (loading) {
    return (
      <div className="card" style={{ padding: "24px" }}>
        <div className="skeleton" style={{ height: "20px", width: "160px", marginBottom: "16px" }} />
        <div className="skeleton" style={{ height: "40px", width: "240px", marginBottom: "20px" }} />
        <div className="skeleton" style={{ height: "14px", width: "100%", borderRadius: "8px" }} />
      </div>
    );
  }

  // Determine badge styling based on sentiment score
  let badgeClass = "badge-pill--success";
  if (positivePct < 60) {
    badgeClass = "badge-pill--danger";
  } else if (positivePct < 75) {
    badgeClass = "badge-pill--warning";
  }

  const effectiveTotal = totalCount || positiveCount + negativeCount;

  return (
    <div className="card" style={{ padding: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <span style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
          Sentiment Health
        </span>
        <span className={`badge-pill ${badgeClass}`}>
          {sentimentLabel}
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginBottom: "8px" }}>
        <span style={{ fontSize: "36px", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--accent-primary)" }}>
          {positivePct.toFixed(1)}%
        </span>
        <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
          positive ({positiveCount.toLocaleString()} / {effectiveTotal.toLocaleString()} reviews)
        </span>
      </div>

      {/* Multi-segment Sentiment Bar */}
      <div
        style={{
          height: "12px",
          width: "100%",
          background: "var(--bg-base)",
          borderRadius: "6px",
          overflow: "hidden",
          display: "flex",
          margin: "16px 0 12px",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <div
          style={{
            width: `${positivePct}%`,
            background: "var(--success)",
            transition: "width 0.4s ease",
          }}
          title={`Positive: ${positivePct}%`}
        />
        <div
          style={{
            width: `${mixedPct}%`,
            background: "var(--warning)",
            transition: "width 0.4s ease",
          }}
          title={`Mixed: ${mixedPct}%`}
        />
        <div
          style={{
            width: `${negativePct}%`,
            background: "var(--danger)",
            transition: "width 0.4s ease",
          }}
          title={`Negative: ${negativePct}%`}
        />
      </div>

      {/* Legend & Breakdown stats */}
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--text-secondary)", marginTop: "10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--success)" }} />
          <span>Positive: {positivePct.toFixed(1)}%</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--warning)" }} />
          <span>Mixed: {mixedPct.toFixed(1)}%</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--danger)" }} />
          <span>Negative: {negativePct.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
