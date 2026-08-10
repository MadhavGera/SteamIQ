"use client";

import React from "react";

export interface ReviewSummaryProps {
  strengths: string[];
  painPoints: string[];
  featureRequests: string[];
  loading?: boolean;
}

export function ReviewSummary({
  strengths,
  painPoints,
  featureRequests,
  loading = false,
}: ReviewSummaryProps) {
  if (loading) {
    return (
      <div className="card" style={{ padding: "24px" }}>
        <div className="skeleton" style={{ height: "20px", width: "220px", marginBottom: "16px" }} />
        <div className="grid grid--3" style={{ gap: "16px" }}>
          {[1, 2, 3].map((n) => (
            <div key={n} className="skeleton" style={{ height: "120px", width: "100%", borderRadius: "8px" }} />
          ))}
        </div>
      </div>
    );
  }

  const hasData = (strengths && strengths.length > 0) || (painPoints && painPoints.length > 0) || (featureRequests && featureRequests.length > 0);

  if (!hasData) {
    return (
      <div className="card" style={{ padding: "24px", textAlign: "center", color: "var(--text-muted)" }}>
        <p style={{ margin: "24px 0" }}>No hierarchical review summary generated yet.</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: "24px" }}>
      <div style={{ marginBottom: "16px" }}>
        <span style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
          Hierarchical Player Feedback Synthesis
        </span>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
          AI Executive Summary & Player Consensus
        </h3>
      </div>

      <div className="grid grid--3" style={{ gap: "20px" }}>
        {/* Core Strengths */}
        <div
          style={{
            background: "var(--bg-surface-raised)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "8px",
            padding: "16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px", color: "var(--success)", fontWeight: 700, fontSize: "13px" }}>
            <span>✓</span>
            <span>Core Strengths</span>
          </div>
          <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "12px", lineHeight: "1.6", color: "var(--text-secondary)" }}>
            {strengths.map((s, i) => (
              <li key={i} style={{ marginBottom: "6px" }}>{s}</li>
            ))}
          </ul>
        </div>

        {/* Primary Complaints */}
        <div
          style={{
            background: "var(--bg-surface-raised)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "8px",
            padding: "16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px", color: "var(--danger)", fontWeight: 700, fontSize: "13px" }}>
            <span>⚠</span>
            <span>Primary Friction Points</span>
          </div>
          <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "12px", lineHeight: "1.6", color: "var(--text-secondary)" }}>
            {painPoints.map((p, i) => (
              <li key={i} style={{ marginBottom: "6px" }}>{p}</li>
            ))}
          </ul>
        </div>

        {/* Feature Requests / Player Wishes */}
        <div
          style={{
            background: "var(--bg-surface-raised)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "8px",
            padding: "16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px", color: "var(--accent-primary)", fontWeight: 700, fontSize: "13px" }}>
            <span>★</span>
            <span>Player Requests & Wishes</span>
          </div>
          <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "12px", lineHeight: "1.6", color: "var(--text-secondary)" }}>
            {featureRequests.map((r, i) => (
              <li key={i} style={{ marginBottom: "6px" }}>{r}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
