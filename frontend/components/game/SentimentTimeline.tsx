"use client";

import React from "react";
import { MonthlySentimentData } from "@/lib/api";

export interface SentimentTimelineProps {
  data: MonthlySentimentData[];
  loading?: boolean;
}

export function SentimentTimeline({ data, loading = false }: SentimentTimelineProps) {
  if (loading) {
    return (
      <div className="card" style={{ padding: "24px" }}>
        <div className="skeleton" style={{ height: "20px", width: "180px", marginBottom: "16px" }} />
        <div className="skeleton" style={{ height: "160px", width: "100%", borderRadius: "8px" }} />
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="card" style={{ padding: "24px", textAlign: "center", color: "var(--text-muted)" }}>
        <p style={{ margin: "32px 0" }}>No monthly sentiment timeline available yet.</p>
      </div>
    );
  }

  const maxReviews = Math.max(...data.map((d) => d.positive_reviews + d.negative_reviews), 1);

  return (
    <div className="card" style={{ padding: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
        <span style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
          Historical Review Trajectory
        </span>
        <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
          Last {data.length} months
        </span>
      </div>

      {/* SVG / Bar Chart Representation */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: "12px", height: "130px", padding: "10px 0", borderBottom: "1px solid var(--border-subtle)" }}>
        {data.map((m, i) => {
          const total = m.positive_reviews + m.negative_reviews;
          const heightPct = Math.max(12, Math.round((total / maxReviews) * 100));
          const posPct = m.net_positive_pct;

          return (
            <div
              key={m.month || i}
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                height: "100%",
                justifyContent: "flex-end",
                position: "relative",
              }}
            >
              {/* Stacked bar */}
              <div
                style={{
                  width: "100%",
                  maxWidth: "36px",
                  height: `${heightPct}%`,
                  borderRadius: "4px 4px 0 0",
                  overflow: "hidden",
                  display: "flex",
                  flexDirection: "column-reverse",
                  background: "var(--bg-base)",
                  border: "1px solid var(--border-subtle)",
                }}
                title={`${m.month}: ${m.positive_reviews} pos / ${m.negative_reviews} neg (${posPct.toFixed(1)}%)`}
              >
                <div style={{ height: `${posPct}%`, background: "var(--success)" }} />
                <div style={{ height: `${100 - posPct}%`, background: "var(--danger)" }} />
              </div>
              <span style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "6px", fontFamily: "var(--font-mono)" }}>
                {m.month.slice(2)}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: "10px", fontSize: "11px", color: "var(--text-muted)" }}>
        <span>Volume & sentiment ratio per month</span>
        <span style={{ color: "var(--success)" }}>■ Positive</span>
        <span style={{ color: "var(--danger)" }}>■ Negative</span>
      </div>
    </div>
  );
}
