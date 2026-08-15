import React from "react";

interface ExecutiveBriefCardProps {
  title: string;
  brief: {
    market_position?: string;
    growth_vector?: string;
  } | null;
  netSentimentPct?: number | null;
}

export function ExecutiveBriefCard({ title, brief, netSentimentPct }: ExecutiveBriefCardProps) {
  const marketPos =
    brief?.market_position ||
    `${title} maintains a strong community footprint with ${netSentimentPct ? `${netSentimentPct.toFixed(1)}%` : "positive"} net positive sentiment across verified user cohorts.`;

  const growthVec =
    brief?.growth_vector ||
    "Focus on post-launch stability, regional pricing alignment, and expanding acclaim around core gameplay pillars.";

  return (
    <div className="card card--raised">
      <div style={{ marginBottom: "14px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
          Executive Intelligence Brief
        </h3>
        <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
          Synthesized review &amp; performance metrics
        </p>
      </div>

      <p style={{ fontSize: "13px", lineHeight: "1.7", color: "var(--text-secondary)", marginBottom: "14px" }}>
        <strong style={{ color: "var(--text-primary)" }}>Market Position:</strong> {marketPos}
      </p>
      <p style={{ fontSize: "13px", lineHeight: "1.7", color: "var(--text-secondary)" }}>
        <strong style={{ color: "var(--text-primary)" }}>Growth Vector:</strong> {growthVec}
      </p>
    </div>
  );
}
