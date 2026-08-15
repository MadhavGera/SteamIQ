import React from "react";

interface ShapAttributionCardProps {
  shapValues: Record<string, number> | null;
  modelVersion?: string | null;
}

const FEATURE_LABELS: Record<string, string> = {
  complaint_density: "Review Complaint Concentration",
  loved_feature_density: "Player Praise & Loved Features",
  developer_avg_review_score: "Developer Track Record",
  publisher_avg_review_score: "Publisher Track Record",
  competitor_density: "Market Competitor Density",
  price_vs_genre_median: "Price vs Genre Benchmark",
  discount_pct: "Promotional Discount Depth",
  price_usd: "Base List Price",
  is_free: "Free-to-Play Monetization",
  developer_game_count: "Developer Catalog Depth",
  publisher_game_count: "Publisher Catalog Depth",
};

export function ShapAttributionCard({ shapValues, modelVersion }: ShapAttributionCardProps) {
  if (!shapValues || Object.keys(shapValues).length === 0) {
    return (
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px" }}>
          <div>
            <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
              ML Feature Attribution (SHAP)
            </h3>
            <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
              Primary drivers and drag factors influencing success score
            </p>
          </div>
          <span className="badge-pill badge-pill--neutral">Explainable AI</span>
        </div>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", padding: "24px 0", textAlign: "center" }}>
          Run the machine learning training pipeline to generate TreeSHAP explainability attributions.
        </p>
      </div>
    );
  }

  // Sort by absolute SHAP impact magnitude descending
  const sortedEntries = Object.entries(shapValues)
    .filter(([, val]) => typeof val === "number" && !isNaN(val))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 6);

  const maxAbsVal = Math.max(0.01, ...sortedEntries.map(([, v]) => Math.abs(v)));

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px" }}>
        <div>
          <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
            ML Feature Attribution (SHAP)
          </h3>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
            Primary drivers and drag factors influencing success score
          </p>
        </div>
        <span className="badge-pill badge-pill--neutral">
          {modelVersion ? `Model ${modelVersion}` : "Explainable AI"}
        </span>
      </div>

      <div className="shap-bar-list" style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
        {sortedEntries.map(([featKey, val]) => {
          const isPos = val >= 0;
          const pctWidth = Math.min(100, Math.max(8, Math.round((Math.abs(val) / maxAbsVal) * 100)));
          const label = FEATURE_LABELS[featKey] || featKey.replace(/_/g, " ");

          return (
            <div key={featKey} className="shap-item" style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span
                style={{
                  width: "210px",
                  fontSize: "13px",
                  fontWeight: 500,
                  color: "var(--text-primary)",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
                title={label}
              >
                {label}
              </span>
              <div className="shap-bar-track" style={{ flex: 1, height: "8px", backgroundColor: "var(--bg-card)", borderRadius: "4px", overflow: "hidden" }}>
                <div
                  className={`shap-bar-fill ${isPos ? "shap-bar-fill--pos" : "shap-bar-fill--neg"}`}
                  style={{
                    width: `${pctWidth}%`,
                    height: "100%",
                    backgroundColor: isPos ? "var(--success)" : "var(--danger)",
                    borderRadius: "4px",
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
              <span
                style={{
                  width: "55px",
                  textAlign: "right",
                  color: isPos ? "var(--success)" : "var(--danger)",
                  fontWeight: 700,
                  fontSize: "13px",
                  fontFamily: "var(--font-mono, monospace)",
                }}
              >
                {isPos ? `+${val.toFixed(2)}` : val.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
