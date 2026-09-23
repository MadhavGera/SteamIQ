import React from "react";
import type { RecommendationBundle } from "@/lib/api";

interface RecommendationsTabProps {
  bundle: RecommendationBundle | null;
  gameTitle: string;
}

const DOMAIN_ICONS: Record<string, string> = {
  "Pricing & Monetization": "payments",
  "Engineering & Quality": "build",
  "Content & Gameplay": "sports_esports",
  "Marketing & Positioning": "campaign",
  "Community & Live Ops": "group",
};

export function RecommendationsTab({ bundle, gameTitle }: RecommendationsTabProps) {
  if (!bundle || bundle.total_recommendations === 0 || !bundle.recommendations.length) {
    return (
      <div className="card" style={{ padding: "64px 24px", textAlign: "center", marginTop: "24px" }}>
        <div
          style={{
            width: "56px",
            height: "56px",
            borderRadius: "50%",
            backgroundColor: "var(--bg-surface)",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "16px",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "28px", color: "var(--text-secondary)" }}>
            lightbulb
          </span>
        </div>
        <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "8px" }}>
          Recommendation Pipeline Pending
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "520px", margin: "0 auto", lineHeight: "1.6" }}>
          SteamIQ synthesizes machine learning TreeSHAP feature attributions and NLP complaint taxonomies into prioritized developer action items. Run `make generate-recommendations` to materialize recommendations.
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "24px" }}>
      {/* ── 1. Summary Strip ── */}
      <div className="kpi-matrix-grid">
        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">ACTION ITEMS</span>
            <span className="badge-pill badge-pill--neutral">Prioritized</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--accent-primary)" }}>
              {bundle.total_recommendations}
            </div>
            <div className="kpi-card__sub">Strategic action vectors ranked by ROI</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">PRIMARY DOMAIN</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ fontSize: "20px", color: "var(--text-primary)" }}>
              {bundle.recommendations[0]?.domain || "Engineering"}
            </div>
            <div className="kpi-card__sub">Highest impact immediate vector</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">ATTRIBUTION MODEL</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ fontSize: "20px", color: "var(--accent-light)" }}>
              {bundle.model_run_id ? "Optuna ML + SHAP" : "Heuristic + NLP"}
            </div>
            <div className="kpi-card__sub">
              {bundle.model_run_id ? `Run ID: ${bundle.model_run_id.slice(0, 8)}...` : "Domain rule synthesis"}
            </div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">DECISION ENGINE</span>
            <span className="badge-pill badge-pill--success">Active</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ fontSize: "20px", color: "var(--success)" }}>
              Hybrid AI
            </div>
            <div className="kpi-card__sub">Non-hallucinatory grounded evidence</div>
          </div>
        </div>
      </div>

      {/* ── 2. Strategic Brief ── */}
      {bundle.priority_action_summary && (
        <div className="card card--raised">
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "8px" }}>
            Priority Action Summary for {gameTitle}
          </h3>
          <p style={{ fontSize: "13px", lineHeight: "1.7", color: "var(--text-secondary)", margin: 0 }}>
            {bundle.priority_action_summary}
          </p>
        </div>
      )}

      {/* ── 3. Prioritized Recommendation Cards ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
        {bundle.recommendations.map((rec) => {
          const domainIcon = DOMAIN_ICONS[rec.domain] || "insights";
          const confPct = Math.round(rec.confidence_score * 100);

          return (
            <div key={rec.id} className="card" style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              {/* Card Header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <div
                    style={{
                      width: "36px",
                      height: "36px",
                      borderRadius: "8px",
                      backgroundColor: "rgba(56, 189, 248, 0.12)",
                      border: "1px solid rgba(56, 189, 248, 0.3)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontWeight: 800,
                      fontSize: "15px",
                      color: "var(--accent-primary)",
                    }}
                  >
                    #{rec.priority_rank}
                  </div>
                  <div>
                    <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
                      {rec.title}
                    </h3>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
                      <span className="material-symbols-outlined" style={{ fontSize: "15px" }}>{domainIcon}</span>
                      <span>{rec.domain}</span>
                    </div>
                  </div>
                </div>

                {/* Badges */}
                <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                  <span
                    className={`badge-pill ${
                      rec.impact_level === "High"
                        ? "badge-pill--success"
                        : rec.impact_level === "Medium"
                        ? "badge-pill--neutral"
                        : "badge-pill--neutral"
                    }`}
                  >
                    {rec.impact_level} Impact
                  </span>
                  <span className="badge-pill badge-pill--neutral">
                    {rec.difficulty_level} Effort
                  </span>

                  {/* Evidence Type Attribution Badge */}
                  {rec.evidence_type === "hybrid" || rec.model_run_id ? (
                    <span
                      className="badge-pill"
                      style={{
                        backgroundColor: "rgba(56, 189, 248, 0.15)",
                        color: "var(--accent-light)",
                        border: "1px solid rgba(56, 189, 248, 0.3)",
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "4px",
                        fontWeight: 600,
                      }}
                      title="Directly derived from ML Success Model TreeSHAP feature attribution & NLP complaint clusters"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>psychology</span>
                      🤖 ML Model + NLP
                    </span>
                  ) : rec.evidence_type === "review_nlp" ? (
                    <span
                      className="badge-pill"
                      style={{
                        backgroundColor: "rgba(45, 212, 191, 0.15)",
                        color: "var(--accent-review-nlp)",
                        border: "1px solid rgba(45, 212, 191, 0.3)",
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "4px",
                        fontWeight: 600,
                      }}
                      title="Derived from sentiment analysis and player complaint category volume"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>reviews</span>
                      💬 Review NLP Heuristic
                    </span>
                  ) : (
                    <span
                      className="badge-pill"
                      style={{
                        backgroundColor: "rgba(251, 146, 60, 0.15)",
                        color: "var(--accent-pricing)",
                        border: "1px solid rgba(251, 146, 60, 0.3)",
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "4px",
                        fontWeight: 600,
                      }}
                      title="Derived from genre IQR pricing distribution and comparable benchmarks"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>payments</span>
                      📊 Pricing Benchmark Rule
                    </span>
                  )}
                </div>
              </div>

              {/* Rationale */}
              <p style={{ fontSize: "13px", lineHeight: "1.7", color: "var(--text-secondary)", margin: 0 }}>
                {rec.rationale}
              </p>

              {/* Action Items List */}
              {rec.action_items && rec.action_items.length > 0 && (
                <div style={{ padding: "12px 16px", backgroundColor: "var(--bg-card)", borderRadius: "6px" }}>
                  <div style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "8px", textTransform: "uppercase" }}>
                    Recommended Action Checklist
                  </div>
                  <ul style={{ margin: 0, paddingLeft: "20px", fontSize: "13px", lineHeight: "1.7", color: "var(--text-secondary)" }}>
                    {rec.action_items.map((action, i) => (
                      <li key={i} style={{ marginBottom: "4px" }}>
                        {action}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Footer / Confidence Meter */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "8px", borderTop: "1px solid var(--border-subtle)", fontSize: "12px", color: "var(--text-secondary)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span>Confidence Score:</span>
                  <div style={{ width: "80px", height: "6px", backgroundColor: "var(--bg-card)", borderRadius: "3px", overflow: "hidden" }}>
                    <div style={{ width: `${confPct}%`, height: "100%", backgroundColor: "var(--accent-primary)", borderRadius: "3px" }} />
                  </div>
                  <span style={{ fontWeight: 700, color: "var(--text-primary)" }}>{confPct}%</span>
                </div>
                {rec.model_run_id && (
                  <span style={{ fontFamily: "var(--font-mono, monospace)", fontSize: "11px" }}>
                    Trace: {rec.model_run_id.slice(0, 10)}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
