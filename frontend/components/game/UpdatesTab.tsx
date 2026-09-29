import React from "react";
import type { UpdateImpactFeed } from "@/lib/api";

interface UpdatesTabProps {
  updatesFeed: UpdateImpactFeed | null;
  gameTitle: string;
}

export function UpdatesTab({ updatesFeed, gameTitle }: UpdatesTabProps) {
  if (!updatesFeed || updatesFeed.total_updates === 0 || !updatesFeed.updates.length) {
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
            history_toggle_off
          </span>
        </div>
        <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "8px" }}>
          No Update Events Recorded for {gameTitle} Yet
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "520px", margin: "0 auto", lineHeight: "1.6" }}>
          SteamIQ records official developer announcements via the Steam News API and calculates 14-day before/after telemetry shifts. Run `make ingest-news` and `make materialize-updates` to analyze update impacts.
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
            <span className="kpi-card__eyebrow">UPDATES ANALYZED</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--accent-primary)" }}>
              {updatesFeed.total_updates}
            </div>
            <div className="kpi-card__sub">Major patches &amp; activity milestones</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">LATEST OBSERVED VERDICT</span>
          </div>
          <div>
            <div
              className="kpi-card__num"
              style={{
                fontSize: "22px",
                color:
                  updatesFeed.latest_verdict?.toLowerCase().includes("positive")
                    ? "var(--success)"
                    : updatesFeed.latest_verdict?.toLowerCase().includes("backlash")
                    ? "var(--danger)"
                    : "var(--text-primary)",
              }}
            >
              {updatesFeed.latest_verdict || "Neutral / Stable"}
            </div>
            <div className="kpi-card__sub">Telemetry response across 14d window</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">AVG SENTIMENT SHIFT</span>
          </div>
          <div>
            <div
              className="kpi-card__num"
              style={{
                color:
                  updatesFeed.average_sentiment_delta != null
                    ? updatesFeed.average_sentiment_delta >= 0
                      ? "var(--success)"
                      : "var(--danger)"
                    : "var(--text-secondary)",
              }}
            >
              {updatesFeed.average_sentiment_delta != null
                ? updatesFeed.average_sentiment_delta >= 0
                  ? `+${updatesFeed.average_sentiment_delta.toFixed(1)}%`
                  : `${updatesFeed.average_sentiment_delta.toFixed(1)}%`
                : "--"}
            </div>
            <div className="kpi-card__sub">Net positive delta pre vs post patch</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">METHODOLOGY</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ fontSize: "22px", color: "var(--accent-light)" }}>
              ±14-Day Window
            </div>
            <div className="kpi-card__sub">Strictly observed telemetry correlation</div>
          </div>
        </div>
      </div>

      {/* ── 2. Non-Causal Methodology Notice ── */}
      <div
        style={{
          padding: "14px 18px",
          backgroundColor: "rgba(56, 189, 248, 0.08)",
          border: "1px solid rgba(56, 189, 248, 0.2)",
          borderRadius: "8px",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          fontSize: "13px",
          color: "var(--text-secondary)",
        }}
      >
        <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)", fontSize: "20px" }}>
          info
        </span>
        <div>
          <strong style={{ color: "var(--text-primary)" }}>Telemetry Note:</strong> Before/after shifts are observed correlations within the 14-day window around patch deployment. External factors such as seasonal promotions, influencer streams, and algorithm shifts can co-occur.
        </div>
      </div>

      {/* ── 3. Update Impact Cards Feed ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {updatesFeed.updates.map((item) => {
          // HONEST-FALLBACK: Numeric sentiment delta or 0
          const delta = item.sentiment_delta_pct ?? 0;
          const isPos = delta >= 0;
          const patchDateStr = item.patch_date ? new Date(item.patch_date).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "Recent";

          return (
            <div key={item.id} className="card" style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "10px" }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                    <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
                      {item.patch_name}
                    </h3>
                    {item.is_inferred ? (
                      <span
                        className="badge-pill"
                        style={{
                          backgroundColor: "rgba(168, 85, 247, 0.15)",
                          color: "var(--accent-inferred)",
                          border: "1px solid rgba(168, 85, 247, 0.3)",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "4px",
                          fontSize: "11px",
                          fontWeight: 600,
                        }}
                        title="Detected via statistical review volume surge rather than official changelog"
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>insights</span>
                        Detected from Activity
                      </span>
                    ) : (
                      <span
                        className="badge-pill"
                        style={{
                          backgroundColor: "rgba(56, 189, 248, 0.15)",
                          color: "var(--accent-light)",
                          border: "1px solid rgba(56, 189, 248, 0.3)",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "4px",
                          fontSize: "11px",
                          fontWeight: 600,
                        }}
                        title="Authentic developer patch notes from official Steam News API"
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>campaign</span>
                        From Steam Announcement
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>
                    Published: <strong>{patchDateStr}</strong> • Window: ±{item.window_days} days
                  </div>
                </div>

                <span
                  className={`badge-pill ${
                    item.observed_sentiment_verdict.toLowerCase().includes("positive")
                      ? "badge-pill--success"
                      : item.observed_sentiment_verdict.toLowerCase().includes("backlash")
                      ? "badge-pill--danger"
                      : "badge-pill--neutral"
                  }`}
                  style={{ fontSize: "12px", fontWeight: 600 }}
                >
                  {item.observed_sentiment_verdict}
                </span>
              </div>

              {/* Pre vs Post Comparison Strip */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                  gap: "12px",
                  padding: "12px",
                  backgroundColor: "var(--bg-card)",
                  borderRadius: "6px",
                  fontSize: "13px",
                }}
              >
                <div>
                  <div style={{ color: "var(--text-secondary)", fontSize: "11px", textTransform: "uppercase" }}>
                    Net Sentiment (Pre → Post)
                  </div>
                  <div style={{ fontWeight: 700, marginTop: "2px", color: "var(--text-primary)" }}>
                    {item.pre_sentiment_positive_pct != null ? `${item.pre_sentiment_positive_pct.toFixed(1)}%` : "--"} →{" "}
                    {item.post_sentiment_positive_pct != null ? `${item.post_sentiment_positive_pct.toFixed(1)}%` : "--"} (
                    <span style={{ color: isPos ? "var(--success)" : "var(--danger)" }}>
                      {isPos ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`}
                    </span>
                    )
                  </div>
                </div>

                <div>
                  <div style={{ color: "var(--text-secondary)", fontSize: "11px", textTransform: "uppercase" }}>
                    Player CCU Shift (Pre → Post)
                  </div>
                  <div style={{ fontWeight: 700, marginTop: "2px", color: "var(--text-primary)" }}>
                    {item.pre_avg_ccu != null && item.post_avg_ccu != null ? (
                      <>
                        {item.pre_avg_ccu.toLocaleString()} → {item.post_avg_ccu.toLocaleString()} (
                        <span
                          style={{
                            color:
                              item.ccu_change_pct != null
                                ? item.ccu_change_pct >= 0
                                  ? "var(--success)"
                                  : "var(--danger)"
                                : "var(--text-secondary)",
                          }}
                        >
                          {item.ccu_change_pct != null
                            ? item.ccu_change_pct >= 0
                              ? `+${item.ccu_change_pct.toFixed(1)}%`
                              : `${item.ccu_change_pct.toFixed(1)}%`
                            : "--"}
                        </span>
                        )
                      </>
                    ) : (
                      <span style={{ color: "var(--text-secondary)", fontWeight: 400 }}>
                        Snapshot accumulation pending
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Resolved / Emerging Complaints */}
              {((item.top_resolved_complaints && item.top_resolved_complaints.length > 0) ||
                (item.top_emerging_complaints && item.top_emerging_complaints.length > 0)) && (
                <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", fontSize: "12px" }}>
                  {item.top_resolved_complaints && item.top_resolved_complaints.length > 0 && (
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--success)" }}>
                      <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>check_circle</span>
                      <span>Resolved complaints: {item.top_resolved_complaints.map((c) => c.category).join(", ")}</span>
                    </div>
                  )}
                  {item.top_emerging_complaints && item.top_emerging_complaints.length > 0 && (
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--danger)" }}>
                      <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>warning</span>
                      <span>Emerging friction: {item.top_emerging_complaints.map((c) => c.category).join(", ")}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Correlation Summary */}
              <p style={{ fontSize: "13px", lineHeight: "1.6", color: "var(--text-secondary)", margin: 0 }}>
                {item.correlation_summary}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
