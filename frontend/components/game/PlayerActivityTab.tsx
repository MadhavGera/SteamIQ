import React, { useMemo, useState } from "react";
import type { PlayerActivity } from "@/lib/api";

interface PlayerActivityTabProps {
  activity: PlayerActivity | null;
  gameTitle: string;
}

export function PlayerActivityTab({ activity, gameTitle }: PlayerActivityTabProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState<"all" | "recent">("all");

  const hasActivityData = Boolean(
    activity &&
      (activity.peak_ccu_24h != null ||
        activity.average_playtime_forever_hours != null ||
        (activity.ccu_history && activity.ccu_history.length > 0))
  );

  // Filter snapshots if timeframe toggled
  const filteredSnapshots = useMemo(() => {
    // HONEST-FALLBACK: Safe empty array fallback for snapshot history points
    const snaps = activity?.ccu_history ?? [];
    if (!snaps.length) return [];
    if (selectedTimeframe === "recent" && snaps.length > 5) {
      return snaps.slice(-5);
    }
    return snaps;
  }, [activity?.ccu_history, selectedTimeframe]);

  // Compute SVG chart points
  const chartData = useMemo(() => {
    if (!filteredSnapshots.length) return null;
    const counts = filteredSnapshots.map((s) => s.player_count);
    const minVal = Math.min(...counts);
    const maxVal = Math.max(...counts);
    const range = maxVal - minVal === 0 ? 1 : maxVal - minVal;

    const width = 640;
    const height = 180;
    const paddingX = 40;
    const paddingTop = 20;
    const paddingBottom = 30;
    const chartH = height - paddingTop - paddingBottom;
    const chartW = width - paddingX * 2;

    const points = filteredSnapshots.map((s, idx) => {
      const x = paddingX + (idx / Math.max(1, filteredSnapshots.length - 1)) * chartW;
      const normalizedY = (s.player_count - minVal) / range;
      const y = paddingTop + chartH - normalizedY * chartH;
      return { x, y, count: s.player_count, date: s.snapshot_at };
    });

    const pathD = points.reduce((acc, pt, idx) => {
      return idx === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
    }, "");

    const areaD = points.length > 0
      ? `${pathD} L ${points[points.length - 1].x} ${height - paddingBottom} L ${points[0].x} ${height - paddingBottom} Z`
      : "";

    return { points, pathD, areaD, minVal, maxVal, width, height, paddingBottom };
  }, [filteredSnapshots]);

  if (!hasActivityData || !activity) {
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
            monitoring
          </span>
        </div>
        <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "8px" }}>
          Player Activity Telemetry Pending for {gameTitle}
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "520px", margin: "0 auto", lineHeight: "1.6" }}>
          Concurrent player counts and empirical playtime metrics are polled during catalog ingestion. Run `make ingest` and `make materialize-marts` to record player activity snapshots.
        </p>
      </div>
    );
  }

  // HONEST-FALLBACK: Safe display string for peak 24h CCU
  const peak24hStr = activity.peak_ccu_24h != null ? activity.peak_ccu_24h.toLocaleString() : "--";
  // HONEST-FALLBACK: Safe display string for current CCU
  const currentCcuStr = activity.current_ccu != null ? activity.current_ccu.toLocaleString() : "--";
  // HONEST-FALLBACK: Safe display string for all time peak CCU
  const allTimePeakStr = activity.all_time_peak_ccu != null ? activity.all_time_peak_ccu.toLocaleString() : "--";
  // HONEST-FALLBACK: Safe display string for average playtime hours
  const avgPlaytimeStr = activity.average_playtime_forever_hours != null ? `${activity.average_playtime_forever_hours}h` : "--";
  // HONEST-FALLBACK: Safe display string for median playtime hours
  const medianPlaytimeStr = activity.median_playtime_forever_hours != null ? `${activity.median_playtime_forever_hours}h` : "--";
  // HONEST-FALLBACK: Safe display string for post-review engagement delta
  const postReviewDeltaStr = activity.post_review_engagement_delta_hours != null ? `+${activity.post_review_engagement_delta_hours}h` : "--";
  // HONEST-FALLBACK: Safe fallback for verified sample size
  const sampleSize = activity.sample_size_reviews ?? 0;
  // HONEST-FALLBACK: Safe empty array for retention cohorts
  const cohorts = activity.playtime_cohorts ?? [];
  const percentiles = activity.playtime_percentiles;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "24px" }}>
      {/* ── 1. Top Telemetry KPI Matrix ── */}
      <div className="kpi-matrix-grid">
        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">24H PEAK CONCURRENT</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--accent-primary)" }}>
              {peak24hStr}
            </div>
            <div className="kpi-card__sub">
              {/* HONEST-FALLBACK: Safe string for owners estimate */}
              Owners Bracket: {activity.owners_estimate ?? "Unestimated"}
            </div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">MEDIAN LIFETIME PLAYTIME</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--text-primary)" }}>
              {medianPlaytimeStr}
            </div>
            <div className="kpi-card__sub">50% of verified players exceed this</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">AVERAGE PLAYTIME</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--accent-light)" }}>
              {avgPlaytimeStr}
            </div>
            <div className="kpi-card__sub">Mean player engagement depth</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">POST-REVIEW ENGAGEMENT</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--success)" }}>
              {postReviewDeltaStr}
            </div>
            <div className="kpi-card__sub">Additional hours played after rating</div>
          </div>
        </div>
      </div>

      {/* ── 2. CCU Time-Series Telemetry & Activity Panel ── */}
      <div className="panel-grid-50-50">
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                <span className="badge-pill badge-pill--success" style={{ fontSize: "11px" }}>
                  ● LIVE TELEMETRY
                </span>
                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                  {activity.telemetry_mode === "snapshot_polled" ? "Steam API Polled Snapshot Feed" : "Telemetry Stream"}
                </span>
              </div>
              <h3 style={{ fontSize: "18px", fontWeight: 700, margin: 0 }}>
                Concurrent Players (CCU)
              </h3>
            </div>
            <div style={{ display: "flex", gap: "8px" }}>
              <button
                type="button"
                onClick={() => setSelectedTimeframe("all")}
                className={`btn btn--sm ${selectedTimeframe === "all" ? "btn--primary" : "btn--secondary"}`}
                style={{ padding: "4px 10px", fontSize: "12px" }}
              >
                All Snapshots
              </button>
              <button
                type="button"
                onClick={() => setSelectedTimeframe("recent")}
                className={`btn btn--sm ${selectedTimeframe === "recent" ? "btn--primary" : "btn--secondary"}`}
                style={{ padding: "4px 10px", fontSize: "12px" }}
              >
                Recent
              </button>
            </div>
          </div>

          {/* SVG CCU Line Chart */}
          {chartData && chartData.points.length > 1 ? (
            <div style={{ width: "100%", overflowX: "auto" }}>
              <svg
                viewBox={`0 0 ${chartData.width} ${chartData.height}`}
                style={{ width: "100%", height: "auto", display: "block" }}
              >
                <defs>
                  <linearGradient id="ccuGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Horizontal reference lines */}
                <line
                  x1="40"
                  y1="20"
                  x2={chartData.width - 40}
                  y2="20"
                  stroke="var(--border-subtle)"
                  strokeDasharray="4 4"
                />
                <line
                  x1="40"
                  y1={chartData.height - chartData.paddingBottom}
                  x2={chartData.width - 40}
                  y2={chartData.height - chartData.paddingBottom}
                  stroke="var(--border-subtle)"
                />

                {/* Shaded Area */}
                <path d={chartData.areaD} fill="url(#ccuGradient)" />

                {/* Trajectory Line */}
                <path
                  d={chartData.pathD}
                  fill="none"
                  stroke="var(--accent-primary)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Data Points */}
                {chartData.points.map((pt, i) => (
                  <g key={i}>
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r="4"
                      fill="var(--bg-card)"
                      stroke="var(--accent-primary)"
                      strokeWidth="2"
                    />
                    <text
                      x={pt.x}
                      y={pt.y - 10}
                      textAnchor="middle"
                      fill="var(--text-secondary)"
                      fontSize="10"
                      fontWeight="600"
                    >
                      {pt.count.toLocaleString()}
                    </text>
                    {pt.date && (
                      <text
                        x={pt.x}
                        y={chartData.height - 10}
                        textAnchor="middle"
                        fill="var(--text-muted)"
                        fontSize="9"
                      >
                        {new Date(pt.date).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                      </text>
                    )}
                  </g>
                ))}
              </svg>
            </div>
          ) : (
            <div style={{ padding: "36px 16px", textAlign: "center", color: "var(--text-secondary)" }}>
              <span className="material-symbols-outlined" style={{ fontSize: "32px", display: "block", marginBottom: "8px" }}>
                query_stats
              </span>
              Current concurrent player count: <strong>{currentCcuStr}</strong>. Additional time-series snapshots will accumulate as periodic telemetry ingestion runs.
            </div>
          )}

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              padding: "12px 16px",
              backgroundColor: "var(--bg-surface)",
              borderRadius: "6px",
              border: "1px solid var(--border-subtle)",
              fontSize: "12px",
            }}
          >
            <div>
              <span style={{ color: "var(--text-muted)" }}>Current Snapshot: </span>
              <strong style={{ color: "var(--text-primary)" }}>{currentCcuStr} CCU</strong>
            </div>
            <div>
              <span style={{ color: "var(--text-muted)" }}>All-Time Observed Peak: </span>
              <strong style={{ color: "var(--accent-light)" }}>{allTimePeakStr} CCU</strong>
            </div>
          </div>
        </div>

        {/* ── Playtime Percentiles & Milestones ── */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
              <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px" }}>
                EMPIRICAL PLAYTIME
              </span>
              <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                {sampleSize > 0 ? `N = ${sampleSize.toLocaleString()} Verified Steam Players` : "Catalog Benchmark"}
              </span>
            </div>
            <h3 style={{ fontSize: "18px", fontWeight: 700, margin: 0 }}>
              Playtime Percentile Benchmarks
            </h3>
          </div>

          <p style={{ color: "var(--text-secondary)", fontSize: "13px", lineHeight: "1.5", margin: 0 }}>
            Playtime percentiles reveal player drop-off vs mastery thresholds. For {gameTitle}, 50% of players engaged beyond {medianPlaytimeStr}, with top-decile players exceeding {percentiles.p90_hours != null ? `${percentiles.p90_hours}h` : "--"}.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "12px", marginTop: "8px" }}>
            <div style={{ padding: "14px", backgroundColor: "var(--bg-surface)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>
                25th Percentile (Entry)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
                {percentiles.p25_hours != null ? `${percentiles.p25_hours}h` : "--"}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                Early-stage campaign progression
              </div>
            </div>

            <div style={{ padding: "14px", backgroundColor: "var(--bg-surface)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>
                50th Percentile (Median)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 700, color: "var(--accent-primary)", marginTop: "4px" }}>
                {percentiles.median_hours != null ? `${percentiles.median_hours}h` : medianPlaytimeStr}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                Core campaign completion
              </div>
            </div>

            <div style={{ padding: "14px", backgroundColor: "var(--bg-surface)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>
                75th Percentile (Dedicated)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 700, color: "var(--accent-light)", marginTop: "4px" }}>
                {percentiles.p75_hours != null ? `${percentiles.p75_hours}h` : "--"}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                Post-game exploration &amp; NG+
              </div>
            </div>

            <div style={{ padding: "14px", backgroundColor: "var(--bg-surface)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>
                90th Percentile (Hardcore)
              </div>
              <div style={{ fontSize: "20px", fontWeight: 700, color: "var(--success)", marginTop: "4px" }}>
                {percentiles.p90_hours != null ? `${percentiles.p90_hours}h` : "--"}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                100% completion &amp; high heat runs
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── 3. Player Retention & Cohort Distribution ── */}
      {cohorts.length > 0 && (
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
              <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px" }}>
                RETENTION PROFILE
              </span>
              <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                Verified Player Cohorts Breakdown
              </span>
            </div>
            <h3 style={{ fontSize: "18px", fontWeight: 700, margin: 0 }}>
              Playtime Engagement Cohorts
            </h3>
          </div>

          {/* Visual Stacked Progress Bar */}
          <div
            style={{
              display: "flex",
              height: "16px",
              borderRadius: "8px",
              overflow: "hidden",
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border-subtle)",
            }}
          >
            {cohorts.map((c, idx) => {
              const colors = [
                "rgba(148, 163, 184, 0.7)",
                "rgba(56, 189, 248, 0.8)",
                "rgba(99, 102, 241, 0.85)",
                "rgba(34, 197, 94, 0.9)",
              ];
              const color = colors[idx % colors.length];
              return (
                <div
                  key={c.cohort}
                  style={{
                    width: `${c.share_pct}%`,
                    backgroundColor: color,
                    transition: "width 0.3s ease",
                  }}
                  title={`${c.label} (${c.hours_range}): ${c.share_pct}%`}
                />
              );
            })}
          </div>

          {/* Cohort Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px" }}>
            {cohorts.map((c, idx) => {
              const accentColors = [
                "var(--text-muted)",
                "var(--accent-primary)",
                "var(--accent-light)",
                "var(--success)",
              ];
              const borderCol = accentColors[idx % accentColors.length];
              return (
                <div
                  key={c.cohort}
                  style={{
                    padding: "16px",
                    backgroundColor: "var(--bg-surface)",
                    borderRadius: "8px",
                    border: "1px solid var(--border-subtle)",
                    borderTop: `3px solid ${borderCol}`,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-primary)" }}>
                      {c.label}
                    </span>
                    <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px" }}>
                      {c.hours_range}
                    </span>
                  </div>
                  <div style={{ fontSize: "24px", fontWeight: 800, color: borderCol, marginBottom: "4px" }}>
                    {c.share_pct.toFixed(1)}%
                  </div>
                  <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    {c.player_count.toLocaleString()} verified players
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── 4. Correlated Update & Activity Surges ── */}
      {activity.recent_event_impacts && activity.recent_event_impacts.length > 0 && (
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div>
            <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px", marginBottom: "6px", display: "inline-block" }}>
              ACTIVITY MILESTONES
            </span>
            <h3 style={{ fontSize: "18px", fontWeight: 700, margin: 0 }}>
              Correlated Event &amp; Patch Impact
            </h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {activity.recent_event_impacts.map((evt, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "16px",
                  backgroundColor: "var(--bg-surface)",
                  borderRadius: "8px",
                  border: "1px solid var(--border-subtle)",
                  flexWrap: "wrap",
                  gap: "12px",
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "14px" }}>
                    {evt.patch_name}
                  </div>
                  <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "2px" }}>
                    {evt.patch_date ? new Date(evt.patch_date).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "--"}
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  {evt.ccu_change_pct != null && (
                    <span
                      style={{
                        fontSize: "13px",
                        fontWeight: 700,
                        color: evt.ccu_change_pct >= 0 ? "var(--success)" : "var(--danger)",
                      }}
                    >
                      {evt.ccu_change_pct >= 0 ? `+${evt.ccu_change_pct.toFixed(1)}%` : `${evt.ccu_change_pct.toFixed(1)}%`} CCU
                    </span>
                  )}
                  {/* HONEST-FALLBACK: Safe string verdict */}
                  <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px" }}>
                    {evt.verdict ?? "Observed Milestone"}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <p style={{ fontSize: "11px", color: "var(--text-muted)", margin: 0, lineHeight: "1.5" }}>
            Note: Activity surges and telemetry milestones reflect observed correlation within official patch windows and do not assert sole direct causality.
          </p>
        </div>
      )}
    </div>
  );
}
