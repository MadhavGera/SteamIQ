"use client";

import React, { useState } from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface MetricRecord {
  n_users?: number;
  ndcg_at_10?: number;
  ndcg_at_10_lo95?: number;
  ndcg_at_10_hi95?: number;
  ndcg_at_20?: number;
  hit_rate_at_10?: number;
  hit_rate_at_10_lo95?: number;
  hit_rate_at_10_hi95?: number;
  hit_rate_at_20?: number;
  mrr_at_10?: number;
  mrr_at_10_lo95?: number;
  mrr_at_10_hi95?: number;
  catalog_coverage_at_10?: number;
  diversity_at_10?: number;
  novelty_at_10?: number;
  head_share_at_10?: number;
}

interface ModelRow {
  model_name: string;
  stage?: string;
  family?: string;
  val_metrics?: MetricRecord;
  test_metrics?: MetricRecord;
  hyperparameters?: Record<string, unknown>;
}

interface LeaderboardPayload {
  models?: ModelRow[];
  metadata?: {
    dataset_version?: string;
    evaluated_at?: string;
  };
}

interface LeaderboardTabProps {
  payload: LeaderboardPayload | null;
}

/** Inline SVG error bar renderer for 95% Confidence Interval */
function ErrorBar({
  val,
  lo,
  hi,
  maxVal = 0.25,
}: {
  val?: number;
  lo?: number;
  hi?: number;
  maxVal?: number;
}) {
  if (val === undefined || val === null) return <span style={{ color: "var(--text-muted)" }}>—</span>;

  // HONEST-FALLBACK: Default error bound to center value if bounds not present
  const minB = lo ?? val;
  // HONEST-FALLBACK: Default error bound to center value if bounds not present
  const maxB = hi ?? val;

  const width = 90;
  const height = 18;

  const scale = (x: number) => Math.max(0, Math.min(width, (x / maxVal) * width));
  const cx = scale(val);
  const x1 = scale(minB);
  const x2 = scale(maxB);

  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
      <span style={{ fontFamily: "var(--font-mono, monospace)", fontSize: "12px", minWidth: "48px", textAlign: "right" }}>
        {val.toFixed(3)}
      </span>
      <svg width={width} height={height} style={{ overflow: "visible" }} aria-label={`Metric ${val.toFixed(3)} with 95% CI ${minB.toFixed(3)} to ${maxB.toFixed(3)}`}>
        {/* Error bar line */}
        <line x1={x1} y1={height / 2} x2={x2} y2={height / 2} stroke="var(--text-muted)" strokeWidth="1.5" />
        {/* Caps */}
        <line x1={x1} y1={height / 2 - 3} x2={x1} y2={height / 2 + 3} stroke="var(--text-muted)" strokeWidth="1.5" />
        <line x1={x2} y1={height / 2 - 3} x2={x2} y2={height / 2 + 3} stroke="var(--text-muted)" strokeWidth="1.5" />
        {/* Mean point marker */}
        <circle cx={cx} cy={height / 2} r="3" fill="var(--accent-primary)" />
      </svg>
    </div>
  );
}

export function LeaderboardTab({ payload }: LeaderboardTabProps) {
  const [split, setSplit] = useState<"test" | "val">("test");

  if (!payload || !payload.models || payload.models.length === 0) {
    return (
      <RecsysEmptyState
        command="make recsys-eval"
        title="Leaderboard Not Materialized"
        message="Leaderboard evaluation benchmarks are missing. Run the evaluation pipeline below to compute metrics across models:"
      />
    );
  }

  const rows = payload.models;

  // Split deployable models from reference/non-deployable models
  const isDeployable = (name: string) => !name.includes("leak") && !name.includes("full");
  const deployableRows = rows.filter((r) => isDeployable(r.model_name));
  const referenceRows = rows.filter((r) => !isDeployable(r.model_name));

  const renderTable = (modelRows: ModelRow[], title: string, subtitle?: string) => (
    <div style={{ marginBottom: "28px" }}>
      <div style={{ marginBottom: "12px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>{title}</h3>
        {subtitle && <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>{subtitle}</p>}
      </div>

      <div className="competitor-table-container">
        <table className="competitor-table">
          <thead>
            <tr>
              <th>Model Name</th>
              <th>Stage</th>
              <th>NDCG@10 (95% CI)</th>
              <th>NDCG@20</th>
              <th>HR@10 (95% CI)</th>
              <th>HR@20</th>
              <th>MRR@10</th>
            </tr>
          </thead>
          <tbody>
            {modelRows.map((row) => {
              const m = split === "test" ? row.test_metrics : row.val_metrics;
              const isChamp = row.stage === "champion" || row.model_name.includes("clean");

              return (
                <tr
                  key={row.model_name}
                  style={{
                    backgroundColor: isChamp ? "rgba(102, 192, 244, 0.05)" : undefined,
                  }}
                >
                  <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span>{row.model_name}</span>
                      {row.stage === "champion" && (
                        <span className="badge-pill badge-pill--success" style={{ fontSize: "9px" }}>
                          Champion
                        </span>
                      )}
                    </div>
                  </td>
                  <td>
                    <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px" }}>
                      {/* HONEST-FALLBACK: Model stage default if omitted */}
                      {row.stage ?? "candidate"}
                    </span>
                  </td>
                  <td>
                    <ErrorBar val={m?.ndcg_at_10} lo={m?.ndcg_at_10_lo95} hi={m?.ndcg_at_10_hi95} maxVal={0.2} />
                  </td>
                  <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                    {m?.ndcg_at_20 !== undefined ? m.ndcg_at_20.toFixed(3) : "—"}
                  </td>
                  <td>
                    <ErrorBar val={m?.hit_rate_at_10} lo={m?.hit_rate_at_10_lo95} hi={m?.hit_rate_at_10_hi95} maxVal={0.3} />
                  </td>
                  <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                    {m?.hit_rate_at_20 !== undefined ? m.hit_rate_at_20.toFixed(3) : "—"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                    {m?.mrr_at_10 !== undefined ? m.mrr_at_10.toFixed(3) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div>
      {/* Header and Val vs Test Toggle */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Offline evaluation over held-out leave-one-out interaction sequences with bootstrap 95% confidence intervals.
          </p>
        </div>

        {/* Split Toggle */}
        <div
          style={{
            display: "inline-flex",
            backgroundColor: "var(--bg-surface-lowest)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-sm)",
            padding: "2px",
          }}
        >
          <button
            type="button"
            onClick={() => setSplit("test")}
            style={{
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: split === "test" ? "var(--accent-primary)" : "transparent",
              color: split === "test" ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
            }}
          >
            Test Split (Primary)
          </button>
          <button
            type="button"
            onClick={() => setSplit("val")}
            style={{
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: split === "val" ? "var(--accent-primary)" : "transparent",
              color: split === "val" ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
            }}
          >
            Validation Split
          </button>
        </div>
      </div>

      {/* 1. Deployable Models Table */}
      {renderTable(
        deployableRows,
        "Deployable Models",
        "Online-safe models that do not rely on test-time feature leakages or all-time item statistics. Headline production champion: recsys_two_tower_clean (with train-only item stats the ranker's gain over two-tower is not significant at +0.0021 [-0.0026, +0.0065]; all-time-stats gain of +0.0111 is leakage-confounded)."
      )}

      {/* 2. Reference Models Table */}
      {referenceRows.length > 0 &&
        renderTable(
          referenceRows,
          "Reference Models",
          "Reference models using all-time item stats (non-deployable upper bounds for diagnostic benchmarking only)."
        )}
    </div>
  );
}
