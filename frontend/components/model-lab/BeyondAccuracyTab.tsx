"use client";

import React from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface BeyondAccuracyRow {
  model_name: string;
  catalog_coverage_at_10?: number;
  novelty_at_10?: number;
  diversity_at_10?: number;
  head_share_at_10?: number;
  gini_coefficient_at_10?: number;
}

interface MmrPoint {
  lambda_param: number;
  ndcg_at_10: number;
  diversity_at_10: number;
  ndcg_at_10_lo95?: number;
  ndcg_at_10_hi95?: number;
  paired_ndcg_diff_vs_lambda1?: number;
}

interface BeyondAccuracyPayload {
  metrics_at_10?: BeyondAccuracyRow[];
  mmr_reranking?: MmrPoint[];
  metadata?: Record<string, unknown>;
}

interface BeyondAccuracyTabProps {
  payload: BeyondAccuracyPayload | null;
}

/** SVG line chart for MMR Trade-Off (Diversity vs NDCG@10 across lambda) */
function MmrTradeoffChart({ points }: { points: MmrPoint[] }) {
  if (!points || points.length === 0) return null;

  // Sort by lambda ascending
  const sorted = [...points].sort((a, b) => a.lambda_param - b.lambda_param);

  const width = 560;
  const height = 240;
  const padding = { top: 20, right: 30, bottom: 40, left: 50 };

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const minNdcg = Math.min(...sorted.map((p) => p.ndcg_at_10)) * 0.9;
  const maxNdcg = Math.max(...sorted.map((p) => p.ndcg_at_10)) * 1.1;

  const minDiv = Math.min(...sorted.map((p) => p.diversity_at_10)) * 0.95;
  const maxDiv = Math.max(...sorted.map((p) => p.diversity_at_10)) * 1.05;

  const getX = (p: MmrPoint) =>
    padding.left + ((p.diversity_at_10 - minDiv) / (maxDiv - minDiv || 1)) * plotWidth;
  const getY = (p: MmrPoint) =>
    padding.top + plotHeight - ((p.ndcg_at_10 - minNdcg) / (maxNdcg - minNdcg || 1)) * plotHeight;

  const pathD = sorted.reduce((acc, p, i) => `${acc} ${i === 0 ? "M" : "L"} ${getX(p)} ${getY(p)}`, "");

  return (
    <div className="card" style={{ padding: "20px", marginBottom: "24px" }}>
      <div style={{ marginBottom: "12px" }}>
        <h4 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)" }}>
          Maximal Marginal Relevance (MMR) Trade-Off Frontier
        </h4>
        <p style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
          Pareto frontier plotting list diversity vs ranking accuracy (NDCG@10) across diversity parameter λ ∈ [0, 1].
        </p>
      </div>

      <div style={{ overflowX: "auto" }}>
        <svg width={width} height={height} style={{ backgroundColor: "var(--bg-surface-low)", borderRadius: "var(--radius-sm)" }} aria-label="MMR Trade-Off Frontier Chart">
          {/* Grid lines */}
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotHeight} stroke="var(--border-subtle)" />
          <line x1={padding.left} y1={padding.top + plotHeight} x2={padding.left + plotWidth} y2={padding.top + plotHeight} stroke="var(--border-subtle)" />

          {/* Line curve */}
          <path d={pathD} fill="none" stroke="var(--accent-primary)" strokeWidth="2.5" />

          {/* Points & Labels */}
          {sorted.map((p, idx) => {
            const cx = getX(p);
            const cy = getY(p);
            return (
              <g key={idx}>
                <circle cx={cx} cy={cy} r="5" fill="var(--accent-primary)" stroke="var(--bg-surface-lowest)" strokeWidth="2" />
                <text
                  x={cx}
                  y={cy - 10}
                  fill="var(--text-primary)"
                  fontSize="11px"
                  textAnchor="middle"
                  fontFamily="var(--font-mono, monospace)"
                >
                  λ={p.lambda_param}
                </text>
              </g>
            );
          })}

          {/* Axis Labels */}
          <text x={width / 2} y={height - 8} fill="var(--text-muted)" fontSize="11px" textAnchor="middle">
            List Diversity (Intra-List Distance) →
          </text>
          <text
            x={-height / 2}
            y={16}
            fill="var(--text-muted)"
            fontSize="11px"
            textAnchor="middle"
            transform="rotate(-90)"
          >
            NDCG@10 →
          </text>
        </svg>
      </div>
    </div>
  );
}

export function BeyondAccuracyTab({ payload }: BeyondAccuracyTabProps) {
  if (!payload || !payload.metrics_at_10 || payload.metrics_at_10.length === 0) {
    return (
      <RecsysEmptyState
        command="make recsys-eval"
        title="Beyond-Accuracy Metrics Not Materialized"
        message="Beyond-accuracy metrics (coverage, novelty, diversity, Gini) have not been computed. Run the evaluation script below:"
      />
    );
  }

  const rows = payload.metrics_at_10;
  const mmrPoints = payload.mmr_reranking || [];

  return (
    <div>
      <div style={{ marginBottom: "20px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
          Beyond-Accuracy Diagnostics
        </h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
          Evaluating recommendation fairness, serendipity, catalog exploration, and concentration inequality.
        </p>
      </div>

      {/* MMR Trade-off Curve */}
      {mmrPoints.length > 0 && <MmrTradeoffChart points={mmrPoints} />}

      {/* Metrics Table */}
      <div className="competitor-table-container">
        <table className="competitor-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Catalog Coverage</th>
              <th>Novelty (self-info bits)</th>
              <th>Diversity (cosine dist)</th>
              <th>Head Share (% top-20% items)</th>
              <th>Gini Inequality</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model_name}>
                <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>{row.model_name}</td>
                <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                  {row.catalog_coverage_at_10 !== undefined ? `${(row.catalog_coverage_at_10 * 100).toFixed(1)}%` : "—"}
                </td>
                <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                  {row.novelty_at_10 !== undefined ? row.novelty_at_10.toFixed(2) : "—"}
                </td>
                <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                  {row.diversity_at_10 !== undefined ? row.diversity_at_10.toFixed(3) : "—"}
                </td>
                <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                  {row.head_share_at_10 !== undefined ? `${(row.head_share_at_10 * 100).toFixed(1)}%` : "—"}
                </td>
                <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                  {row.gini_coefficient_at_10 !== undefined ? row.gini_coefficient_at_10.toFixed(3) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
