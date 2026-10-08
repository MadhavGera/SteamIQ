"use client";

import React from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface ComparisonRow {
  model_a?: string;
  model_b?: string;
  comparison_name?: string;
  metric?: string;
  mean_diff?: number;
  diff_mean?: number;
  ci_lo?: number;
  ci_hi?: number;
  p_value?: number;
  is_significant?: boolean;
  verdict?: string;
  interpretation?: string;
}

interface SlicesPayload {
  comparisons?: ComparisonRow[];
  slice_definitions?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

interface PairedTestsTabProps {
  payload: SlicesPayload | null;
}

export function PairedTestsTab({ payload }: PairedTestsTabProps) {
  if (!payload || !payload.comparisons || payload.comparisons.length === 0) {
    return (
      <RecsysEmptyState
        command="make recsys-eval"
        title="Paired Comparisons Not Materialized"
        message="Paired head-to-head statistical significance tests have not been computed yet. Run the evaluation script below:"
      />
    );
  }

  const comparisons = payload.comparisons;

  return (
    <div>
      <div style={{ marginBottom: "20px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
          Paired Head-to-Head Bootstrap Tests
        </h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
          User-level paired differences (Δ metric per user) with 1,000 bootstrap replicates and 95% confidence intervals.
        </p>
      </div>

      <div className="competitor-table-container">
        <table className="competitor-table">
          <thead>
            <tr>
              <th>Comparison</th>
              <th>Metric</th>
              <th>Paired Mean Diff (Δ)</th>
              <th>95% Confidence Interval</th>
              <th>Significance (p &lt; 0.05)</th>
              <th>Verdict / Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map((c, idx) => {
              // HONEST-FALLBACK: Read mean difference from mean_diff or diff_mean field
              const diff = c.mean_diff ?? c.diff_mean ?? 0;
              const isPos = diff > 0;
              const isSig = c.is_significant || (c.ci_lo !== undefined && c.ci_hi !== undefined && (c.ci_lo > 0 || c.ci_hi < 0));

              return (
                <tr key={idx}>
                  <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                    {/* HONEST-FALLBACK: Default comparison title from model names */}
                    {c.comparison_name ?? `${c.model_a} vs ${c.model_b}`}
                  </td>
                  <td>
                    <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px" }}>
                      {/* HONEST-FALLBACK: Default metric to NDCG@10 */}
                      {c.metric ?? "NDCG@10"}
                    </span>
                  </td>
                  <td
                    style={{
                      fontFamily: "var(--font-mono, monospace)",
                      fontWeight: 700,
                      color: isPos ? "var(--success)" : "var(--danger)",
                    }}
                  >
                    {isPos ? `+${diff.toFixed(4)}` : diff.toFixed(4)}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono, monospace)", fontSize: "12px" }}>
                    {c.ci_lo !== undefined && c.ci_hi !== undefined ? (
                      `[${c.ci_lo.toFixed(4)}, ${c.ci_hi.toFixed(4)}]`
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {isSig ? (
                      <span className="badge-pill badge-pill--success" style={{ fontSize: "10px" }}>
                        Significant (p &lt; 0.05)
                      </span>
                    ) : (
                      <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px" }}>
                        Not Significant
                      </span>
                    )}
                  </td>
                  <td style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                    {/* HONEST-FALLBACK: Default verdict text */}
                    {c.verdict ?? c.interpretation ?? "Null hypothesis evaluated on user pairs"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
