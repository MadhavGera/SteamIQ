"use client";

import React, { useState } from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface TwoTowerVariant {
  variant?: string;
  variant_name?: string;
  description?: string;
  test_ndcg_at_10_mean?: number;
  val_ndcg_at_10_mean?: number;
  test_ndcg_at_10_std?: number;
  val_ndcg_at_10_std?: number;
  paired_ndcg_diff_vs_full?: number;
  paired_diff_vs_full?: number;
  paired_ndcg_diff_lo95?: number;
  paired_ndcg_diff_hi95?: number;
  ci_lo95?: number;
  ci_hi95?: number;
  significant_vs_full?: boolean;
}

interface RankerAblation {
  ablated_group?: string;
  dropped_group?: string;
  description?: string;
  val_ndcg_at_10?: number;
  test_ndcg_at_10?: number;
  val_ndcg_diff_vs_full?: number;
  test_ndcg_diff_vs_full?: number;
  delta_ndcg_vs_full?: number;
  paired_diff_vs_full?: number;
  val_ndcg_diff_lo95?: number;
  val_ndcg_diff_hi95?: number;
  test_ndcg_diff_lo95?: number;
  test_ndcg_diff_hi95?: number;
  is_significant_test?: boolean;
  active_features_count?: number;
  features_remaining?: number;
}

interface AblationsPayload {
  variants?: TwoTowerVariant[];
  ablations?: RankerAblation[];
  metadata?: Record<string, unknown>;
}

interface AblationsTabProps {
  ttPayload: AblationsPayload | null;
  rankerPayload: AblationsPayload | null;
}

export function AblationsTab({ ttPayload, rankerPayload }: AblationsTabProps) {
  const [subTab, setSubTab] = useState<"two_tower" | "ranker">("two_tower");

  const ttVariants = ttPayload?.variants || [];
  const rankerAblations = rankerPayload?.ablations || [];

  if (ttVariants.length === 0 && rankerAblations.length === 0) {
    return (
      <RecsysEmptyState
        command="make recsys-eval"
        title="Ablation Studies Not Materialized"
        message="Feature drop and architecture ablation reports are not available. Run the evaluation pipeline below:"
      />
    );
  }

  // Find max NDCG for horizontal bar scaling
  const allNdcgs = [
    ...ttVariants.map((v) => v.test_ndcg_at_10_mean || 0),
    ...rankerAblations.map((a) => a.test_ndcg_at_10 || 0),
  ];
  const maxNdcg = Math.max(0.1, ...allNdcgs);

  return (
    <div>
      {/* Sub-tab navigation */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
            Feature Group & Architecture Ablations
          </h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
            Measuring the empirical impact on ranking accuracy when feature modalities and components are withheld.
          </p>
        </div>

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
            onClick={() => setSubTab("two_tower")}
            style={{
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: subTab === "two_tower" ? "var(--accent-primary)" : "transparent",
              color: subTab === "two_tower" ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
            }}
          >
            Two-Tower Ablations
          </button>
          <button
            type="button"
            onClick={() => setSubTab("ranker")}
            style={{
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: subTab === "ranker" ? "var(--accent-primary)" : "transparent",
              color: subTab === "ranker" ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
            }}
          >
            Ranker Ablations
          </button>
        </div>
      </div>

      {/* Two-Tower Ablations View */}
      {subTab === "two_tower" && (
        <div>
          {ttVariants.length === 0 ? (
            <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)" }}>
              No Two-Tower ablation runs recorded in report.
            </div>
          ) : (
            <div className="card" style={{ padding: "20px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                {ttVariants.map((v, idx) => {
                  /* HONEST-FALLBACK: Mean NDCG fallback if omitted */
                  const val = v.test_ndcg_at_10_mean ?? v.val_ndcg_at_10_mean ?? 0;
                  const pct = Math.min(100, Math.max(5, (val / maxNdcg) * 100));
                  /* HONEST-FALLBACK: Variant label fallback if omitted */
                  const variantLabel = v.variant_name ?? v.variant ?? "Variant";
                  const isFull = idx === 0 || variantLabel.includes("full");
                  /* HONEST-FALLBACK: Paired diff fallback if omitted */
                  const diff = v.paired_ndcg_diff_vs_full ?? v.paired_diff_vs_full;

                  return (
                    <div key={idx} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                        <span style={{ fontWeight: 600, color: isFull ? "var(--accent-primary)" : "var(--text-primary)" }}>
                          {variantLabel}
                          {isFull && <span style={{ marginLeft: "6px", fontSize: "11px", color: "var(--text-muted)" }}>(Baseline)</span>}
                        </span>
                        <div style={{ display: "flex", gap: "12px", fontFamily: "var(--font-mono, monospace)" }}>
                          <span style={{ color: "var(--text-primary)" }}>{val.toFixed(4)} NDCG@10</span>
                          {diff !== undefined && !isFull && (
                            <span style={{ color: diff < 0 ? "var(--danger)" : "var(--success)" }}>
                              {diff > 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Bar Track */}
                      <div style={{ height: "10px", backgroundColor: "var(--bg-surface-low)", borderRadius: "4px", overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            backgroundColor: isFull ? "var(--accent-primary)" : "var(--text-secondary)",
                            borderRadius: "4px",
                            transition: "width 0.4s ease",
                          }}
                        />
                      </div>
                      {v.description && (
                        <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>{v.description}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Ranker Ablations View */}
      {subTab === "ranker" && (
        <div>
          {rankerAblations.length === 0 ? (
            <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)" }}>
              No Ranker ablation runs recorded in report.
            </div>
          ) : (
            <div className="card" style={{ padding: "20px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                {rankerAblations.map((a, idx) => {
                  /* HONEST-FALLBACK: Metric fallback if omitted */
                  const val = a.val_ndcg_at_10 ?? a.test_ndcg_at_10 ?? 0;
                  const pct = Math.min(100, Math.max(5, (val / maxNdcg) * 100));
                  /* HONEST-FALLBACK: Delta fallback if omitted */
                  const delta = a.val_ndcg_diff_vs_full ?? a.test_ndcg_diff_vs_full ?? a.delta_ndcg_vs_full ?? a.paired_diff_vs_full;
                  /* HONEST-FALLBACK: Dropped group fallback name */
                  const groupName = a.ablated_group ?? a.dropped_group ?? "Feature Group";

                  return (
                    <div key={idx} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                        <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                          Drop {groupName}
                        </span>
                        <div style={{ display: "flex", gap: "12px", fontFamily: "var(--font-mono, monospace)" }}>
                          <span style={{ color: "var(--text-primary)" }}>{val.toFixed(4)} NDCG@10</span>
                          {delta !== undefined && (
                            <span style={{ color: delta < 0 ? "var(--danger)" : "var(--success)" }}>
                              {delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Bar Track */}
                      <div style={{ height: "10px", backgroundColor: "var(--bg-surface-low)", borderRadius: "4px", overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            backgroundColor: "var(--accent-primary)",
                            borderRadius: "4px",
                            transition: "width 0.4s ease",
                          }}
                        />
                      </div>
                      {a.description && (
                        <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>{a.description}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
