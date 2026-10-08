"use client";

import React, { useState } from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface ShapGlobalPayload {
  global_shap?: Record<string, number> | Array<{ feature: string; mean_abs_shap: number }>;
  feature_importance_gain?: Record<string, number> | Array<{ feature: string; importance: number }>;
  feature_importance_split?: Record<string, number> | Array<{ feature: string; importance: number }>;
  metadata?: Record<string, unknown>;
}

interface ExplainabilityTabProps {
  payload: ShapGlobalPayload | null;
}

export function ExplainabilityTab({ payload }: ExplainabilityTabProps) {
  const [metricMode, setMetricMode] = useState<"shap" | "gain" | "split">("shap");

  if (!payload || (!payload.global_shap && !payload.feature_importance_gain)) {
    return (
      <RecsysEmptyState
        command="make recsys-eval"
        title="Global Explainability Report Missing"
        message="TreeSHAP global attribution values and LightGBM feature importance rankings have not been materialized. Run the evaluation pipeline below:"
      />
    );
  }

  // Parse dictionary or array entries into normalized { feature, value } list
  const extractEntries = (
    data?: Record<string, number> | Array<{ feature: string; mean_abs_shap?: number; importance?: number }>
  ): Array<{ feature: string; value: number }> => {
    if (!data) return [];
    if (Array.isArray(data)) {
      return data.map((item) => ({
        feature: item.feature,
        // HONEST-FALLBACK: Read feature importance from mean_abs_shap or importance field
        value: item.mean_abs_shap ?? item.importance ?? 0,
      }));
    }
    return Object.entries(data).map(([feature, value]) => ({
      feature,
      value: typeof value === "number" ? value : 0,
    }));
  };

  const activeData =
    metricMode === "shap"
      ? extractEntries(payload.global_shap)
      : metricMode === "gain"
      ? extractEntries(payload.feature_importance_gain)
      : extractEntries(payload.feature_importance_split);

  // Sort descending by importance magnitude
  const sorted = [...activeData].sort((a, b) => b.value - a.value).slice(0, 15);
  const maxVal = Math.max(0.001, ...sorted.map((item) => item.value));

  return (
    <div>
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
          <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
            Global Feature Attributions (Explainability)
          </h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
            TreeSHAP mean absolute feature attributions and LightGBM tree importance metrics across ranker features.
          </p>
        </div>

        {/* Importance Type Switch */}
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
            onClick={() => setMetricMode("shap")}
            style={{
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: metricMode === "shap" ? "var(--accent-primary)" : "transparent",
              color: metricMode === "shap" ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
            }}
          >
            TreeSHAP (|SHAP|)
          </button>
          <button
            type="button"
            onClick={() => setMetricMode("gain")}
            style={{
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: metricMode === "gain" ? "var(--accent-primary)" : "transparent",
              color: metricMode === "gain" ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
            }}
          >
            Total Gain
          </button>
          <button
            type="button"
            onClick={() => setMetricMode("split")}
            style={{
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: metricMode === "split" ? "var(--accent-primary)" : "transparent",
              color: metricMode === "split" ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
            }}
          >
            Split Count
          </button>
        </div>
      </div>

      {/* SHAP Bar List */}
      <div className="card" style={{ padding: "24px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          {sorted.map((item, idx) => {
            const pct = Math.min(100, Math.max(6, Math.round((item.value / maxVal) * 100)));
            const label = item.feature.replace(/_/g, " ");

            return (
              <div key={idx} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <span
                  style={{
                    width: "220px",
                    fontSize: "13px",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                  title={item.feature}
                >
                  {label}
                </span>

                <div
                  style={{
                    flex: 1,
                    height: "10px",
                    backgroundColor: "var(--bg-surface-low)",
                    borderRadius: "4px",
                    overflow: "hidden",
                  }}
                >
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

                <span
                  style={{
                    width: "70px",
                    textAlign: "right",
                    fontFamily: "var(--font-mono, monospace)",
                    fontSize: "12px",
                    color: "var(--text-primary)",
                  }}
                >
                  {item.value.toFixed(4)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
