"use client";

import React from "react";
import { RecsysModelInfo } from "@/lib/api";

interface RecsysModelSelectorProps {
  models: RecsysModelInfo[];
  selectedModel: string;
  onSelectModel: (modelName: string) => void;
  disabled?: boolean;
}

export function RecsysModelSelector({
  models,
  selectedModel,
  onSelectModel,
  disabled = false,
}: RecsysModelSelectorProps) {
  // Allowed models in Recommendation Engine per user request: ItemKNN, ALS, Two Tower
  const ALLOWED_RECOMMENDER_MODELS = [
    "recsys_two_tower_clean",
    "recsys_two_tower",
    "recsys_itemknn",
    "recsys_als",
  ];
  const deployableModels = models.filter(
    (m) => m.deployable && ALLOWED_RECOMMENDER_MODELS.includes(m.model_name)
  );

  const getCleanDisplayName = (m: RecsysModelInfo) => {
    if (m.model_name === "recsys_itemknn") return "ItemKNN";
    if (m.model_name === "recsys_als") return "ALS";
    if (m.model_name === "recsys_two_tower_clean" || m.model_name === "recsys_two_tower") return "Two Tower";
    return m.display_name;
  };

  // Format metric string strictly from API returned values (never hardcoded)
  const formatMetric = (m: RecsysModelInfo) => {
    if (m.test_ndcg_at_10_mean === null || m.test_ndcg_at_10_mean === undefined) {
      return "not evaluated";
    }
    const meanStr = m.test_ndcg_at_10_mean.toFixed(3);
    if (m.test_ndcg_at_10_std !== null && m.test_ndcg_at_10_std !== undefined && m.test_ndcg_at_10_std > 0) {
      return `NDCG@10: ${meanStr} ± ${m.test_ndcg_at_10_std.toFixed(3)} (${m.n_seeds} seeds)`;
    }
    return `NDCG@10: ${meanStr}`;
  };

  if (deployableModels.length === 0) {
    return null;
  }

  return (
    <div style={{ marginBottom: "24px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "8px",
          marginBottom: "12px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "var(--accent-primary)" }}>
            tune
          </span>
          <label style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-primary)" }}>
            Recommendation Engine:
          </label>
        </div>
        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
          Selectable models are trained offline on the frozen dataset.
        </span>
      </div>

      {/* Segmented Grid / Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
          gap: "10px",
        }}
      >
        {deployableModels.map((m) => {
          const isSelected = selectedModel === m.model_name;
          const isReady = m.is_ready;

          return (
            <button
              key={m.model_name}
              type="button"
              disabled={disabled || !isReady}
              onClick={() => onSelectModel(m.model_name)}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                justifyContent: "space-between",
                padding: "12px 14px",
                borderRadius: "var(--radius-card)",
                backgroundColor: isSelected ? "var(--bg-surface-raised)" : "var(--bg-surface)",
                border: isSelected ? "2px solid var(--accent-primary)" : "1px solid var(--border-subtle)",
                boxShadow: isSelected ? "0 0 12px var(--accent-glow)" : "none",
                cursor: !isReady || disabled ? "not-allowed" : "pointer",
                opacity: !isReady || disabled ? 0.5 : 1,
                textAlign: "left",
                transition: "all var(--transition-fast)",
                minHeight: "84px",
              }}
            >
              <div style={{ width: "100%", display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "6px" }}>
                <span
                  style={{
                    fontSize: "13px",
                    fontWeight: isSelected ? 700 : 600,
                    color: isSelected ? "var(--accent-primary)" : "var(--text-primary)",
                  }}
                >
                  {getCleanDisplayName(m)}
                </span>

                {m.is_preview && (
                  <span
                    className="badge-pill badge-pill--warning"
                    style={{ fontSize: "9px", padding: "1px 5px", textTransform: "uppercase" }}
                    title="Preview: Serving-path online retrieval differs from offline re-ranking."
                  >
                    Preview
                  </span>
                )}
              </div>

              <div style={{ marginTop: "6px", width: "100%" }}>
                {isReady ? (
                  <span
                    style={{
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "11px",
                      color: isSelected ? "var(--text-primary)" : "var(--text-secondary)",
                    }}
                  >
                    {formatMetric(m)}
                  </span>
                ) : (
                  <span className="badge-pill badge-pill--danger" style={{ fontSize: "10px", padding: "1px 6px" }}>
                    Not trained / unready
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
