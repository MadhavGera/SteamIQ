"use client";

import React from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface StudyModelRow {
  model_name?: string;
  hits_k?: number;
  n_users?: number;
  ndcg_at_10?: number;
  hit_rate_at_10?: number;
  mean_rank?: number;
  mean_rank_lo95?: number;
  mean_rank_hi95?: number;
  n_eval_items?: number;
}

interface ColdStartStudy {
  study_type?: string;
  variant_label?: string;
  primary_evidence?: boolean;
  is_optimistic_proxy?: boolean;
  n_new_items?: number;
  n_eval_users?: number;
  n_held_out_items?: number;
  limitations_note?: string;
  note?: string;
  is_fair_simulation?: boolean;
  models?: StudyModelRow[] | Record<string, StudyModelRow>;
}

interface ColdStartPayload {
  item_holdout_study?: ColdStartStudy;
  sparse_targets_study?: ColdStartStudy;
  simulated_new_items_study?: ColdStartStudy;
  metadata?: Record<string, unknown>;
}

interface ColdStartTabProps {
  payload: ColdStartPayload | null;
}

export function ColdStartTab({ payload }: ColdStartTabProps) {
  if (!payload) {
    return (
      <RecsysEmptyState
        command="make recsys-eval"
        title="Cold-Start & Slices Report Not Materialized"
        message="Cold-start and catalog slice evaluation data is missing. Run the evaluation script below:"
      />
    );
  }

  const holdoutStudy = payload.item_holdout_study;
  const sparseStudy = payload.sparse_targets_study;
  const newItemsStudy = payload.simulated_new_items_study;

  const renderStudyTable = (study?: ColdStartStudy, title?: string, subtitle?: string) => {
    if (!study || !study.models) return null;

    const rows: StudyModelRow[] = Array.isArray(study.models)
      ? study.models
      : Object.entries(study.models).map(([name, data]) => ({
          model_name: name,
          ...data,
        }));

    const noteText = study.note ?? study.limitations_note;

    const formatModelName = (name?: string) => {
      if (!name) return "Model";
      if (name === "two_tower_masked_all_proxies_neutralized") return "Two-Tower (All Proxies Neutralized z=0)";
      if (name === "two_tower_masked_standardized_z0") return "Two-Tower (Train Pop z=0, Reviews Intact)";
      if (name === "two_tower_masked_train_pos_raw_0") return "Two-Tower (Train Pop Raw 0)";
      if (name === "two_tower_masked") return "Two-Tower (ID Masked, Popularity Leaked)";
      if (name === "two_tower_unmasked") return "Two-Tower (Unmasked, ID & Popularity Leaked)";
      if (name.includes("two_tower_masked")) return "Two-Tower Masked [Reference: All-Time Stats]";
      if (name === "two_tower_intact") return "Two-Tower Intact (Reference)";
      if (name === "two_tower" || name.includes("all-time-stats architecture") || name === "two_tower (all-time-stats architecture, clean=False)") return "Two-Tower Holdout [all-time-stats architecture, clean=False]";
      if (name === "content") return "Content Cosine";
      if (name === "popularity") return "Popularity";
      if (name === "als") return "ALS";
      if (name === "itemknn") return "Item-kNN";
      if (name === "random") return "Random (Analytic)";
      return name;
    };

    return (
      <div style={{ marginBottom: "28px" }}>
        <div style={{ marginBottom: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <h4 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)" }}>{title}</h4>
            {study.primary_evidence && (
              <span
                style={{
                  fontSize: "10px",
                  padding: "2px 6px",
                  borderRadius: "var(--radius-xs)",
                  backgroundColor: "rgba(16, 185, 129, 0.15)",
                  color: "var(--success)",
                  fontWeight: 700,
                  textTransform: "uppercase",
                }}
              >
                Primary Benchmark
              </span>
            )}
            {study.is_optimistic_proxy && (
              <span
                style={{
                  fontSize: "10px",
                  padding: "2px 6px",
                  borderRadius: "var(--radius-xs)",
                  backgroundColor: "rgba(245, 158, 11, 0.15)",
                  color: "var(--warning)",
                  fontWeight: 700,
                  textTransform: "uppercase",
                }}
              >
                Optimistic Proxy
              </span>
            )}
          </div>
          {subtitle && (
            <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>{subtitle}</p>
          )}
        </div>

        <div className="competitor-table-container">
          <table className="competitor-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Hits / Sample</th>
                <th>NDCG@10</th>
                <th>HR@10</th>
                <th>Mean Rank (95% CI)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => {
                const sampleCount = row.n_users ?? study.n_eval_users ?? 0;
                const hitsStr = row.hits_k !== undefined && sampleCount > 0 ? `${row.hits_k} / ${sampleCount}` : "—";
                const meanRankStr =
                  row.mean_rank !== undefined && row.mean_rank_lo95 !== undefined && row.mean_rank_hi95 !== undefined
                    ? `${row.mean_rank.toFixed(1)} [${row.mean_rank_lo95.toFixed(1)}, ${row.mean_rank_hi95.toFixed(1)}]`
                    : row.mean_rank !== undefined
                    ? row.mean_rank.toFixed(1)
                    : "—";

                return (
                  <tr key={idx}>
                    <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                      {/* HONEST-FALLBACK: Default model name in cold start study */}
                      {formatModelName(row.model_name)}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono, monospace)" }}>{hitsStr}</td>
                    <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                      {row.ndcg_at_10 !== undefined ? row.ndcg_at_10.toFixed(4) : "—"}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                      {row.hit_rate_at_10 !== undefined ? row.hit_rate_at_10.toFixed(4) : "—"}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono, monospace)", fontSize: "12px" }}>{meanRankStr}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Verbatim limitations caveat text */}
        {noteText && (
          <div
            style={{
              marginTop: "8px",
              padding: "10px 14px",
              backgroundColor: "var(--bg-surface-lowest)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              fontSize: "12px",
              color: "var(--text-secondary)",
              lineHeight: 1.5,
            }}
          >
            <strong style={{ color: study.primary_evidence ? "var(--success)" : "var(--warning)" }}>
              {study.primary_evidence ? "Protocol Description: " : "Caveat / Simulation Note: "}
            </strong>
            {noteText}
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <div style={{ marginBottom: "20px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
          Cold-Start & Discovery Diagnostics
        </h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
          Benchmarking recommendations under zero-interaction catalog holdouts and inference-time masking simulations.
        </p>
        <div
          style={{
            marginTop: "12px",
            padding: "10px 14px",
            backgroundColor: "rgba(99, 102, 241, 0.08)",
            border: "1px solid rgba(99, 102, 241, 0.25)",
            borderRadius: "var(--radius-sm)",
            fontSize: "12px",
            color: "var(--text-primary)",
            lineHeight: 1.5,
          }}
        >
          <strong>Empirical Finding:</strong> Conditions (a) and (b) leak popularity proxies. Under honest zero-interaction holdout evaluation, content-only recommender beats random discovery (mean rank 1,084.6 vs random 1,544.0), whereas two-tower dual encoders score unseen items at or below random (mean rank 2,126.6 vs random 1,544.0) once popularity proxies are neutralized.
        </div>
      </div>

      {/* 1. Primary Evidence: True Item Holdout Study */}
      {renderStudyTable(
        holdoutStudy,
        "1. True Item Holdout Study (90% Catalog Retrained, N=184 Users)",
        "300 torso/tail items completely withheld from training pairs; models retrained from scratch."
      )}

      {/* 2. Sparse Targets Subgroup Study */}
      {renderStudyTable(
        sparseStudy,
        "2. Sparse Targets Subgroup Study (<= 10 Train Positives, N=553 Users)",
        "Target item ID masked at inference on model trained with full catalog (optimistic semantic prior proxy)."
      )}

      {/* 3. Simulated New Items Study */}
      {renderStudyTable(
        newItemsStudy,
        "3. Simulated 10% New Items Study (C2 Fair Simulation, N=437 Users)",
        "Inference-time masking simulation with rating and review popularity proxies neutralized to catalog mean."
      )}
    </div>
  );
}
