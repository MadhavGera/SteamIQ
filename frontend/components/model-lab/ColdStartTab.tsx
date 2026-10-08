"use client";

import React from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface StudyModelRow {
  model_name?: string;
  ndcg_at_10?: number;
  hit_rate_at_10?: number;
  n_eval_items?: number;
}

interface ColdStartStudy {
  study_type?: string;
  variant_label?: string;
  n_new_items?: number;
  n_eval_users?: number;
  limitations_note?: string;
  is_fair_simulation?: boolean;
  models?: StudyModelRow[] | Record<string, StudyModelRow>;
}

interface ColdStartPayload {
  sparse_targets_study?: ColdStartStudy;
  simulated_new_items_study?: ColdStartStudy;
  item_holdout_study?: ColdStartStudy;
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

  const newItemsStudy = payload.simulated_new_items_study;
  const sparseStudy = payload.sparse_targets_study;
  const holdoutStudy = payload.item_holdout_study;

  const renderStudyTable = (study?: ColdStartStudy, title?: string) => {
    if (!study || !study.models) return null;

    const rows: StudyModelRow[] = Array.isArray(study.models)
      ? study.models
      : Object.entries(study.models).map(([name, data]) => ({
          model_name: name,
          ...data,
        }));

    return (
      <div style={{ marginBottom: "24px" }}>
        <div style={{ marginBottom: "8px" }}>
          <h4 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)" }}>{title}</h4>
          {study.variant_label && (
            <p style={{ fontSize: "12px", color: "var(--text-secondary)" }}>{study.variant_label}</p>
          )}
        </div>

        <div className="competitor-table-container">
          <table className="competitor-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>NDCG@10</th>
                <th>HR@10</th>
                <th>Evaluated Items / Users</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                    {/* HONEST-FALLBACK: Default model name in cold start study */}
                    {row.model_name ?? "Model"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                    {row.ndcg_at_10 !== undefined ? row.ndcg_at_10.toFixed(4) : "—"}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono, monospace)" }}>
                    {row.hit_rate_at_10 !== undefined ? row.hit_rate_at_10.toFixed(4) : "—"}
                  </td>
                  <td style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    {/* HONEST-FALLBACK: Default sample count if not specified */}
                    {row.n_eval_items ?? study.n_new_items ?? study.n_eval_users ?? "All"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Verbatim limitations caveat text */}
        {study.limitations_note && (
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
            <strong style={{ color: "var(--warning)" }}>Caveat / Protocol Note: </strong>
            {study.limitations_note}
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <div style={{ marginBottom: "20px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
          Cold-Start & Catalog Slice Diagnostics
        </h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
          Benchmarking model performance under zero-interaction cold items, sparse user histories, and head vs tail items.
        </p>
      </div>

      {/* 1. Simulated New Items Study */}
      {renderStudyTable(
        newItemsStudy,
        "Zero-Interaction Simulated New Items Study"
      )}

      {/* 2. Sparse Targets Study */}
      {renderStudyTable(
        sparseStudy,
        "Sparse Target & Short User History Slices"
      )}

      {/* 3. Item Holdout Study */}
      {renderStudyTable(
        holdoutStudy,
        "Catalog Stratification (Head, Torso, Tail Items)"
      )}
    </div>
  );
}
