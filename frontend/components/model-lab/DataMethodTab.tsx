"use client";

import React from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface DatasetManifest {
  universe_size?: number;
  n_users?: number;
  n_interactions?: number;
  n_positives?: number;
  sparsity_pct?: number;
  density?: number;
  dataset_version?: string;
  min_interactions_per_user?: number;
  min_interactions_per_item?: number;
  created_at?: string;
}

interface LimitationItem {
  id: string;
  title: string;
  description: string;
  severity?: string;
}

interface DataMethodPayload {
  manifest?: DatasetManifest;
  split_protocol?: string | Record<string, unknown>;
  limitations?: LimitationItem[];
  metadata?: Record<string, unknown>;
}

interface DataMethodTabProps {
  datasetPayload: DataMethodPayload | null;
  limitationsPayload: DataMethodPayload | null;
}

export function DataMethodTab({
  datasetPayload,
  limitationsPayload,
}: DataMethodTabProps) {
  if (!datasetPayload && !limitationsPayload) {
    return (
      <RecsysEmptyState
        command="make recsys-eval"
        title="Dataset Manifest & Limitations Not Materialized"
        message="Dataset documentation and engineering limitation manifests are missing. Run the pipeline below:"
      />
    );
  }

  const manifest = datasetPayload?.manifest;
  const protocol = datasetPayload?.split_protocol;
  const limitations = limitationsPayload?.limitations || [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* 1. Dataset Manifest */}
      {manifest && (
        <section>
          <div style={{ marginBottom: "14px" }}>
            <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
              Dataset Manifest & Universe Counts
            </h3>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
              Frozen dataset snapshot parameters and interaction matrix topology.
            </p>
          </div>

          <div className="kpi-matrix-grid" style={{ marginBottom: "0" }}>
            <div className="kpi-card">
              <div className="kpi-card__top">
                <span className="kpi-card__eyebrow">Modeled Universe</span>
                <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)", fontSize: "20px" }}>
                  grid_view
                </span>
              </div>
              <div className="kpi-card__num" style={{ color: "var(--text-primary)" }}>
                {manifest.universe_size?.toLocaleString() || "3,000"}
              </div>
              <div className="kpi-card__sub">Catalog games</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-card__top">
                <span className="kpi-card__eyebrow">Training Users</span>
                <span className="material-symbols-outlined" style={{ color: "var(--accent-light)", fontSize: "20px" }}>
                  group
                </span>
              </div>
              <div className="kpi-card__num" style={{ color: "var(--text-primary)" }}>
                {manifest.n_users?.toLocaleString() || "—"}
              </div>
              <div className="kpi-card__sub">Filtered user sequences</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-card__top">
                <span className="kpi-card__eyebrow">Interactions</span>
                <span className="material-symbols-outlined" style={{ color: "var(--success)", fontSize: "20px" }}>
                  thumb_up
                </span>
              </div>
              <div className="kpi-card__num" style={{ color: "var(--text-primary)" }}>
                {manifest.n_interactions?.toLocaleString() || manifest.n_positives?.toLocaleString() || "—"}
              </div>
              <div className="kpi-card__sub">Positive user-game pairs</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-card__top">
                <span className="kpi-card__eyebrow">Matrix Density</span>
                <span className="material-symbols-outlined" style={{ color: "var(--warning)", fontSize: "20px" }}>
                  data_array
                </span>
              </div>
              <div className="kpi-card__num" style={{ color: "var(--text-primary)" }}>
                {manifest.density !== undefined
                  ? `${(manifest.density * 100).toFixed(3)}%`
                  : manifest.sparsity_pct !== undefined
                  ? `${(100 - manifest.sparsity_pct).toFixed(3)}%`
                  : "—"}
              </div>
              <div className="kpi-card__sub">Observed positive density</div>
            </div>
          </div>
        </section>
      )}

      {/* 2. Split Protocol */}
      {protocol && (
        <section>
          <div style={{ marginBottom: "12px" }}>
            <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
              Leave-One-Out Evaluation Protocol
            </h3>
          </div>

          <div
            className="card"
            style={{
              padding: "20px",
              backgroundColor: "var(--bg-surface-low)",
              lineHeight: 1.6,
              fontSize: "13px",
              color: "var(--text-secondary)",
            }}
          >
            {typeof protocol === "string" ? (
              <p style={{ whiteSpace: "pre-wrap" }}>{protocol}</p>
            ) : (
              <pre
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "12px",
                  color: "var(--text-primary)",
                  overflowX: "auto",
                }}
              >
                {JSON.stringify(protocol, null, 2)}
              </pre>
            )}
          </div>
        </section>
      )}

      {/* 3. Verbatim Limitations List */}
      {limitations.length > 0 && (
        <section>
          <div style={{ marginBottom: "14px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span className="material-symbols-outlined" style={{ fontSize: "20px", color: "var(--warning)" }}>
                report_problem
              </span>
              <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
                Recommender System Limitations & Assumptions
              </h3>
            </div>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
              Catalog limitations and methodological boundaries rendered verbatim from the engineering audit.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "12px" }}>
            {limitations.map((lim) => (
              <div
                key={lim.id}
                className="card"
                style={{
                  padding: "16px",
                  backgroundColor: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "6px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
                  <span style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-primary)" }}>
                    {lim.title}
                  </span>
                  <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px", fontFamily: "var(--font-mono, monospace)" }}>
                    {lim.id}
                  </span>
                </div>
                <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                  {lim.description}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
