"use client";

import React from "react";

interface RouterRow {
  model_name: string;
  accuracy: number;
  accuracy_ci?: [number, number];
  macro_f1: number;
  macro_f1_ci?: [number, number];
  ece?: number;
  is_headline?: boolean;
}

interface AssistantTabProps {
  leaderboardPayload: Record<string, unknown> | null;
  calibrationPayload: Record<string, unknown> | null;
  retrievalPayload: Record<string, unknown> | null;
  errorsPayload: Record<string, unknown> | null;
}

export function AssistantTab({
  leaderboardPayload,
  calibrationPayload,
  retrievalPayload,
  errorsPayload,
}: AssistantTabProps) {
  if (!leaderboardPayload) {
    return (
      <div
        className="card"
        style={{
          padding: "48px 24px",
          textAlign: "center",
          background: "var(--bg-surface-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "12px",
        }}
      >
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "40px", color: "var(--text-muted)", marginBottom: "12px" }}
        >
          smart_toy
        </span>
        <h3 style={{ fontSize: "18px", fontWeight: 600, color: "var(--text-primary)" }}>
          Assistant Reports Not Yet Materialized
        </h3>
        <p
          style={{
            fontSize: "14px",
            color: "var(--text-secondary)",
            maxWidth: "480px",
            margin: "8px auto 0",
          }}
        >
          Run <code>make assistant-train</code> to fit the intent routers, calibrate temperature scaling,
          and generate empirical benchmark reports.
        </p>
      </div>
    );
  }

  const routerRows = (leaderboardPayload.routers as RouterRow[]) || [];
  const verifiedGoldCount = Number(leaderboardPayload.verified_gold_count || 0);
  const totalGoldCount = Number(leaderboardPayload.total_gold_count || 0);
  const isPreliminary = verifiedGoldCount < 100;

  const cal = (calibrationPayload || {}) as Record<string, unknown>;
  const tau = Number(cal.tau || 0.85);
  const selectivePrecision = Number(cal.selective_precision || 0);
  const coverage = Number(cal.coverage || 0);
  const oosAuroc = Number(cal.oos_auroc || 0);
  const ece = Number(cal.ece || 0);

  const ret = (retrievalPayload || {}) as Record<string, unknown>;
  const denseRecall3 = Number(ret.dense_recall_at_3 || 0);
  const tfidfRecall3 = Number(ret.tfidf_recall_at_3 || 0);
  const denseMrr = Number(ret.dense_mrr || 0);
  const tfidfMrr = Number(ret.tfidf_mrr || 0);

  const errors = (errorsPayload?.error_breakdown as Array<{
    query: string;
    true_intent: string;
    pred_intent: string;
    confidence: number;
    resolved_method?: string;
  }>) || [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* Gold Queries Verification Status Banner */}
      {isPreliminary && (
        <div
          style={{
            padding: "14px 18px",
            background: "rgba(245, 158, 11, 0.12)",
            border: "1px solid rgba(245, 158, 11, 0.35)",
            borderRadius: "8px",
            display: "flex",
            alignItems: "center",
            gap: "12px",
          }}
        >
          <span className="material-symbols-outlined" style={{ color: "#fbbf24", fontSize: "22px" }}>
            warning
          </span>
          <div>
            <strong style={{ color: "#fbbf24", fontSize: "14px" }}>
              unverified gold: preliminary
            </strong>
            <p style={{ margin: "2px 0 0", fontSize: "13px", color: "var(--text-secondary)" }}>
              {verifiedGoldCount} of {totalGoldCount} gold query annotations are human-verified. Metrics
              are preliminary until full expert verification milestone (&ge;100 verified queries).
            </p>
          </div>
        </div>
      )}

      {/* 1. Router Variants Leaderboard */}
      <div
        className="card"
        style={{
          background: "var(--bg-surface-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "12px",
          padding: "20px",
        }}
      >
        <div style={{ marginBottom: "16px" }}>
          <h2 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)" }}>
            Intent Router Variants & Headline Benchmark
          </h2>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Evaluated on held-out gold domain queries across 12 domain intents plus explicit out-of-scope.
          </p>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table className="table" style={{ width: "100%", fontSize: "13px", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)", textAlign: "left" }}>
                <th style={{ padding: "10px" }}>Router Architecture</th>
                <th style={{ padding: "10px" }}>Role</th>
                <th style={{ padding: "10px" }}>Accuracy (95% CI)</th>
                <th style={{ padding: "10px" }}>Macro-F1 (95% CI)</th>
                <th style={{ padding: "10px" }}>ECE</th>
              </tr>
            </thead>
            <tbody>
              {routerRows.map((r, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: "1px solid var(--border-subtle)",
                    background: r.is_headline ? "rgba(99, 102, 241, 0.06)" : "transparent",
                  }}
                >
                  <td style={{ padding: "10px", fontWeight: r.is_headline ? 700 : 500 }}>
                    <code>{r.model_name}</code>
                    {r.is_headline && (
                      <span
                        className="badge-pill"
                        style={{
                          marginLeft: "8px",
                          background: "rgba(99, 102, 241, 0.2)",
                          color: "#a5b4fc",
                          fontSize: "11px",
                        }}
                      >
                        Headline
                      </span>
                    )}
                  </td>
                  <td style={{ padding: "10px", color: "var(--text-secondary)" }}>
                    {r.is_headline ? "Headline Production Router" : "Baseline Model"}
                  </td>
                  <td style={{ padding: "10px" }}>
                    {(r.accuracy * 100).toFixed(1)}%
                    {r.accuracy_ci && (
                      <span style={{ color: "var(--text-muted)", fontSize: "11px", marginLeft: "4px" }}>
                        [{(r.accuracy_ci[0] * 100).toFixed(1)}%, {(r.accuracy_ci[1] * 100).toFixed(1)}%]
                      </span>
                    )}
                  </td>
                  <td style={{ padding: "10px" }}>
                    {r.macro_f1.toFixed(3)}
                    {r.macro_f1_ci && (
                      <span style={{ color: "var(--text-muted)", fontSize: "11px", marginLeft: "4px" }}>
                        [{r.macro_f1_ci[0].toFixed(3)}, {r.macro_f1_ci[1].toFixed(3)}]
                      </span>
                    )}
                  </td>
                  <td style={{ padding: "10px" }}>
                    {r.ece !== undefined ? r.ece.toFixed(4) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Calibration & Selective Abstention Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "16px" }}>
        <div
          className="card"
          style={{
            background: "var(--bg-surface-elevated)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "12px",
            padding: "20px",
          }}
        >
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "8px" }}>
            Uncertainty & Calibration
          </h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "16px" }}>
            Multi-class Temperature Scaling on held-out calibration split.
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "13px" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Abstention Threshold (tau):</span>
              <strong>{tau.toFixed(2)}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Selective Precision at tau:</span>
              <strong style={{ color: selectivePrecision >= 0.95 ? "#34d399" : "inherit" }}>
                {(selectivePrecision * 100).toFixed(1)}% (Target &ge; 95%)
              </strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Query Coverage at tau:</span>
              <strong>{(coverage * 100).toFixed(1)}%</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Expected Calibration Error:</span>
              <strong>{ece.toFixed(4)}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Out-of-Scope AUROC:</span>
              <strong>{oosAuroc.toFixed(3)}</strong>
            </div>
          </div>
        </div>

        {/* 3. Extractive Retrieval Fallback Benchmark */}
        <div
          className="card"
          style={{
            background: "var(--bg-surface-elevated)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "12px",
            padding: "20px",
          }}
        >
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "8px" }}>
            Extractive Passage Retrieval (RAG)
          </h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "16px" }}>
            Retrieval fallback over pre-materialized review intelligence passages.
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "13px" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>MiniLM Recall@3:</span>
              <strong>{(denseRecall3 * 100).toFixed(1)}%</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>TF-IDF Cosine Recall@3:</span>
              <span style={{ color: "var(--text-secondary)" }}>{(tfidfRecall3 * 100).toFixed(1)}%</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>MiniLM MRR:</span>
              <strong>{denseMrr.toFixed(3)}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>TF-IDF Cosine MRR:</span>
              <span style={{ color: "var(--text-secondary)" }}>{tfidfMrr.toFixed(3)}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Generative Hallucination Rate:</span>
              <strong style={{ color: "#34d399" }}>0.0% (Strictly Extractive)</strong>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Error Breakdown Diagnostic Table */}
      {errors.length > 0 && (
        <div
          className="card"
          style={{
            background: "var(--bg-surface-elevated)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "12px",
            padding: "20px",
          }}
        >
          <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "6px" }}>
            Router Error Analysis Sample
          </h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "16px" }}>
            Held-out gold queries where the headline router misclassified or abstained.
          </p>

          <div style={{ overflowX: "auto" }}>
            <table className="table" style={{ width: "100%", fontSize: "12px", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-subtle)", textAlign: "left" }}>
                  <th style={{ padding: "8px" }}>Query</th>
                  <th style={{ padding: "8px" }}>True Intent</th>
                  <th style={{ padding: "8px" }}>Predicted Intent</th>
                  <th style={{ padding: "8px" }}>Confidence</th>
                  <th style={{ padding: "8px" }}>Method</th>
                </tr>
              </thead>
              <tbody>
                {errors.slice(0, 8).map((err, idx) => (
                  <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "8px", fontStyle: "italic", maxWidth: "280px" }}>
                      &ldquo;{err.query}&rdquo;
                    </td>
                    <td style={{ padding: "8px" }}>
                      <code>{err.true_intent}</code>
                    </td>
                    <td style={{ padding: "8px", color: "#f87171" }}>
                      <code>{err.pred_intent}</code>
                    </td>
                    <td style={{ padding: "8px" }}>
                      {(err.confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "8px", color: "var(--text-muted)" }}>
                      {err.resolved_method || "logistic"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
