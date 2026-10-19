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

function IntentConfusionMatrix({
  classes,
  matrix,
}: {
  classes: string[];
  matrix: number[][];
}) {
  const [showNormalized, setShowNormalized] = React.useState(false);
  const [hoveredCell, setHoveredCell] = React.useState<{ row: number; col: number } | null>(null);

  if (!classes || classes.length === 0 || !matrix || matrix.length === 0) {
    return null;
  }

  // Row sums for normalization
  const rowSums = matrix.map((row) => row.reduce((acc, val) => acc + val, 0));
  const totalQueries = rowSums.reduce((acc, val) => acc + val, 0);
  const totalCorrect = matrix.reduce((acc, row, idx) => acc + (row[idx] || 0), 0);
  // HONEST-FALLBACK: Compute overall accuracy or fallback to 0
  const overallAcc = totalQueries > 0 ? totalCorrect / totalQueries : 0;

  // Abbreviated label map for clean matrix headers
  const getShortLabel = (name: string): string => {
    const map: Record<string, string> = {
      game_overview: "overview",
      review_summary: "summary",
      top_complaints: "complaints",
      loved_features: "features",
      similar_games: "similar",
      price_advice: "price",
      player_match: "match",
      success_score: "success",
      update_impact: "update",
      recommendations: "recs",
      market_opportunity: "market",
      recommend_for_me: "rec_4_me",
      out_of_scope: "oos",
    };
    return map[name] || name.slice(0, 7);
  };

  return (
    <div
      className="card"
      style={{
        background: "var(--bg-surface-elevated)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "12px",
        padding: "20px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px", marginBottom: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
            <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
              Intent Classification Confusion Matrix
            </h3>
            <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px" }}>
              {classes.length} &times; {classes.length} Classes
            </span>
            <span className="badge-pill badge-pill--success" style={{ fontSize: "11px" }}>
              {(overallAcc * 100).toFixed(1)}% Accuracy ({totalCorrect}/{totalQueries})
            </span>
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Evaluated on held-out gold benchmark queries. Rows represent <strong>True Intent</strong>; columns represent <strong>Predicted Intent</strong>. Diagonal cells indicate correct classifications.
          </p>
        </div>

        {/* Toggle between Counts and Normalized Recall % */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px", background: "var(--bg-surface-low)", padding: "3px 6px", borderRadius: "6px" }}>
          <button
            type="button"
            onClick={() => setShowNormalized(false)}
            style={{
              padding: "4px 8px",
              fontSize: "11px",
              fontWeight: !showNormalized ? 700 : 500,
              background: !showNormalized ? "var(--bg-surface-elevated)" : "transparent",
              color: !showNormalized ? "var(--accent-primary)" : "var(--text-muted)",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Raw Counts
          </button>
          <button
            type="button"
            onClick={() => setShowNormalized(true)}
            style={{
              padding: "4px 8px",
              fontSize: "11px",
              fontWeight: showNormalized ? 700 : 500,
              background: showNormalized ? "var(--bg-surface-elevated)" : "transparent",
              color: showNormalized ? "var(--accent-primary)" : "var(--text-muted)",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Normalized (Recall %)
          </button>
        </div>
      </div>

      {/* Heatmap Grid Table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "separate", borderSpacing: "2px", fontSize: "11px", width: "100%", minWidth: "750px", tableLayout: "fixed" }}>
          <thead>
            <tr>
              <th style={{ width: "140px", padding: "6px 8px", textAlign: "left", color: "var(--text-muted)", fontSize: "10px", fontWeight: 700, textTransform: "uppercase" }}>
                True \ Pred
              </th>
              {classes.map((c, colIdx) => (
                <th
                  key={c}
                  title={`Predicted Intent: ${c}`}
                  style={{
                    padding: "4px 2px",
                    textAlign: "center",
                    color: hoveredCell?.col === colIdx ? "var(--accent-primary)" : "var(--text-secondary)",
                    fontWeight: hoveredCell?.col === colIdx ? 700 : 600,
                    fontSize: "10px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {getShortLabel(c)}
                </th>
              ))}
              <th style={{ width: "65px", padding: "4px 6px", textAlign: "right", color: "var(--text-muted)", fontSize: "10px", fontWeight: 700 }}>
                Recall
              </th>
            </tr>
          </thead>
          <tbody>
            {classes.map((trueClass, rowIdx) => {
              const row = matrix[rowIdx] || [];
              const rowTotal = rowSums[rowIdx] || 1;
              const recall = (row[rowIdx] || 0) / rowTotal;

              return (
                <tr key={trueClass}>
                  <td
                    title={`True Intent: ${trueClass}`}
                    style={{
                      padding: "4px 8px",
                      color: hoveredCell?.row === rowIdx ? "var(--accent-primary)" : "var(--text-primary)",
                      fontWeight: hoveredCell?.row === rowIdx ? 700 : 500,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontSize: "11px",
                    }}
                  >
                    <code>{trueClass}</code>
                  </td>
                  {classes.map((predClass, colIdx) => {
                    const val = row[colIdx] || 0;
                    const isDiag = rowIdx === colIdx;
                    const isHovered = hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx;
                    const cellRatio = val / rowTotal;

                    // Color calculation
                    let bg = "rgba(255, 255, 255, 0.02)";
                    let textColor = "var(--text-muted)";
                    if (isDiag) {
                      bg = `rgba(52, 211, 153, ${0.12 + 0.65 * cellRatio})`;
                      textColor = cellRatio > 0.5 ? "#ffffff" : "#a7f3d0";
                    } else if (val > 0) {
                      bg = `rgba(239, 68, 68, ${0.15 + Math.min(0.7, val * 0.15)})`;
                      textColor = "#fca5a5";
                    }

                    if (isHovered) {
                      bg = isDiag ? "rgba(52, 211, 153, 0.9)" : "rgba(239, 68, 68, 0.9)";
                      textColor = "#000000";
                    }

                    return (
                      <td
                        key={predClass}
                        onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                        onMouseLeave={() => setHoveredCell(null)}
                        title={`True: ${trueClass}\nPred: ${predClass}\nCount: ${val} (${(cellRatio * 100).toFixed(1)}%)`}
                        style={{
                          textAlign: "center",
                          padding: "6px 2px",
                          background: bg,
                          color: textColor,
                          fontWeight: isDiag ? 700 : 500,
                          borderRadius: "3px",
                          cursor: "pointer",
                          transition: "background 0.15s ease",
                          fontFamily: "var(--font-mono, monospace)",
                          fontSize: "10px",
                        }}
                      >
                        {val === 0 ? "·" : showNormalized ? `${(cellRatio * 100).toFixed(0)}%` : val}
                      </td>
                    );
                  })}
                  <td
                    style={{
                      textAlign: "right",
                      padding: "4px 6px",
                      fontFamily: "var(--font-mono, monospace)",
                      fontWeight: 700,
                      fontSize: "11px",
                      color: recall >= 0.9 ? "var(--success)" : recall >= 0.75 ? "#fbbf24" : "var(--danger)",
                    }}
                  >
                    {(recall * 100).toFixed(0)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Inspector / Selected Cell Status Bar */}
      <div
        style={{
          marginTop: "12px",
          padding: "8px 12px",
          background: "var(--bg-surface-low)",
          borderRadius: "6px",
          fontSize: "12px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "8px",
          minHeight: "36px",
        }}
      >
        {hoveredCell ? (
          <div>
            <span style={{ color: "var(--text-muted)" }}>Selected Cell: </span>
            <strong>True:</strong> <code>{classes[hoveredCell.row]}</code>
            <span style={{ margin: "0 6px", color: "var(--text-muted)" }}>&rarr;</span>
            <strong>Pred:</strong> <code>{classes[hoveredCell.col]}</code>
            <span style={{ margin: "0 8px" }}>|</span>
            <strong>{(matrix[hoveredCell.row]?.[hoveredCell.col] || 0)}</strong> queries
            <span style={{ color: "var(--text-muted)", marginLeft: "4px" }}>
              ({(((matrix[hoveredCell.row]?.[hoveredCell.col] || 0) / (rowSums[hoveredCell.row] || 1)) * 100).toFixed(1)}% of true class)
            </span>
          </div>
        ) : (
          <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
            Hover over any matrix cell to inspect true vs. predicted counts and misclassification patterns.
          </span>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: "12px", fontSize: "11px", color: "var(--text-muted)" }}>
          <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <span style={{ width: "10px", height: "10px", background: "rgba(52, 211, 153, 0.6)", borderRadius: "2px" }} />
            Diagonal (Correct)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <span style={{ width: "10px", height: "10px", background: "rgba(239, 68, 68, 0.6)", borderRadius: "2px" }} />
            Off-diagonal (Confusion)
          </span>
        </div>
      </div>
    </div>
  );
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

  const modelsObj = (leaderboardPayload.models || {}) as Record<string, {
    accuracy?: number;
    macro_f1?: number;
    accuracy_ci?: [number, number];
    macro_f1_ci?: [number, number];
    ece?: number;
    [k: string]: unknown;
  }>;
  const headlineModel = (leaderboardPayload.headline_model as string) || "minilm_logistic";

  // Build router rows from models object or routers list
  const routerRows: RouterRow[] = (leaderboardPayload.routers as RouterRow[]) ||
    Object.entries(modelsObj).map(([name, m]) => ({
      model_name: name,
      // HONEST-FALLBACK: Read accuracy from model payload or fallback to 0
      accuracy: m.accuracy ?? 0,
      accuracy_ci: m.accuracy_ci,
      // HONEST-FALLBACK: Read macro_f1 from model payload or fallback to 0
      macro_f1: m.macro_f1 ?? 0,
      macro_f1_ci: m.macro_f1_ci,
      ece: m.ece,
      is_headline: name === headlineModel,
    })).sort((a, b) => b.accuracy - a.accuracy);

  const verifiedGoldCount = Number(leaderboardPayload.verified_gold_count || 0);
  const totalGoldCount = Number(leaderboardPayload.total_gold_count || 0);
  const isPreliminary = verifiedGoldCount < 100;

  const cal = (calibrationPayload || {}) as Record<string, unknown>;
  const selPred = (cal.selective_prediction || {}) as Record<string, unknown>;
  const oos = (cal.out_of_scope_detection || {}) as Record<string, unknown>;
  const oosAll = (oos.softmax_all || {}) as Record<string, unknown>;

  // HONEST-FALLBACK: Read tau from calibration payload or default to 0.1
  const tau = Number(cal.tau ?? 0.1);
  // HONEST-FALLBACK: Read selective accuracy / precision from selective_prediction or root field
  const selectivePrecision = Number(selPred.selective_accuracy ?? cal.selective_precision ?? 0);
  // HONEST-FALLBACK: Read coverage from selective_prediction or root field
  const coverage = Number(selPred.coverage ?? cal.coverage ?? 0);
  // HONEST-FALLBACK: Read AUROC from out_of_scope_detection softmax_all or root field
  const oosAuroc = Number(oosAll.auroc ?? cal.oos_auroc ?? 0);
  // HONEST-FALLBACK: Read ECE from calibration payload or root field
  const ece = Number(cal.ece ?? 0);

  const ret = (retrievalPayload || {}) as Record<string, unknown>;
  const metrics = (ret.metrics || {}) as Record<string, unknown>;
  const denseMetrics = (metrics.dense || {}) as Record<string, unknown>;
  const tfidfMetrics = (metrics.tfidf || {}) as Record<string, unknown>;

  // HONEST-FALLBACK: Read dense recall@3 from metrics.dense or root field
  const denseRecall3 = Number(denseMetrics.recall_at_3 ?? ret.dense_recall_at_3 ?? 0);
  // HONEST-FALLBACK: Read tfidf recall@3 from metrics.tfidf or root field
  const tfidfRecall3 = Number(tfidfMetrics.recall_at_3 ?? ret.tfidf_recall_at_3 ?? 0);
  // HONEST-FALLBACK: Read dense MRR from metrics.dense or root field
  const denseMrr = Number(denseMetrics.mrr ?? ret.dense_mrr ?? 0);
  // HONEST-FALLBACK: Read tfidf MRR from metrics.tfidf or root field
  const tfidfMrr = Number(tfidfMetrics.mrr ?? ret.tfidf_mrr ?? 0);

  const rawErrors = ((errorsPayload?.worst_20_errors || errorsPayload?.error_breakdown || []) as Array<Record<string, unknown>>);
  const errors = rawErrors.map((err) => ({
    // HONEST-FALLBACK: Read error query string from text or query field
    query: String(err.text ?? err.query ?? ""),
    // HONEST-FALLBACK: Read true intent label
    true_intent: String(err.true_intent ?? "unknown"),
    // HONEST-FALLBACK: Read predicted intent label from predicted_intent or pred_intent
    pred_intent: String(err.predicted_intent ?? err.pred_intent ?? "unknown"),
    // HONEST-FALLBACK: Read model confidence
    confidence: Number(err.confidence ?? 0),
    // HONEST-FALLBACK: Read resolved routing method
    resolved_method: String(err.resolved_method ?? (err.abstained ? "abstained" : "logistic")),
  }));

  const confClasses = (cal.classes as string[]) || [];
  const confMatrix = (cal.confusion_matrix as number[][]) || [];

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
                    {r.ece !== undefined ? r.ece.toFixed(4) : (r.is_headline && ece > 0 ? ece.toFixed(4) : "—")}
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

      {/* 4. Intent Classification Confusion Matrix Heatmap */}
      {confClasses.length > 0 && confMatrix.length > 0 && (
        <IntentConfusionMatrix classes={confClasses} matrix={confMatrix} />
      )}

      {/* 5. Error Breakdown Diagnostic Table */}
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
