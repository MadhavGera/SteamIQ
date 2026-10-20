"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { api, RecsysModelInfo, RecsysCatalogItem, RecsysForYouResponse } from "@/lib/api";
import { RecsysProvenanceFooter } from "./RecsysProvenanceFooter";

interface RecsysCompareViewProps {
  models: RecsysModelInfo[];
  selectedGames: RecsysCatalogItem[];
  defaultModelA?: string;
  defaultModelB?: string;
}

export function RecsysCompareView({
  models,
  selectedGames,
  defaultModelA = "recsys_two_tower_clean",
  defaultModelB = "recsys_itemknn",
}: RecsysCompareViewProps) {
  // Allowed models in Recommendation Engine per user request: ItemKNN, ALS, Two Tower
  const ALLOWED_RECOMMENDER_MODELS = [
    "recsys_two_tower_clean",
    "recsys_two_tower",
    "recsys_itemknn",
    "recsys_als",
  ];
  const deployableModels = models.filter(
    (m) => m.deployable && m.is_ready && ALLOWED_RECOMMENDER_MODELS.includes(m.model_name)
  );

  const getCleanDisplayName = (m: RecsysModelInfo) => {
    if (m.model_name === "recsys_itemknn") return "ItemKNN";
    if (m.model_name === "recsys_als") return "ALS";
    if (m.model_name === "recsys_two_tower_clean" || m.model_name === "recsys_two_tower") return "Two Tower";
    return m.display_name;
  };

  // Pick initial models that are deployable
  // HONEST-FALLBACK: Default to first deployable model if specified default is unavailable
  const initialA = deployableModels.some((m) => m.model_name === defaultModelA)
    ? defaultModelA
    // HONEST-FALLBACK: Fallback model name from first deployable option
    : (deployableModels[0]?.model_name ?? "");

  // HONEST-FALLBACK: Default to second deployable model if available
  const initialB = deployableModels.some((m) => m.model_name === defaultModelB)
    ? defaultModelB
    // HONEST-FALLBACK: Fallback to second model or first model if only one exists
    : (deployableModels[1]?.model_name ?? deployableModels[0]?.model_name ?? "");

  const [modelA, setModelA] = useState(initialA);
  const [modelB, setModelB] = useState(initialB);

  const [resA, setResA] = useState<RecsysForYouResponse | null>(null);
  const [resB, setResB] = useState<RecsysForYouResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const likedAppIdsKey = selectedGames.map((g) => g.app_id).join(",");

  useEffect(() => {
    const likedAppIds = selectedGames.map((g) => g.app_id);
    if (likedAppIds.length === 0 || !modelA || !modelB) {
      setResA(null);
      setResB(null);
      return;
    }

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    Promise.all([
      api.forYou({ liked_app_ids: likedAppIds, model: modelA, k: 10, explain: true }),
      api.forYou({ liked_app_ids: likedAppIds, model: modelB, k: 10, explain: true }),
    ])
      .then(([a, b]) => {
        if (isMounted) {
          setResA(a);
          setResB(b);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load model comparison");
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [likedAppIdsKey, modelA, modelB, selectedGames]);

  // Compute Jaccard Overlap Index
  const appIdsA = new Set(resA?.items.map((it) => it.app_id) || []);
  const appIdsB = new Set(resB?.items.map((it) => it.app_id) || []);

  const arrA = Array.from(appIdsA);
  const arrB = Array.from(appIdsB);
  const intersection = new Set(arrA.filter((id) => appIdsB.has(id)));
  const union = new Set(arrA.concat(arrB));
  const jaccardPct = union.size > 0 ? (intersection.size / union.size) * 100 : 0;

  return (
    <div style={{ marginTop: "16px" }}>
      {/* Model Selection Header */}
      <div
        className="card"
        style={{
          padding: "16px 20px",
          marginBottom: "20px",
          backgroundColor: "var(--bg-surface-low)",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
            {/* Model A Selector */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <label style={{ fontSize: "13px", fontWeight: 700, color: "var(--accent-primary)" }}>
                Model A:
              </label>
              <select
                value={modelA}
                onChange={(e) => setModelA(e.target.value)}
                style={{
                  padding: "6px 12px",
                  borderRadius: "var(--radius-sm)",
                  backgroundColor: "var(--bg-surface)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border-strong)",
                  fontSize: "13px",
                  fontWeight: 600,
                }}
              >
                {deployableModels.map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {getCleanDisplayName(m)}
                  </option>
                ))}
              </select>
            </div>

            <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>VS</span>

            {/* Model B Selector */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <label style={{ fontSize: "13px", fontWeight: 700, color: "var(--warning)" }}>
                Model B:
              </label>
              <select
                value={modelB}
                onChange={(e) => setModelB(e.target.value)}
                style={{
                  padding: "6px 12px",
                  borderRadius: "var(--radius-sm)",
                  backgroundColor: "var(--bg-surface)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border-strong)",
                  fontSize: "13px",
                  fontWeight: 600,
                }}
              >
                {deployableModels.map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {getCleanDisplayName(m)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Jaccard Overlap Metric Pill */}
          {resA && resB && (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "10px",
                padding: "6px 14px",
                backgroundColor: "var(--bg-surface-raised)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-pill)",
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "var(--accent-light)" }}>
                grain
              </span>
              <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                Jaccard Overlap:{" "}
                <strong style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono, monospace)" }}>
                  {jaccardPct.toFixed(1)}%
                </strong>{" "}
                ({intersection.size} shared / {union.size} distinct)
              </span>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div style={{ padding: "16px", backgroundColor: "var(--danger-bg)", color: "var(--danger)", borderRadius: "var(--radius-sm)", marginBottom: "20px" }}>
          {error}
        </div>
      )}

      {isLoading && (
        <div style={{ padding: "40px 0", textAlign: "center" }}>
          <div className="loading-bars" style={{ margin: "0 auto 12px" }}>
            <span />
            <span />
            <span />
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>Computing dual model recommendations...</p>
        </div>
      )}

      {/* Side-by-side Top-10 Lists */}
      {!isLoading && resA && resB && (
        <div>
          <div className="panel-grid-50-50" style={{ alignItems: "start" }}>
            {/* Column A */}
            <div>
              <div
                style={{
                  padding: "10px 14px",
                  backgroundColor: "var(--bg-surface-raised)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  marginBottom: "12px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <h4 style={{ fontSize: "14px", fontWeight: 700, color: "var(--accent-primary)" }}>
                  {resA.model_name}
                </h4>
                <span style={{ fontSize: "11px", fontFamily: "var(--font-mono, monospace)", color: "var(--text-muted)" }}>
                  {resA.took_ms.toFixed(1)} ms
                </span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {resA.items.map((item) => {
                  const isShared = appIdsB.has(item.app_id);
                  return (
                    <div
                      key={item.app_id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "10px 12px",
                        backgroundColor: isShared ? "var(--bg-surface-low)" : "var(--bg-surface)",
                        border: isShared ? "1px solid var(--accent-primary)" : "1px solid var(--border-subtle)",
                        borderRadius: "var(--radius-sm)",
                        gap: "10px",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", minWidth: 0 }}>
                        <span
                          style={{
                            fontFamily: "var(--font-mono, monospace)",
                            fontWeight: 700,
                            fontSize: "12px",
                            color: "var(--text-muted)",
                            width: "24px",
                          }}
                        >
                          #{item.rank}
                        </span>
                        <div style={{ minWidth: 0 }}>
                          <Link
                            href={`/games/${item.app_id}?tab=overview`}
                            style={{
                              fontSize: "13px",
                              fontWeight: 600,
                              color: "var(--text-primary)",
                              display: "block",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {item.title}
                          </Link>
                          <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono, monospace)" }}>
                            Score: {item.score.toFixed(3)}
                          </span>
                        </div>
                      </div>

                      {isShared && (
                        <span className="badge-pill badge-pill--success" style={{ fontSize: "9px" }}>
                          Shared
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>

              <RecsysProvenanceFooter
                modelName={resA.model_name}
                modelRunId={resA.model_run_id}
                datasetVersion={resA.dataset_version}
                tookMs={resA.took_ms}
              />
            </div>

            {/* Column B */}
            <div>
              <div
                style={{
                  padding: "10px 14px",
                  backgroundColor: "var(--bg-surface-raised)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  marginBottom: "12px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <h4 style={{ fontSize: "14px", fontWeight: 700, color: "var(--warning)" }}>
                  {resB.model_name}
                </h4>
                <span style={{ fontSize: "11px", fontFamily: "var(--font-mono, monospace)", color: "var(--text-muted)" }}>
                  {resB.took_ms.toFixed(1)} ms
                </span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {resB.items.map((item) => {
                  const isShared = appIdsA.has(item.app_id);
                  return (
                    <div
                      key={item.app_id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "10px 12px",
                        backgroundColor: isShared ? "var(--bg-surface-low)" : "var(--bg-surface)",
                        border: isShared ? "1px solid var(--warning)" : "1px solid var(--border-subtle)",
                        borderRadius: "var(--radius-sm)",
                        gap: "10px",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", minWidth: 0 }}>
                        <span
                          style={{
                            fontFamily: "var(--font-mono, monospace)",
                            fontWeight: 700,
                            fontSize: "12px",
                            color: "var(--text-muted)",
                            width: "24px",
                          }}
                        >
                          #{item.rank}
                        </span>
                        <div style={{ minWidth: 0 }}>
                          <Link
                            href={`/games/${item.app_id}?tab=overview`}
                            style={{
                              fontSize: "13px",
                              fontWeight: 600,
                              color: "var(--text-primary)",
                              display: "block",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {item.title}
                          </Link>
                          <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono, monospace)" }}>
                            Score: {item.score.toFixed(3)}
                          </span>
                        </div>
                      </div>

                      {isShared && (
                        <span className="badge-pill badge-pill--success" style={{ fontSize: "9px" }}>
                          Shared
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>

              <RecsysProvenanceFooter
                modelName={resB.model_name}
                modelRunId={resB.model_run_id}
                datasetVersion={resB.dataset_version}
                tookMs={resB.took_ms}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
