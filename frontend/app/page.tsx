"use client";

import React, { useState, useEffect, useTransition } from "react";
import {
  api,
  RecsysModelInfo,
  RecsysCatalogItem,
  RecsysForYouResponse,
  RecsysHealthResponse,
} from "@/lib/api";
import { RecsysLibraryBuilder } from "@/components/recsys/RecsysLibraryBuilder";
import { RecsysModelSelector } from "@/components/recsys/RecsysModelSelector";
import { RecsysResultCard } from "@/components/recsys/RecsysResultCard";
import { RecsysCompareView } from "@/components/recsys/RecsysCompareView";
import { RecsysProvenanceFooter } from "@/components/recsys/RecsysProvenanceFooter";
import { RecsysEmptyState } from "@/components/recsys/RecsysEmptyState";

export default function ForYouPage() {
  const [health, setHealth] = useState<RecsysHealthResponse | null>(null);
  const [models, setModels] = useState<RecsysModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedGames, setSelectedGames] = useState<RecsysCatalogItem[]>([]);
  const [isCompareMode, setIsCompareMode] = useState<boolean>(false);
  const [recommendations, setRecommendations] = useState<RecsysForYouResponse | null>(null);

  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);
  const [isLoadingRecs, setIsLoadingRecs] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [, startTransition] = useTransition();

  // 1. Initial health & models check
  useEffect(() => {
    let isMounted = true;
    async function init() {
      try {
        const [healthRes, modelsRes] = await Promise.all([
          api.recsysHealth(),
          api.recsysModels(),
        ]);
        if (isMounted) {
          setHealth(healthRes);
          setModels(modelsRes);

          // Allowed models per user request: ItemKNN, ALS, Two Tower
          const ALLOWED_RECOMMENDER_MODELS = [
            "recsys_two_tower_clean",
            "recsys_two_tower",
            "recsys_itemknn",
            "recsys_als",
          ];
          const readyDeployables = modelsRes.filter(
            (m) => m.deployable && m.is_ready && ALLOWED_RECOMMENDER_MODELS.includes(m.model_name)
          );
          // Default to Two Tower
          const defaultModel =
            readyDeployables.find((m) => m.model_name.startsWith("recsys_two_tower")) ||
            readyDeployables[0];
          if (defaultModel) {
            setSelectedModel(defaultModel.model_name);
          }

          // Pre-fill 3 benchmark games (Elden Ring, Hollow Knight, Hades) for immediate model comparison
          try {
            const trioIds = [1245620, 367520, 1145360];
            const prefilled: RecsysCatalogItem[] = [];
            for (const id of trioIds) {
              const searchRes = await api.searchRecsysItems(String(id), 1);
              const match = searchRes.items.find((it) => it.app_id === id);
              if (match) {
                prefilled.push(match);
              }
            }
            if (prefilled.length > 0) {
              setSelectedGames(prefilled);
            }
          } catch {
            // Ignore prefill errors
          }

          setIsLoadingHealth(false);
        }
      } catch {
        if (isMounted) {
          setIsLoadingHealth(false);
        }
      }
    }
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  // 2. Fetch personalized recommendations whenever library or single model changes
  useEffect(() => {
    if (isCompareMode) return;
    if (selectedGames.length === 0 || !selectedModel) {
      setRecommendations(null);
      return;
    }

    let isMounted = true;
    setIsLoadingRecs(true);
    setError(null);

    const likedIds = selectedGames.map((g) => g.app_id);

    api
      .forYou({
        liked_app_ids: likedIds,
        model: selectedModel,
        k: 10,
        explain: true,
      })
      .then((res) => {
        if (isMounted) {
          startTransition(() => {
            setRecommendations(res);
            setIsLoadingRecs(false);
          });
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to fetch recommendations");
          setIsLoadingRecs(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedGames, selectedModel, isCompareMode]);

  // Loading Initial State Skeleton
  if (isLoadingHealth) {
    return (
      <main className="container" style={{ paddingTop: "88px", paddingBottom: "64px" }}>
        <div className="skeleton" style={{ height: "40px", width: "320px", marginBottom: "16px" }} />
        <div className="skeleton" style={{ height: "20px", width: "480px", marginBottom: "32px" }} />
        <div className="skeleton" style={{ height: "200px", width: "100%", marginBottom: "24px" }} />
      </main>
    );
  }

  // If recommender health reports not ready or no deployable models are available
  const hasUsableModel = health?.is_ready && models.some((m) => m.deployable && m.is_ready);
  if (!hasUsableModel) {
    return (
      <main className="container" style={{ paddingTop: "88px", paddingBottom: "64px" }}>
        <RecsysEmptyState
          command="make recsys-all"
          title="Recommender Pipeline Not Materialized"
          message="The cold-start neural recommender system requires trained model artifacts and database item vectors. Execute the pipeline command below to train all candidate models, evaluate offline benchmarks, and project item embeddings:"
        />
      </main>
    );
  }

  // HONEST-FALLBACK: Read catalog universe size from health or recommendations response
  const universeSize = health?.universe_size ?? recommendations?.universe_size ?? null;

  return (
    <main className="container" style={{ paddingTop: "88px", paddingBottom: "64px" }}>
      {/* Page Hero & Mode Switch */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: "16px",
          marginBottom: "28px",
          borderBottom: "1px solid var(--border-subtle)",
          paddingBottom: "20px",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
            <span className="badge-pill badge-pill--neutral" style={{ color: "var(--accent-primary)" }}>
              Offline-Trained ML Core
            </span>
            {universeSize !== null && (
              <span className="badge-pill badge-pill--neutral" style={{ fontFamily: "var(--font-mono, monospace)" }}>
                Catalog: {universeSize.toLocaleString()} games
              </span>
            )}
          </div>
          <h1 style={{ fontSize: "28px", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
            For You: Personalized Game Recommendations
          </h1>
          <p style={{ fontSize: "14px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Build your seed library to test cold-start inference across candidate recommender models with live explainability.
          </p>
        </div>

        {/* Mode Toggle: Single Model vs Compare */}
        <div
          style={{
            display: "inline-flex",
            backgroundColor: "var(--bg-surface-lowest)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-sm)",
            padding: "3px",
          }}
        >
          <button
            type="button"
            onClick={() => setIsCompareMode(false)}
            style={{
              padding: "6px 14px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "13px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: !isCompareMode ? "var(--accent-primary)" : "transparent",
              color: !isCompareMode ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
              transition: "all var(--transition-fast)",
            }}
          >
            Single Model
          </button>
          <button
            type="button"
            onClick={() => setIsCompareMode(true)}
            style={{
              padding: "6px 14px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              fontSize: "13px",
              fontWeight: 600,
              cursor: "pointer",
              backgroundColor: isCompareMode ? "var(--accent-primary)" : "transparent",
              color: isCompareMode ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
              transition: "all var(--transition-fast)",
            }}
          >
            Compare Models
          </button>
        </div>
      </div>

      {/* 1. Interactive Library Builder */}
      <RecsysLibraryBuilder
        selectedGames={selectedGames}
        onChange={setSelectedGames}
        universeSize={universeSize}
      />

      {/* 2. Server Warnings & Unknown IDs Alert Banner */}
      {recommendations && (recommendations.warning || (recommendations.unknown_app_ids && recommendations.unknown_app_ids.length > 0)) && (
        <div
          style={{
            padding: "12px 16px",
            backgroundColor: "var(--warning-bg)",
            border: "1px solid rgba(199, 178, 74, 0.4)",
            borderRadius: "var(--radius-sm)",
            color: "var(--warning)",
            fontSize: "13px",
            marginBottom: "20px",
            display: "flex",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            warning
          </span>
          <div>
            {recommendations.warning && <div>{recommendations.warning}</div>}
            {recommendations.unknown_app_ids && recommendations.unknown_app_ids.length > 0 && (
              <div style={{ fontSize: "12px", marginTop: "2px" }}>
                Unmodeled App IDs omitted from cold-start inference:{" "}
                <span style={{ fontFamily: "var(--font-mono, monospace)" }}>
                  {recommendations.unknown_app_ids.join(", ")}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 3. Compare View Mode */}
      {isCompareMode ? (
        <RecsysCompareView
          models={models}
          selectedGames={selectedGames}
          defaultModelA={selectedModel}
        />
      ) : (
        /* 4. Single Model View Mode */
        <div>
          {/* Segmented Model Selector */}
          <RecsysModelSelector
            models={models}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
            disabled={isLoadingRecs}
          />

          {error && (
            <div
              style={{
                padding: "16px",
                backgroundColor: "var(--danger-bg)",
                color: "var(--danger)",
                borderRadius: "var(--radius-sm)",
                marginBottom: "20px",
              }}
            >
              {error}
            </div>
          )}

          {/* Loading Indicator */}
          {isLoadingRecs && (
            <div style={{ padding: "48px 0", textAlign: "center" }}>
              <div className="loading-bars" style={{ margin: "0 auto 12px" }}>
                <span />
                <span />
                <span />
              </div>
              <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                Evaluating candidate rank scores...
              </p>
            </div>
          )}

          {/* Recommendations Grid */}
          {!isLoadingRecs && recommendations && (
            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "16px",
                }}
              >
                <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
                  Top {recommendations.items.length} Recommendations
                </h3>
                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                  Ranked by {recommendations.model_name}
                </span>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
                  gap: "16px",
                }}
              >
                {recommendations.items.map((item) => (
                  <RecsysResultCard
                    key={item.app_id}
                    item={item}
                    modelName={recommendations.model_name}
                  />
                ))}
              </div>

              {/* Provenance Footer */}
              <RecsysProvenanceFooter
                modelName={recommendations.model_name}
                modelRunId={recommendations.model_run_id}
                datasetVersion={recommendations.dataset_version}
                universeSize={recommendations.universe_size}
                tookMs={recommendations.took_ms}
              />
            </div>
          )}

          {/* Prompt when 0 games selected */}
          {!isLoadingRecs && selectedGames.length === 0 && (
            <div
              className="card"
              style={{
                padding: "48px 24px",
                textAlign: "center",
                backgroundColor: "var(--bg-surface-low)",
                border: "1px dashed var(--border-subtle)",
              }}
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "36px", color: "var(--text-muted)", marginBottom: "12px" }}
              >
                view_in_ar
              </span>
              <h3 style={{ fontSize: "16px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "6px" }}>
                Your library is empty
              </h3>
              <p style={{ fontSize: "13px", color: "var(--text-secondary)", maxWidth: "440px", margin: "0 auto" }}>
                Search for games you love or select one of the Sample Libraries above to generate personalized recommendations.
              </p>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
