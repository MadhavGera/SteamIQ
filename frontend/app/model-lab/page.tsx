"use client";

import React, { useState, useEffect } from "react";
import { api, RecsysReport } from "@/lib/api";
import { LeaderboardTab } from "@/components/model-lab/LeaderboardTab";
import { PairedTestsTab } from "@/components/model-lab/PairedTestsTab";
import { AblationsTab } from "@/components/model-lab/AblationsTab";
import { ColdStartTab } from "@/components/model-lab/ColdStartTab";
import { BeyondAccuracyTab } from "@/components/model-lab/BeyondAccuracyTab";
import { TrainingCurvesTab } from "@/components/model-lab/TrainingCurvesTab";
import { ExplainabilityTab } from "@/components/model-lab/ExplainabilityTab";
import { DataMethodTab } from "@/components/model-lab/DataMethodTab";
import { AssistantTab } from "@/components/model-lab/AssistantTab";

type TabKey =
  | "leaderboard"
  | "paired"
  | "ablations"
  | "cold_start"
  | "beyond_accuracy"
  | "curves"
  | "explainability"
  | "data_method"
  | "assistant";

const REPORT_KEYS = [
  "leaderboard",
  "slices",
  "ablations",
  "ranker_ablations",
  "cold_start",
  "beyond_accuracy",
  "training_curves",
  "shap_global",
  "dataset",
  "limitations",
  "assistant_leaderboard",
  "assistant_calibration",
  "assistant_retrieval",
  "assistant_errors",
];

export default function ModelLabPage() {
  const [activeTab, setActiveTab] = useState<TabKey>("leaderboard");
  const [reports, setReports] = useState<Record<string, RecsysReport>>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;

    async function loadReports() {
      const fetched: Record<string, RecsysReport> = {};

      await Promise.all(
        REPORT_KEYS.map(async (key) => {
          try {
            const rep = await api.recsysReport(key);
            fetched[key] = rep;
          } catch {
            // Gracefully ignore missing reports; tabs will render their respective empty states
          }
        })
      );

      if (isMounted) {
        setReports(fetched);
        setIsLoading(false);
      }
    }

    loadReports();
    return () => {
      isMounted = false;
    };
  }, []);

  const TABS: Array<{ key: TabKey; label: string; icon: string }> = [
    { key: "leaderboard", label: "Leaderboard", icon: "leaderboard" },
    { key: "paired", label: "Paired Tests", icon: "compare_arrows" },
    { key: "ablations", label: "Ablations", icon: "tune" },
    { key: "cold_start", label: "Cold-Start & Slices", icon: "ac_unit" },
    { key: "beyond_accuracy", label: "Beyond Accuracy", icon: "insights" },
    { key: "curves", label: "Training Curves", icon: "show_chart" },
    { key: "explainability", label: "Explainability", icon: "psychology" },
    { key: "data_method", label: "Data & Method", icon: "description" },
    { key: "assistant", label: "Assistant", icon: "chat_spark" },
  ];

  // HONEST-FALLBACK: Read dataset version from leaderboard report or fallback string
  const datasetVer = reports["leaderboard"]?.dataset_version ?? "frozen_catalog";

  return (
    <main className="container" style={{ paddingTop: "88px", paddingBottom: "64px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: "16px",
          marginBottom: "24px",
          borderBottom: "1px solid var(--border-subtle)",
          paddingBottom: "16px",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
            <span className="badge-pill badge-pill--neutral" style={{ color: "var(--accent-primary)" }}>
              Model Evaluation & Benchmarking Center
            </span>
            <span className="badge-pill badge-pill--neutral" style={{ fontFamily: "var(--font-mono, monospace)" }}>
              Dataset: {datasetVer}
            </span>
          </div>
          <h1 style={{ fontSize: "28px", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
            Model Lab
          </h1>
          <p style={{ fontSize: "14px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Comprehensive offline evaluation metrics, paired statistical tests, fairness diagnostics, and TreeSHAP explainability.
            Headline production champion: <strong style={{ color: "var(--accent-primary)" }}>recsys_two_tower_clean</strong>. With train-only item stats there is no evidence of gain (CI includes 0); the all-time-stats gain is leakage-confounded.
          </p>
        </div>
      </div>

      {/* Tabs Navigation Bar */}
      <div
        style={{
          display: "flex",
          gap: "8px",
          overflowX: "auto",
          borderBottom: "1px solid var(--border-subtle)",
          marginBottom: "24px",
          paddingBottom: "8px",
        }}
      >
        {TABS.map((tab) => {
          const isActive = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 16px",
                borderRadius: "var(--radius-sm)",
                border: "none",
                backgroundColor: isActive ? "var(--bg-surface-raised)" : "transparent",
                color: isActive ? "var(--accent-primary)" : "var(--text-secondary)",
                fontSize: "13px",
                fontWeight: isActive ? 700 : 500,
                cursor: "pointer",
                whiteSpace: "nowrap",
                transition: "all var(--transition-fast)",
                borderBottom: isActive ? "2px solid var(--accent-primary)" : "2px solid transparent",
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                {tab.icon}
              </span>
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Loading Skeleton */}
      {isLoading ? (
        <div style={{ padding: "40px 0", textAlign: "center" }}>
          <div className="loading-bars" style={{ margin: "0 auto 12px" }}>
            <span />
            <span />
            <span />
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>Loading evaluation reports...</p>
        </div>
      ) : (
        /* Tab Content Views */
        <div>
          {activeTab === "leaderboard" && (
            <LeaderboardTab payload={reports["leaderboard"]?.payload || null} />
          )}

          {activeTab === "paired" && (
            <PairedTestsTab payload={reports["slices"]?.payload || null} />
          )}

          {activeTab === "ablations" && (
            <AblationsTab
              ttPayload={reports["ablations"]?.payload || null}
              rankerPayload={reports["ranker_ablations"]?.payload || null}
            />
          )}

          {activeTab === "cold_start" && (
            <ColdStartTab payload={reports["cold_start"]?.payload || null} />
          )}

          {activeTab === "beyond_accuracy" && (
            <BeyondAccuracyTab payload={reports["beyond_accuracy"]?.payload || null} />
          )}

          {activeTab === "curves" && (
            <TrainingCurvesTab payload={reports["training_curves"]?.payload || null} />
          )}

          {activeTab === "explainability" && (
            <ExplainabilityTab payload={reports["shap_global"]?.payload || null} />
          )}

          {activeTab === "data_method" && (
            <DataMethodTab
              datasetPayload={reports["dataset"]?.payload || null}
              limitationsPayload={reports["limitations"]?.payload || null}
            />
          )}

          {activeTab === "assistant" && (
            <AssistantTab
              leaderboardPayload={reports["assistant_leaderboard"]?.payload || null}
              calibrationPayload={reports["assistant_calibration"]?.payload || null}
              retrievalPayload={reports["assistant_retrieval"]?.payload || null}
              errorsPayload={reports["assistant_errors"]?.payload || null}
            />
          )}
        </div>
      )}
    </main>
  );
}
