"use client";

import React from "react";

interface RecsysProvenanceFooterProps {
  modelName: string;
  modelRunId: string | null;
  datasetVersion: string;
  universeSize?: number;
  tookMs: number;
}

export function RecsysProvenanceFooter({
  modelName,
  modelRunId,
  datasetVersion,
  universeSize,
  tookMs,
}: RecsysProvenanceFooterProps) {
  // Shorten run ID for display (e.g. 8 chars)
  const shortRunId = modelRunId ? modelRunId.slice(0, 8) : "local";

  return (
    <footer
      className="recsys-provenance-footer"
      style={{
        marginTop: "24px",
        padding: "12px 16px",
        backgroundColor: "var(--bg-surface-lowest)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-sm)",
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "12px",
        fontSize: "12px",
        color: "var(--text-muted)",
        fontFamily: "var(--font-mono, monospace)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", color: "var(--text-secondary)" }}>
          <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "var(--accent-primary)" }}>
            memory
          </span>
          Model: <strong style={{ color: "var(--text-primary)" }}>{modelName}</strong>
        </span>
        <span>•</span>
        <span>Run: <strong style={{ color: "var(--text-secondary)" }}>{shortRunId}</strong></span>
        <span>•</span>
        <span>Dataset: <strong style={{ color: "var(--text-secondary)" }}>{datasetVersion}</strong></span>
        {universeSize !== undefined && (
          <>
            <span>•</span>
            <span>Catalog: <strong style={{ color: "var(--text-secondary)" }}>{universeSize.toLocaleString()} games</strong></span>
          </>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
        <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "var(--success)" }}>
          timer
        </span>
        <span>Serving latency: <strong style={{ color: "var(--text-primary)" }}>{tookMs.toFixed(1)} ms</strong></span>
      </div>
    </footer>
  );
}
