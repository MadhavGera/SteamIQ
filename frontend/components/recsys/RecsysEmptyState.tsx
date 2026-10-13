"use client";

import React from "react";

import { api } from "@/lib/api";

interface RecsysEmptyStateProps {
  command?: string;
  title?: string;
  message?: string;
  onSuccess?: () => void;
}

export function RecsysEmptyState({
  command = "make recsys-all",
  title = "Recommender Service Not Ready",
  message = "Trained recommender models or serving artifacts are not available in the database. You can materialize them directly with the button below or copy the terminal command:",
  onSuccess,
}: RecsysEmptyStateProps) {
  const [copied, setCopied] = React.useState(false);
  const [isRunning, setIsRunning] = React.useState(false);
  const [runSuccess, setRunSuccess] = React.useState(false);
  const [runError, setRunError] = React.useState<string | null>(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRun = async () => {
    setIsRunning(true);
    setRunError(null);
    try {
      await api.materializeReports();
      setRunSuccess(true);
      setTimeout(() => {
        if (onSuccess) {
          onSuccess();
        } else {
          window.location.reload();
        }
      }, 1200);
    } catch (err) {
      setRunError(err instanceof Error ? err.message : "Failed to synchronize benchmarks");
      setIsRunning(false);
    }
  };

  return (
    <div
      className="card"
      style={{
        padding: "36px 24px",
        textAlign: "center",
        maxWidth: "680px",
        margin: "40px auto",
        backgroundColor: "var(--bg-surface-low)",
        border: "1px solid var(--border-strong)",
      }}
    >
      <div
        style={{
          width: "48px",
          height: "48px",
          borderRadius: "50%",
          backgroundColor: runSuccess ? "rgba(34, 197, 94, 0.15)" : "var(--warning-bg)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 16px",
        }}
      >
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "28px", color: runSuccess ? "var(--success, #22c55e)" : "var(--warning)" }}
        >
          {runSuccess ? "check_circle" : "smart_toy"}
        </span>
      </div>

      <h3 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "8px" }}>
        {title}
      </h3>
      <p style={{ fontSize: "14px", color: "var(--text-secondary)", marginBottom: "24px", lineHeight: 1.6 }}>
        {message}
      </p>

      {/* Primary Action: Direct One-Click Button */}
      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={handleRun}
          disabled={isRunning || runSuccess}
          className="btn btn--primary"
          style={{
            padding: "10px 24px",
            fontSize: "14px",
            fontWeight: "700",
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            background: runSuccess
              ? "var(--success, #22c55e)"
              : "linear-gradient(135deg, var(--accent-primary) 0%, #38bdf8 100%)",
            color: "var(--bg-surface-lowest)",
            border: "none",
            borderRadius: "8px",
            cursor: isRunning || runSuccess ? "default" : "pointer",
            boxShadow: "0 4px 14px rgba(0, 0, 0, 0.25)",
          }}
        >
          <span
            className="material-symbols-outlined"
            style={{
              fontSize: "18px",
              animation: isRunning ? "spin 1s linear infinite" : "none",
            }}
          >
            {isRunning ? "refresh" : runSuccess ? "check" : "bolt"}
          </span>
          {isRunning
            ? "Materializing Benchmarks..."
            : runSuccess
            ? "Materialized! Refreshing..."
            : "Compute / Materialize Benchmarks Now"}
        </button>
      </div>

      {runError && (
        <div style={{ marginBottom: "16px", fontSize: "13px", color: "#ef4444" }}>
          Failed: {runError}
        </div>
      )}

      {/* Secondary Alternative: Terminal Command */}
      <div style={{ marginTop: "12px", borderTop: "1px solid var(--border-subtle)", paddingTop: "16px" }}>
        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "8px" }}>
          Or run from terminal:
        </div>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "12px",
            backgroundColor: "var(--bg-surface-lowest)",
            border: "1px solid var(--border-subtle)",
            padding: "8px 14px",
            borderRadius: "var(--radius-sm)",
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "13px",
            color: "var(--accent-primary)",
          }}
        >
          <span>$ {command}</span>
          <button
            onClick={handleCopy}
            aria-label="Copy shell command"
            style={{
              background: "transparent",
              border: "none",
              color: copied ? "var(--success)" : "var(--text-muted)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              padding: "2px",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
              {copied ? "check" : "content_copy"}
            </span>
          </button>
        </div>
        {copied && (
          <div style={{ marginTop: "6px", fontSize: "11px", color: "var(--success)" }}>
            Copied to clipboard!
          </div>
        )}
      </div>
    </div>
  );
}
