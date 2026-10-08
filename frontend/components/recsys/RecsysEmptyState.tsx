"use client";

import React from "react";

interface RecsysEmptyStateProps {
  command?: string;
  title?: string;
  message?: string;
}

export function RecsysEmptyState({
  command = "make recsys-all",
  title = "Recommender Service Not Ready",
  message = "Trained recommender models or serving artifacts are not available in the database. Run the pipeline command below to train the candidate models, ranker, and item projections:",
}: RecsysEmptyStateProps) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
          backgroundColor: "var(--warning-bg)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 16px",
        }}
      >
        <span className="material-symbols-outlined" style={{ fontSize: "28px", color: "var(--warning)" }}>
          terminal
        </span>
      </div>

      <h3 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "8px" }}>
        {title}
      </h3>
      <p style={{ fontSize: "14px", color: "var(--text-secondary)", marginBottom: "20px", lineHeight: 1.6 }}>
        {message}
      </p>

      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "12px",
          backgroundColor: "var(--bg-surface-lowest)",
          border: "1px solid var(--border-subtle)",
          padding: "10px 16px",
          borderRadius: "var(--radius-sm)",
          fontFamily: "var(--font-mono, monospace)",
          fontSize: "14px",
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
          <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
            {copied ? "check" : "content_copy"}
          </span>
        </button>
      </div>
      {copied && (
        <div style={{ marginTop: "8px", fontSize: "12px", color: "var(--success)" }}>
          Copied to clipboard!
        </div>
      )}
    </div>
  );
}
