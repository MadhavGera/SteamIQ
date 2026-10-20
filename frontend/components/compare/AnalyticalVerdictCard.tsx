"use client";

import React from "react";
import Link from "next/link";
import type { VerdictData } from "./comparisonData";

interface AnalyticalVerdictCardProps {
  gameAName: string;
  gameBName: string;
  verdict: VerdictData;
}

export function AnalyticalVerdictCard({
  gameAName,
  gameBName,
  verdict,
}: AnalyticalVerdictCardProps) {
  return (
    <section
      style={{
        backgroundColor: "var(--bg-surface)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-card)",
        padding: "24px",
        marginBottom: "28px",
      }}
    >
      {/* Section Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px" }}>
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "22px", color: "var(--accent-primary)" }}
        >
          psychology
        </span>
        <div>
          <h2 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
            SteamIQ Analytical Verdict
          </h2>
          <p style={{ fontSize: "13px", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
            Data-backed player profiling, structural distinctions, and tailored recommendations.
          </p>
        </div>
      </div>

      {/* Choose A vs Choose B Split Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "16px",
          marginBottom: "20px",
        }}
      >
        {/* Choose Game A Card */}
        <div
          style={{
            backgroundColor: "var(--bg-surface-low)",
            border: "1px solid rgba(102, 192, 244, 0.25)",
            borderRadius: "var(--radius-sm)",
            padding: "18px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "12px",
              color: "var(--accent-primary)",
              fontWeight: 700,
              fontSize: "14px",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
              recommend
            </span>
            <span>Choose {gameAName} if you want:</span>
          </div>
          <ul style={{ margin: 0, paddingLeft: "18px", display: "flex", flexDirection: "column", gap: "8px" }}>
            {verdict.chooseA.map((point, i) => (
              <li key={i} style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                {point}
              </li>
            ))}
          </ul>
        </div>

        {/* Choose Game B Card */}
        <div
          style={{
            backgroundColor: "var(--bg-surface-low)",
            border: "1px solid rgba(16, 185, 129, 0.25)",
            borderRadius: "var(--radius-sm)",
            padding: "18px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "12px",
              color: "#10b981",
              fontWeight: 700,
              fontSize: "14px",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
              recommend
            </span>
            <span>Choose {gameBName} if you want:</span>
          </div>
          <ul style={{ margin: 0, paddingLeft: "18px", display: "flex", flexDirection: "column", gap: "8px" }}>
            {verdict.chooseB.map((point, i) => (
              <li key={i} style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                {point}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* The Core Distinction Highlight Box */}
      {verdict.keyDistinction && (
        <div
          style={{
            backgroundColor: "rgba(102, 192, 244, 0.06)",
            borderLeft: "3px solid var(--accent-primary)",
            padding: "14px 18px",
            borderRadius: "0 var(--radius-sm) var(--radius-sm) 0",
            marginBottom: "24px",
          }}
        >
          <div
            style={{
              fontSize: "12px",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: "var(--accent-primary)",
              marginBottom: "4px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
              lightbulb
            </span>
            Core Architectural Distinction
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-primary)", lineHeight: 1.6, margin: 0 }}>
            {verdict.keyDistinction}
          </p>
        </div>
      )}

      {/* Explore Deeper with SteamIQ AI Interactive Prompts */}
      {verdict.followUpPrompts && verdict.followUpPrompts.length > 0 && (
        <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "18px" }}>
          <div
            style={{
              fontSize: "12px",
              fontWeight: 700,
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              marginBottom: "10px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "var(--accent-primary)" }}>
              chat
            </span>
            Explore Deeper with SteamIQ Assistant:
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {verdict.followUpPrompts.map((prompt, idx) => (
              <Link
                key={idx}
                href={`/ask?q=${encodeURIComponent(prompt)}`}
                className="popular-chip"
                style={{
                  cursor: "pointer",
                  padding: "6px 14px",
                  fontSize: "12px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  backgroundColor: "var(--bg-surface-low)",
                  border: "1px solid var(--border-subtle)",
                  color: "var(--text-primary)",
                  borderRadius: "var(--radius-pill)",
                  textDecoration: "none",
                  transition: "all var(--transition-fast)",
                }}
              >
                <span
                  className="material-symbols-outlined"
                  style={{ fontSize: "14px", color: "var(--accent-primary)" }}
                >
                  arrow_forward
                </span>
                <span>{prompt}</span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
