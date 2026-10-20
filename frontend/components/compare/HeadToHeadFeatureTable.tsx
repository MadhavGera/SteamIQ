"use client";

import React from "react";
import type { FeatureRow } from "./comparisonData";

interface HeadToHeadFeatureTableProps {
  gameAName: string;
  gameBName: string;
  features: FeatureRow[];
}

export function HeadToHeadFeatureTable({
  gameAName,
  gameBName,
  features,
}: HeadToHeadFeatureTableProps) {
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
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "18px" }}>
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "22px", color: "var(--accent-primary)" }}
        >
          table_chart
        </span>
        <div>
          <h2 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
            Head-to-Head Feature Comparison
          </h2>
          <p style={{ fontSize: "13px", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
            Comprehensive architectural and gameplay breakdown between {gameAName} and {gameBName}.
          </p>
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "separate",
            borderSpacing: "0",
            fontSize: "13px",
            textAlign: "left",
          }}
        >
          <thead>
            <tr style={{ backgroundColor: "var(--bg-surface-low)" }}>
              <th
                style={{
                  padding: "12px 16px",
                  color: "var(--text-muted)",
                  fontWeight: 700,
                  fontSize: "11px",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  borderBottom: "1px solid var(--border-subtle)",
                  width: "22%",
                }}
              >
                Feature
              </th>
              <th
                style={{
                  padding: "12px 16px",
                  color: "var(--accent-primary)",
                  fontWeight: 700,
                  fontSize: "13px",
                  borderBottom: "1px solid var(--border-subtle)",
                  width: "39%",
                }}
              >
                {gameAName}
              </th>
              <th
                style={{
                  padding: "12px 16px",
                  color: "#10b981",
                  fontWeight: 700,
                  fontSize: "13px",
                  borderBottom: "1px solid var(--border-subtle)",
                  width: "39%",
                }}
              >
                {gameBName}
              </th>
            </tr>
          </thead>
          <tbody>
            {features.map((row, idx) => {
              const isEven = idx % 2 === 0;
              return (
                <tr
                  key={row.label}
                  style={{
                    backgroundColor: isEven ? "rgba(0, 0, 0, 0.15)" : "transparent",
                    transition: "background-color var(--transition-fast)",
                  }}
                >
                  <td
                    style={{
                      padding: "12px 16px",
                      color: "var(--text-primary)",
                      fontWeight: 600,
                      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                      verticalAlign: "top",
                    }}
                  >
                    {row.label}
                  </td>
                  <td
                    style={{
                      padding: "12px 16px",
                      color: "var(--text-secondary)",
                      lineHeight: 1.5,
                      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                      verticalAlign: "top",
                    }}
                  >
                    {row.gameA}
                  </td>
                  <td
                    style={{
                      padding: "12px 16px",
                      color: "var(--text-secondary)",
                      lineHeight: 1.5,
                      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                      verticalAlign: "top",
                    }}
                  >
                    {row.gameB}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
