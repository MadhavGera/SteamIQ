"use client";

import React from "react";
import type { ScorecardItem } from "./comparisonData";

interface ExperienceScorecardProps {
  gameAName: string;
  gameBName: string;
  scorecard: ScorecardItem[];
}

export function ExperienceScorecard({
  gameAName,
  gameBName,
  scorecard,
}: ExperienceScorecardProps) {
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
          style={{ fontSize: "22px", color: "var(--warning)" }}
        >
          military_tech
        </span>
        <div>
          <h2 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
            Which Has the Better Experience?
          </h2>
          <p style={{ fontSize: "13px", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
            Category-by-category comparative advantage between {gameAName} and {gameBName}.
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
                  width: "28%",
                }}
              >
                Category
              </th>
              <th
                style={{
                  padding: "12px 16px",
                  color: "var(--text-muted)",
                  fontWeight: 700,
                  fontSize: "11px",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  borderBottom: "1px solid var(--border-subtle)",
                  width: "28%",
                }}
              >
                Advantage / Winner
              </th>
              <th
                style={{
                  padding: "12px 16px",
                  color: "var(--text-muted)",
                  fontWeight: 700,
                  fontSize: "11px",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  borderBottom: "1px solid var(--border-subtle)",
                  width: "44%",
                }}
              >
                Why / Context
              </th>
            </tr>
          </thead>
          <tbody>
            {scorecard.map((item, idx) => {
              const isEven = idx % 2 === 0;

              // Determine badge style
              let badgeBg = "rgba(102, 192, 244, 0.12)";
              let badgeBorder = "rgba(102, 192, 244, 0.35)";
              let badgeColor = "var(--accent-primary)";
              const displayName = item.winnerName;

              if (item.winner === "gameB") {
                badgeBg = "rgba(16, 185, 129, 0.12)";
                badgeBorder = "rgba(16, 185, 129, 0.35)";
                badgeColor = "#10b981";
              } else if (item.winner === "both") {
                badgeBg = "rgba(199, 178, 74, 0.12)";
                badgeBorder = "rgba(199, 178, 74, 0.35)";
                badgeColor = "var(--warning)";
              }

              return (
                <tr
                  key={item.category}
                  style={{
                    backgroundColor: isEven ? "rgba(0, 0, 0, 0.15)" : "transparent",
                  }}
                >
                  <td
                    style={{
                      padding: "12px 16px",
                      color: "var(--text-primary)",
                      fontWeight: 600,
                      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                      verticalAlign: "middle",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span
                        className="material-symbols-outlined"
                        style={{ fontSize: "16px", color: "var(--text-muted)" }}
                      >
                        {item.icon}
                      </span>
                      <span>{item.category}</span>
                    </div>
                  </td>
                  <td
                    style={{
                      padding: "12px 16px",
                      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                      verticalAlign: "middle",
                    }}
                  >
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        padding: "3px 10px",
                        borderRadius: "var(--radius-pill)",
                        backgroundColor: badgeBg,
                        border: `1px solid ${badgeBorder}`,
                        color: badgeColor,
                        fontSize: "12px",
                        fontWeight: 700,
                      }}
                    >
                      {displayName}
                    </span>
                  </td>
                  <td
                    style={{
                      padding: "12px 16px",
                      color: "var(--text-secondary)",
                      lineHeight: 1.5,
                      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                      verticalAlign: "middle",
                      fontSize: "12.5px",
                    }}
                  >
                    {item.context}
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
