"use client";

import React from "react";
import { LovedFeatureData } from "@/lib/api";

export interface LovedFeaturesProps {
  features: LovedFeatureData[];
  loading?: boolean;
}

export function LovedFeatures({ features, loading = false }: LovedFeaturesProps) {
  if (loading) {
    return (
      <div className="card" style={{ padding: "24px" }}>
        <div className="skeleton" style={{ height: "20px", width: "180px", marginBottom: "16px" }} />
        {[1, 2, 3, 4].map((n) => (
          <div key={n} className="skeleton" style={{ height: "36px", width: "100%", marginBottom: "10px" }} />
        ))}
      </div>
    );
  }

  if (!features || features.length === 0) {
    return (
      <div className="card" style={{ padding: "24px", textAlign: "center", color: "var(--text-muted)" }}>
        <p style={{ margin: "24px 0" }}>No feature appreciation patterns detected yet.</p>
      </div>
    );
  }

  const maxMentions = Math.max(...features.map((f) => f.mention_count), 1);

  return (
    <div className="card" style={{ padding: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
        <span style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
          Loved Features & Acclaimed Mechanics
        </span>
        <span style={{ fontSize: "12px", color: "var(--success)", fontWeight: 600 }}>
          Top positive drivers
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {features.map((f, i) => {
          const widthPct = Math.max(15, Math.round((f.mention_count / maxMentions) * 100));

          return (
            <div key={f.feature_name || i} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{f.feature_name}</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "var(--text-secondary)" }}>
                  {f.mention_count.toLocaleString()} mentions · {f.praise_intensity}% praise
                </span>
              </div>

              <div
                style={{
                  height: "6px",
                  width: "100%",
                  background: "var(--bg-base)",
                  borderRadius: "3px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${widthPct}%`,
                    background: "var(--success)",
                    borderRadius: "3px",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
