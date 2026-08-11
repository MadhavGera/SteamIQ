"use client";

import Link from "next/link";
import { CompetitorItem } from "@/lib/api";

interface CompetitorRadarTeaserProps {
  appId: number;
  competitors: CompetitorItem[];
}

export function CompetitorRadarTeaser({
  appId,
  competitors,
}: CompetitorRadarTeaserProps) {
  const top3 = competitors.slice(0, 3);

  if (top3.length === 0) return null;

  return (
    <Link href={`/games/${appId}?tab=competitors`} className="card card--hoverable">
      <div className="flex items-center justify-between" style={{ marginBottom: "12px" }}>
        <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)" }}>
          Competitor Radar
        </h3>
        <span className="badge-pill badge-pill--neutral">pgvector</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "13px" }}>
        {top3.map((comp) => (
          <div key={comp.app_id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ color: "var(--text-primary)", fontWeight: 500 }} className="truncate max-w-[170px]">
              {comp.name}
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              {comp.price_delta_usd !== null && comp.price_delta_usd !== undefined && comp.price_delta_usd !== 0 && (
                <span
                  style={{
                    fontSize: "10px",
                    fontFamily: "var(--font-mono)",
                    color: comp.price_delta_usd > 0 ? "var(--warning)" : "var(--success)",
                  }}
                >
                  {comp.price_delta_usd > 0 ? `+$${comp.price_delta_usd.toFixed(0)}` : `-$${Math.abs(comp.price_delta_usd).toFixed(0)}`}
                </span>
              )}
              <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-primary)", fontWeight: 700 }}>
                {comp.similarity_pct}%
              </span>
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: "16px",
          fontSize: "12px",
          fontWeight: 600,
          color: "var(--accent-primary)",
          display: "flex",
          alignItems: "center",
          gap: "4px",
        }}
      >
        View full competitor matrix{" "}
        <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>
          arrow_forward
        </span>
      </div>
    </Link>
  );
}
