"use client";

import Link from "next/link";
import { CompetitorItem, formatPrice } from "@/lib/api";
import { useUserMode } from "@/lib/UserModeContext";

interface CompetitorTableProps {
  competitors: CompetitorItem[];
  sourceName?: string;
  isProcessed?: boolean;
}

export function CompetitorTable({
  competitors,
  sourceName = "Selected Game",
  isProcessed = true,
}: CompetitorTableProps) {
  const { isPlayer, isDeveloper } = useUserMode();

  if (!isProcessed || competitors.length === 0) {
    return (
      <div className="card" style={{ padding: "48px 24px", textAlign: "center" }}>
        <div style={{ width: "48px", height: "48px", borderRadius: "12px", backgroundColor: "rgba(94, 194, 240, 0.1)", color: "var(--accent-primary)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
          <span className="material-symbols-outlined" style={{ fontSize: "24px" }}>radar</span>
        </div>
        <h3 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)" }}>
          {isPlayer ? "Similar Games Ingestion Pending" : "Competitor Discovery In Progress"}
        </h3>
        <p style={{ maxWidth: "420px", margin: "8px auto 0", fontSize: "14px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
          {isPlayer
            ? "We are currently analyzing gameplay themes and tags to find the best matching games."
            : "Vector embeddings and similarity matrix for this game have not been generated yet. Run make process-embeddings to compute benchmark titles."}
        </p>
      </div>
    );
  }

  return (
    <div className="competitor-table-container">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 24px", borderBottom: "1px solid var(--border-subtle)" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)", fontSize: "20px" }}>
              {isPlayer ? "sports_esports" : "hub"}
            </span>
            <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
              {isPlayer ? `Similar Games to ${sourceName}` : "Top Market Competitors"}
            </h3>
          </div>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
            {isPlayer
              ? `Games with matching gameplay feel, genre DNA, and tags to ${sourceName}`
              : `Ranked by multi-dimensional semantic vector similarity to ${sourceName} (pgvector 384-dim)`}
          </p>
        </div>
        <span className="badge-pill badge-pill--neutral" style={{ fontFamily: "var(--font-mono)" }}>
          {competitors.length} {isPlayer ? "Similar Titles" : "Nearest Titles"}
        </span>
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table className="competitor-table">
          <thead>
            <tr>
              <th style={{ width: "60px", textAlign: "center" }}>Rank</th>
              <th>Game Title</th>
              <th style={{ width: "160px" }}>Match</th>
              {isDeveloper && <th style={{ width: "150px" }}>Market Presence</th>}
              <th>Price &amp; Delta</th>
              <th>Steam Sentiment</th>
              <th>Overlapping Tags</th>
            </tr>
          </thead>
          <tbody>
            {competitors.map((comp) => {
              const isFree = comp.price_usd === "0.00" || comp.price_usd === null;
              const formattedPrice = formatPrice(comp.price_usd, isFree);
              const priceDelta = comp.price_delta_usd;

              return (
                <tr key={comp.app_id}>
                  {/* Rank */}
                  <td style={{ textAlign: "center", fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--text-muted)", fontSize: "12px" }}>
                    #{comp.rank}
                  </td>

                  {/* Game & Capsule */}
                  <td>
                    <Link
                      href={`/games/${comp.app_id}`}
                      style={{ display: "flex", alignItems: "center", gap: "12px", textDecoration: "none" }}
                    >
                      {comp.header_image ? (
                        <img
                          src={comp.header_image}
                          alt={comp.name}
                          style={{ width: "64px", height: "32px", objectFit: "cover", borderRadius: "4px", border: "1px solid var(--border-subtle)", flexShrink: 0 }}
                        />
                      ) : (
                        <div style={{ width: "64px", height: "32px", backgroundColor: "var(--bg-base)", borderRadius: "4px", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                          <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "var(--text-muted)" }}>sports_esports</span>
                        </div>
                      )}
                      <div style={{ minWidth: 0 }}>
                        <span style={{ fontSize: "14px", fontWeight: 600, color: "var(--text-primary)", display: "block" }}>
                          {comp.name}
                        </span>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                          <span>ID: {comp.app_id}</span>
                          {comp.genres && comp.genres.length > 0 && (
                            <>
                              <span>•</span>
                              <span style={{ maxWidth: "160px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {comp.genres.map((g) => g.description).join(", ")}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </Link>
                  </td>

                  {/* Similarity Score */}
                  <td>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", fontFamily: "var(--font-mono)" }}>
                        <span style={{ fontWeight: 700, color: "var(--accent-primary)" }}>
                          {comp.similarity_pct}%
                        </span>
                        <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                          {(comp.similarity_score).toFixed(3)}
                        </span>
                      </div>
                      <div style={{ height: "6px", width: "100%", backgroundColor: "var(--bg-base)", borderRadius: "3px", overflow: "hidden" }}>
                        <div
                          style={{
                            height: "100%",
                            width: `${Math.min(100, Math.max(0, comp.similarity_pct))}%`,
                            background: "linear-gradient(90deg, var(--accent-secondary), var(--accent-primary))",
                            borderRadius: "3px",
                            transition: "width 0.4s ease",
                          }}
                        />
                      </div>
                    </div>
                  </td>

                  {/* Market Presence (Dev Mode Only) */}
                  {isDeveloper && (
                    <td>
                      {comp.market_presence !== null && comp.market_presence !== undefined ? (
                        <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "12px", fontFamily: "var(--font-mono)" }}>
                            <span style={{ fontWeight: 700, color: "var(--accent-light)" }}>
                              {comp.market_presence.toFixed(1)} / 100
                            </span>
                          </div>
                          <div style={{ height: "6px", width: "100%", backgroundColor: "var(--bg-base)", borderRadius: "3px", overflow: "hidden" }}>
                            <div
                              style={{
                                height: "100%",
                                width: `${Math.min(100, Math.max(5, comp.market_presence))}%`,
                                backgroundColor: "var(--accent-light)",
                                borderRadius: "3px",
                              }}
                            />
                          </div>
                        </div>
                      ) : (
                        <span
                          className="badge-pill badge-pill--neutral"
                          style={{
                            fontSize: "11px",
                            color: "var(--text-muted)",
                            padding: "2px 8px",
                            fontFamily: "var(--font-mono)",
                            backgroundColor: "var(--bg-base)",
                            border: "1px dashed var(--border-subtle)",
                          }}
                        >
                          Not yet scored
                        </span>
                      )}
                    </td>
                  )}

                  {/* Price & Delta */}
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", fontFamily: "var(--font-mono)", fontSize: "12px" }}>
                      <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{formattedPrice}</span>
                      {priceDelta !== null && priceDelta !== undefined && (
                        <span
                          style={{
                            fontSize: "10px",
                            padding: "1px 5px",
                            borderRadius: "4px",
                            fontWeight: 600,
                            backgroundColor: priceDelta > 0 ? "rgba(255, 180, 171, 0.15)" : priceDelta < 0 ? "rgba(74, 222, 128, 0.15)" : "var(--bg-base)",
                            color: priceDelta > 0 ? "var(--danger)" : priceDelta < 0 ? "var(--success)" : "var(--text-muted)",
                          }}
                        >
                          {priceDelta > 0 ? `+$${priceDelta.toFixed(2)}` : priceDelta < 0 ? `-$${Math.abs(priceDelta).toFixed(2)}` : "Same"}
                        </span>
                      )}
                    </div>
                  </td>

                  {/* Sentiment */}
                  <td>
                    {comp.review_pct !== null && comp.review_pct !== undefined ? (
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <span
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "12px",
                            fontWeight: 700,
                            color: comp.review_pct >= 80 ? "var(--success)" : comp.review_pct >= 60 ? "var(--warning)" : "var(--danger)",
                          }}
                        >
                          {comp.review_pct}%
                        </span>
                        <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                          ({(comp.positive_reviews + comp.negative_reviews).toLocaleString()})
                        </span>
                      </div>
                    ) : (
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "var(--text-muted)" }}>N/A</span>
                    )}
                  </td>

                  {/* Shared Tags */}
                  <td>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                      {comp.shared_tags && comp.shared_tags.length > 0 ? (
                        comp.shared_tags.map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: "11px",
                              backgroundColor: "var(--bg-base)",
                              border: "1px solid var(--border-subtle)",
                              color: "var(--text-secondary)",
                              padding: "2px 7px",
                              borderRadius: "4px",
                            }}
                          >
                            {tag}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: "var(--text-muted)", fontSize: "12px" }}>—</span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
