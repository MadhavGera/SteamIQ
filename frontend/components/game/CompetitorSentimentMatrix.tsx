"use client";

import Link from "next/link";
import { CompetitorItem, CompetitorSourceGame, formatPrice } from "@/lib/api";

interface CompetitorSentimentMatrixProps {
  sourceGame: CompetitorSourceGame;
  competitors: CompetitorItem[];
}

export function CompetitorSentimentMatrix({
  sourceGame,
  competitors,
}: CompetitorSentimentMatrixProps) {
  const topCompetitors = competitors.slice(0, 3);

  if (topCompetitors.length === 0) return null;

  return (
    <div className="card" style={{ padding: "24px" }}>
      {/* Section Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)", fontSize: "20px" }}>
              compare_arrows
            </span>
            <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
              Feature &amp; Sentiment Benchmark Matrix
            </h3>
          </div>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Side-by-side market positioning across top 3 closest competitors
          </p>
        </div>
        <span className="badge-pill badge-pill--neutral">
          pgvector Cosine Nearest
        </span>
      </div>

      {/* Grid comparing Source vs Top 3 */}
      <div className="competitor-matrix-grid">
        {/* Source Game (Focus) */}
        <div className="competitor-card competitor-card--focus">
          <div
            style={{
              position: "absolute",
              top: "-10px",
              left: "14px",
              backgroundColor: "var(--accent-primary)",
              color: "var(--bg-base)",
              fontSize: "10px",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              padding: "2px 8px",
              borderRadius: "var(--radius-pill)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Analyzed Game
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "4px", marginBottom: "14px" }}>
            {sourceGame.header_image ? (
              <img
                src={sourceGame.header_image}
                alt={sourceGame.name}
                style={{ width: "72px", height: "36px", objectFit: "cover", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}
              />
            ) : (
              <div style={{ width: "72px", height: "36px", backgroundColor: "var(--bg-base)", borderRadius: "var(--radius-sm)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "var(--text-muted)" }}>sports_esports</span>
              </div>
            )}
            <div style={{ minWidth: 0, flex: 1 }}>
              <h4 style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {sourceGame.name}
              </h4>
              <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--accent-primary)" }}>
                Source Baseline
              </span>
            </div>
          </div>

          <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "12px", display: "flex", flexDirection: "column", gap: "10px", fontSize: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Price</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--text-primary)" }}>
                {formatPrice(sourceGame.final_price_usd, sourceGame.final_price_usd === "0.00")}
              </span>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Steam Sentiment</span>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontWeight: 600,
                  // HONEST-FALLBACK: Null-safe numeric comparison for badge color
                  color: (sourceGame.review_pct ?? 0) >= 80 ? "var(--success)" : "var(--warning)",
                }}
              >
                {sourceGame.review_pct !== null ? `${sourceGame.review_pct}% Positive` : "N/A"}
              </span>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Review Volume</span>
              <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                {(sourceGame.positive_reviews + sourceGame.negative_reviews).toLocaleString()}
              </span>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Benchmark Fit</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-primary)" }}>
                100% (Base)
              </span>
            </div>
          </div>
        </div>

        {/* Competitor Cards */}
        {topCompetitors.map((comp) => {
          const isFree = comp.price_usd === "0.00" || comp.price_usd === null;
          const formattedPrice = formatPrice(comp.price_usd, isFree);
          const priceDelta = comp.price_delta_usd;

          return (
            <div key={comp.app_id} className="competitor-card">
              <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "14px" }}>
                {comp.header_image ? (
                  <img
                    src={comp.header_image}
                    alt={comp.name}
                    style={{ width: "72px", height: "36px", objectFit: "cover", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}
                  />
                ) : (
                  <div style={{ width: "72px", height: "36px", backgroundColor: "var(--bg-base)", borderRadius: "var(--radius-sm)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "var(--text-muted)" }}>sports_esports</span>
                  </div>
                )}
                <div style={{ minWidth: 0, flex: 1 }}>
                  <Link
                    href={`/games/${comp.app_id}`}
                    style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", display: "block", textDecoration: "none" }}
                  >
                    {comp.name}
                  </Link>
                  <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    Rank #{comp.rank}
                  </span>
                </div>
              </div>

              <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "12px", display: "flex", flexDirection: "column", gap: "10px", fontSize: "12px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ color: "var(--text-muted)" }}>Price</span>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", fontFamily: "var(--font-mono)" }}>
                    <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{formattedPrice}</span>
                    {priceDelta !== null && priceDelta !== undefined && (
                      <span
                        style={{
                          fontSize: "10px",
                          color: priceDelta > 0 ? "var(--danger)" : priceDelta < 0 ? "var(--success)" : "var(--text-muted)",
                        }}
                      >
                        ({priceDelta > 0 ? `+$${priceDelta.toFixed(2)}` : priceDelta < 0 ? `-$${Math.abs(priceDelta).toFixed(2)}` : "Same"})
                      </span>
                    )}
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Steam Sentiment</span>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontWeight: 600,
                      // HONEST-FALLBACK: Null-safe numeric comparison for badge color
                      color: (comp.review_pct ?? 0) >= 80 ? "var(--success)" : "var(--warning)",
                    }}
                  >
                    {comp.review_pct !== null ? `${comp.review_pct}% Positive` : "N/A"}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Review Volume</span>
                  <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                    {(comp.positive_reviews + comp.negative_reviews).toLocaleString()}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Similarity Match</span>
                  <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-primary)" }}>
                    {comp.similarity_pct}%
                  </span>
                </div>
              </div>

              {/* Shared Tags Pill Row */}
              {comp.shared_tags && comp.shared_tags.length > 0 && (
                <div style={{ borderTop: "1px solid rgba(62, 72, 78, 0.3)", paddingTop: "10px", marginTop: "10px" }}>
                  <span style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
                    Shared Tags
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                    {comp.shared_tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        style={{
                          fontSize: "10px",
                          backgroundColor: "var(--bg-base)",
                          border: "1px solid var(--border-subtle)",
                          color: "var(--text-secondary)",
                          padding: "1px 6px",
                          borderRadius: "4px",
                        }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
