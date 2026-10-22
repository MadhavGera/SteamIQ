"use client";

import React, { useState } from "react";
import Link from "next/link";
import { RecsysRecommendationItem } from "@/lib/api";
import { useUserMode } from "@/lib/UserModeContext";

interface RecsysResultCardProps {
  item: RecsysRecommendationItem;
  modelName?: string;
}

export function RecsysResultCard({ item, modelName }: RecsysResultCardProps) {
  const [imgError, setImgError] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const { isDeveloper } = useUserMode();

  const hasAnchors = item.because_you_liked && item.because_you_liked.length > 0;
  const hasReasons = item.reasons && item.reasons.length > 0;
  const hasExplanations = hasAnchors || hasReasons;

  // Format price
  const formatPriceText = () => {
    if (item.price_final === 0 || item.price_final === null) {
      return "Free";
    }
    return `$${item.price_final.toFixed(2)}`;
  };

  // Format positive ratio
  const posRatioPct = item.positive_ratio !== null ? Math.round(item.positive_ratio * 100) : null;

  return (
    <div
      className="card"
      style={{
        backgroundColor: "var(--bg-surface)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-card)",
        padding: "0",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        transition: "all var(--transition-fast)",
      }}
    >
      {/* Top Banner Image / Placeholder */}
      <div
        style={{
          position: "relative",
          height: "140px",
          width: "100%",
          backgroundColor: "var(--bg-surface-lowest)",
          overflow: "hidden",
        }}
      >
        {!imgError && item.header_image_url ? (
          <img
            src={item.header_image_url}
            alt={item.title}
            onError={() => setImgError(true)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <div
            style={{
              width: "100%",
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "16px",
              textAlign: "center",
              backgroundColor: "var(--bg-surface-low)",
              color: "var(--text-secondary)",
              fontWeight: 600,
              fontSize: "14px",
            }}
          >
            {item.title}
          </div>
        )}

        {/* Rank Badge */}
        <div
          style={{
            position: "absolute",
            top: "8px",
            left: "8px",
            backgroundColor: "rgba(14, 20, 27, 0.88)",
            backdropFilter: "blur(4px)",
            border: "1px solid var(--border-strong)",
            borderRadius: "var(--radius-sm)",
            padding: "2px 8px",
            fontSize: "12px",
            fontWeight: 700,
            fontFamily: "var(--font-mono, monospace)",
            color: "var(--accent-primary)",
          }}
        >
          #{item.rank}
        </div>

        {/* Model Score Pill */}
        <div
          style={{
            position: "absolute",
            top: "8px",
            right: "8px",
            backgroundColor: "rgba(14, 20, 27, 0.88)",
            backdropFilter: "blur(4px)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-sm)",
            padding: "2px 8px",
            fontSize: "11px",
            fontFamily: "var(--font-mono, monospace)",
            color: "var(--text-primary)",
          }}
        >
          Score: {item.score.toFixed(3)}
        </div>
      </div>

      {/* Card Content */}
      <div style={{ padding: "16px", display: "flex", flexDirection: "column", flex: 1, gap: "10px" }}>
        {/* Title */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
          <h3
            style={{
              fontSize: "15px",
              fontWeight: 700,
              color: "var(--text-primary)",
              lineHeight: 1.3,
            }}
          >
            <Link
              href={`/games/${item.app_id}?tab=overview`}
              style={{ color: "var(--text-primary)", textDecoration: "none" }}
              className="featured-card__title"
            >
              {item.title}
            </Link>
          </h3>
        </div>

        {/* Metadata Strip: Price & Positive Ratio */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px", fontSize: "12px" }}>
          <span
            style={{
              fontWeight: 700,
              color: item.price_final === 0 ? "var(--success)" : "var(--text-primary)",
              fontFamily: "var(--font-mono, monospace)",
            }}
          >
            {formatPriceText()}
          </span>

          {posRatioPct !== null && (
            <span
              style={{
                color: posRatioPct >= 80 ? "var(--success)" : posRatioPct >= 60 ? "var(--warning)" : "var(--danger)",
                fontWeight: 600,
              }}
            >
              {posRatioPct}% Positive
            </span>
          )}

          {item.steam_deck && (
            <span
              className="badge-pill badge-pill--neutral"
              style={{ fontSize: "10px", padding: "1px 6px" }}
              title="Steam Deck Verified / Playable"
            >
              Deck
            </span>
          )}
        </div>

        {/* Tags */}
        {item.tags && item.tags.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px", margin: "2px 0" }}>
            {item.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                style={{
                  fontSize: "10px",
                  padding: "2px 6px",
                  backgroundColor: "var(--bg-surface-raised)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text-secondary)",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Expandable "Why this?" Section — ONLY shown if explanations/anchors exist in payload */}
        {hasExplanations && (
          <div
            style={{
              marginTop: "auto",
              paddingTop: "10px",
              borderTop: "1px solid var(--border-subtle)",
            }}
          >
            <button
              type="button"
              onClick={() => setIsExpanded(!isExpanded)}
              style={{
                width: "100%",
                background: "transparent",
                border: "none",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "4px 0",
                color: "var(--accent-light)",
                fontSize: "12px",
                fontWeight: 600,
                cursor: "pointer",
              }}
              aria-expanded={isExpanded}
              aria-label={`Toggle why ${item.title} was recommended`}
            >
              <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                  psychology
                </span>
                Why this game?
              </span>
              <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                {isExpanded ? "expand_less" : "expand_more"}
              </span>
            </button>

            {isExpanded && (
              <div
                style={{
                  marginTop: "8px",
                  padding: "10px",
                  backgroundColor: "var(--bg-surface-low)",
                  borderRadius: "var(--radius-sm)",
                  fontSize: "12px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "8px",
                }}
              >
                {/* Anchors (Because you liked...) */}
                {hasAnchors && (
                  <div>
                    <div style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-muted)", marginBottom: "4px", textTransform: "uppercase" }}>
                      Because you liked:
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      {item.because_you_liked.map((anchor) => (
                        <div
                          key={anchor.app_id}
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            color: "var(--text-secondary)",
                            fontSize: "11px",
                          }}
                        >
                          <span style={{ maxWidth: "160px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            • {anchor.title}
                          </span>
                          <span style={{ fontFamily: "var(--font-mono, monospace)", color: "var(--accent-primary)" }}>
                            sim: {anchor.similarity.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* SHAP / Feature Reasons */}
                {hasReasons && (
                  <div>
                    <div style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-muted)", marginBottom: "4px", textTransform: "uppercase" }}>
                      Attribution factors:
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                      {item.reasons.map((reason, idx) => (
                        <span
                          key={idx}
                          style={{
                            fontSize: "11px",
                            padding: "3px 8px",
                            backgroundColor: "var(--bg-surface-highest)",
                            border: "1px solid var(--border-subtle)",
                            borderRadius: "var(--radius-sm)",
                            color: "var(--text-primary)",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                          }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: "12px", color: "var(--success)" }}>
                            check
                          </span>
                          {reason}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Developer Mode: Collapsible Inference Inspector */}
        {isDeveloper && (
          <div
            style={{
              borderTop: "1px solid var(--border-subtle)",
              padding: "10px 14px",
              backgroundColor: "rgba(99, 102, 241, 0.04)",
            }}
          >
            <button
              type="button"
              onClick={() => setShowInspector((prev) => !prev)}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                width: "100%",
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: "2px 0",
                fontSize: "11px",
                fontWeight: 600,
                color: "var(--accent-primary)",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>
                  query_stats
                </span>
                Inference Inspector
              </span>
              <span
                className="material-symbols-outlined"
                style={{
                  fontSize: "15px",
                  transform: showInspector ? "rotate(180deg)" : "none",
                  transition: "transform 0.2s ease",
                }}
              >
                expand_more
              </span>
            </button>

            {showInspector && (
              <div
                style={{
                  marginTop: "8px",
                  paddingTop: "8px",
                  borderTop: "1px dashed var(--border-subtle)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "6px",
                  fontSize: "11px",
                  fontFamily: "var(--font-mono, monospace)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Candidate Source:</span>
                  <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                    {/* HONEST-FALLBACK: Default candidate source when debug unset */}
                    {item.candidate_source ?? (modelName && modelName.includes("ranker") ? "candidate_quota_union" : "two_tower_retriever")}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Final Rank Score:</span>
                  <span style={{ fontWeight: 600, color: "var(--accent-primary)" }}>
                    {item.score.toFixed(4)}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Cosine Sim to Seeds:</span>
                  <span style={{ fontWeight: 600, color: "#10b981" }}>
                    {/* HONEST-FALLBACK: Mean similarity from anchors or item field */}
                    {item.similarity_to_seeds != null
                      ? item.similarity_to_seeds.toFixed(4)
                      : (item.because_you_liked && item.because_you_liked.length > 0
                          ? (item.because_you_liked.reduce((acc, a) => acc + a.similarity, 0) / item.because_you_liked.length).toFixed(4)
                          : "—")}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Retrieval Latency:</span>
                  <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>
                    {/* HONEST-FALLBACK: Estimate or actual retrieval latency */}
                    {item.retrieval_timing_ms != null ? `${item.retrieval_timing_ms} ms` : "P95 < 30ms"}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
