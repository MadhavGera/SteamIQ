"use client";

import React, { useState } from "react";
import Link from "next/link";
import { GameCandidate } from "@/lib/api";

interface CandidateGameCardProps {
  candidate: GameCandidate;
}

export function CandidateGameCard({ candidate }: CandidateGameCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [imageError, setImageError] = useState(false);

  // Format score as percentage if in 0..1 range
  const displayScore =
    candidate.score > 0 && candidate.score <= 1
      ? `${(candidate.score * 100).toFixed(1)}%`
      : candidate.score.toFixed(2);

  const fallbackImage = `https://cdn.cloudflare.steamstatic.com/steam/apps/${candidate.app_id}/header.jpg`;
  const imageSrc = imageError
    // HONEST-FALLBACK: Fallback to official steam static CDN header image when custom image fails
    ? fallbackImage
    // HONEST-FALLBACK: Fallback to official steam CDN when header_image is absent
    : (candidate.header_image ?? fallbackImage);

  return (
    <div
      className="card"
      style={{
        background: "var(--bg-surface-elevated)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "12px",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        transition: "transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease",
        boxShadow: "0 4px 14px rgba(0, 0, 0, 0.25)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "var(--border-accent)";
        e.currentTarget.style.transform = "translateY(-2px)";
        e.currentTarget.style.boxShadow = "0 8px 24px rgba(0, 210, 255, 0.15)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "var(--border-subtle)";
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.boxShadow = "0 4px 14px rgba(0, 0, 0, 0.25)";
      }}
    >
      {/* Thumbnail with overlay badges */}
      <div style={{ position: "relative", width: "100%", height: "140px", background: "var(--bg-surface-low)" }}>
        {/* Next.js img or raw img for dynamic external domains without domain whitelisting issues */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageSrc}
          alt={candidate.title}
          onError={() => setImageError(true)}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "block",
          }}
        />

        {/* Rank Badge */}
        <div
          style={{
            position: "absolute",
            top: "8px",
            left: "8px",
            background: "rgba(10, 15, 29, 0.85)",
            backdropFilter: "blur(6px)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "6px",
            padding: "2px 8px",
            fontSize: "12px",
            fontWeight: 700,
            color: candidate.rank === 1 ? "#fbbf24" : "var(--accent-primary)",
          }}
        >
          #{candidate.rank}
        </div>

        {/* Similarity Score Badge */}
        <div
          style={{
            position: "absolute",
            top: "8px",
            right: "8px",
            background: "rgba(10, 15, 29, 0.85)",
            backdropFilter: "blur(6px)",
            border: "1px solid rgba(0, 210, 255, 0.3)",
            borderRadius: "6px",
            padding: "2px 8px",
            fontSize: "12px",
            fontWeight: 700,
            color: "var(--accent-primary)",
            display: "flex",
            alignItems: "center",
            gap: "4px",
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>
            hub
          </span>
          {displayScore}
        </div>

        {/* Price & Rating Overlay at Bottom */}
        <div
          style={{
            position: "absolute",
            bottom: "6px",
            right: "8px",
            display: "flex",
            gap: "6px",
          }}
        >
          {candidate.positive_ratio !== null && candidate.positive_ratio !== undefined && (
            <span
              style={{
                background: "rgba(16, 185, 129, 0.85)",
                color: "#ffffff",
                padding: "1px 6px",
                borderRadius: "4px",
                fontSize: "11px",
                fontWeight: 600,
              }}
            >
              {candidate.positive_ratio}% Positive
            </span>
          )}
          {candidate.price_usd && (
            <span
              style={{
                background: "rgba(10, 15, 29, 0.85)",
                color: "var(--text-primary)",
                padding: "1px 6px",
                borderRadius: "4px",
                fontSize: "11px",
                fontWeight: 600,
                border: "1px solid var(--border-subtle)",
              }}
            >
              {candidate.price_usd.startsWith("$") ? candidate.price_usd : `$${candidate.price_usd}`}
            </span>
          )}
        </div>
      </div>

      {/* Content Details */}
      <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", flex: 1, gap: "8px" }}>
        {/* Title */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "8px" }}>
          <h3
            style={{
              fontSize: "16px",
              fontWeight: 700,
              color: "var(--text-primary)",
              lineHeight: 1.3,
              margin: 0,
            }}
          >
            {candidate.title}
          </h3>
          <Link
            href={`https://store.steampowered.com/app/${candidate.app_id}`}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "var(--text-muted)",
              display: "inline-flex",
              alignItems: "center",
              textDecoration: "none",
            }}
            title="Open on Steam Store"
          >
            <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
              open_in_new
            </span>
          </Link>
        </div>

        {/* Tags */}
        {candidate.tags && candidate.tags.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
            {candidate.tags.slice(0, 4).map((tag, i) => (
              <span
                key={i}
                style={{
                  fontSize: "11px",
                  padding: "2px 6px",
                  background: "var(--bg-surface-high)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "4px",
                  color: "var(--text-secondary)",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Match Rationale Snippet */}
        <div
          style={{
            fontSize: "12px",
            color: "var(--text-secondary)",
            background: "var(--bg-surface-low)",
            padding: "8px 10px",
            borderRadius: "6px",
            borderLeft: "3px solid var(--accent-primary)",
            lineHeight: 1.4,
            marginTop: "auto",
          }}
        >
          <strong style={{ color: "var(--accent-primary)" }}>Match Rationale: </strong>
          {candidate.match_reason}
        </div>

        {/* Collapsible Explanations */}
        <div style={{ marginTop: "4px" }}>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-muted)",
              fontSize: "11px",
              cursor: "pointer",
              padding: "4px 0",
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>
              {isExpanded ? "expand_less" : "expand_more"}
            </span>
            <span>{isExpanded ? "Hide Model Diagnostics" : "Inspect Model Provenance"}</span>
          </button>

          {isExpanded && (
            <div
              style={{
                marginTop: "6px",
                padding: "8px 10px",
                background: "var(--bg-surface-high)",
                borderRadius: "6px",
                fontSize: "11px",
                color: "var(--text-secondary)",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <div>
                <span style={{ color: "var(--text-muted)" }}>Model Architecture: </span>
                <code>{candidate.score_type}</code>
              </div>
              <div>
                <span style={{ color: "var(--text-muted)" }}>Steam App ID: </span>
                <code>{candidate.app_id}</code>
              </div>
              {(() => {
                const sharedTags = Array.isArray(candidate.explanation?.shared_tags)
                  ? (candidate.explanation?.shared_tags as string[])
                  : [];
                if (sharedTags.length === 0) return null;
                return (
                  <div>
                    <span style={{ color: "var(--text-muted)" }}>Shared Feature Attributes: </span>
                    <span>{sharedTags.join(", ")}</span>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
