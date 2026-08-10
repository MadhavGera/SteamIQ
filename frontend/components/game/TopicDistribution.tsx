"use client";

import React from "react";
import { ReviewTopicData } from "@/lib/api";

export interface TopicDistributionProps {
  topics: ReviewTopicData[];
  onTopicSelect?: (topicId: number) => void;
  loading?: boolean;
}

export function TopicDistribution({
  topics,
  onTopicSelect,
  loading = false,
}: TopicDistributionProps) {
  if (loading) {
    return (
      <div className="card" style={{ padding: "24px" }}>
        <div className="skeleton" style={{ height: "20px", width: "180px", marginBottom: "16px" }} />
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
          {[1, 2, 3, 4, 5].map((n) => (
            <div key={n} className="skeleton" style={{ height: "32px", width: "140px", borderRadius: "16px" }} />
          ))}
        </div>
      </div>
    );
  }

  if (!topics || topics.length === 0) {
    return (
      <div className="card" style={{ padding: "24px", textAlign: "center", color: "var(--text-muted)" }}>
        <p style={{ margin: "24px 0" }}>No semantic topic clusters extracted yet.</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
        <span style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
          Extracted Review Topics (BERTopic)
        </span>
        <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
          {topics.length} semantic clusters
        </span>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
        {topics.map((t) => {
          const isPositive = t.sentiment_score >= 0.8;
          const isModerate = t.sentiment_score >= 0.6;
          const color = isPositive ? "var(--success)" : isModerate ? "var(--warning)" : "var(--danger)";

          return (
            <div
              key={t.topic_id}
              onClick={() => onTopicSelect && onTopicSelect(t.topic_id)}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                background: "var(--bg-surface-raised)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-pill)",
                padding: "8px 14px",
                fontSize: "13px",
                fontWeight: 500,
                color: "var(--text-primary)",
                cursor: onTopicSelect ? "pointer" : "default",
                transition: "all var(--transition-fast)",
              }}
            >
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: color }} />
              <span>{t.label}</span>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: 600,
                  fontFamily: "var(--font-mono)",
                  color: "var(--text-muted)",
                  background: "var(--bg-base)",
                  padding: "2px 6px",
                  borderRadius: "10px",
                }}
              >
                {t.review_count} revs
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
