"use client";

import React, { useState } from "react";
import { ComplaintCategoryData } from "@/lib/api";
import { SnippetModal } from "./SnippetModal";

export interface ComplaintBreakdownProps {
  complaints: ComplaintCategoryData[];
  onViewSnippets?: (complaint: ComplaintCategoryData) => void;
  loading?: boolean;
}

export function ComplaintBreakdown({
  complaints,
  onViewSnippets,
  loading = false,
}: ComplaintBreakdownProps) {
  const [activeComplaint, setActiveComplaint] = useState<ComplaintCategoryData | null>(null);

  if (loading) {
    return (
      <div className="card" style={{ padding: "24px" }}>
        <div className="skeleton" style={{ height: "20px", width: "200px", marginBottom: "16px" }} />
        {[1, 2, 3].map((n) => (
          <div key={n} className="skeleton" style={{ height: "48px", width: "100%", marginBottom: "10px" }} />
        ))}
      </div>
    );
  }

  if (!complaints || complaints.length === 0) {
    return (
      <div className="card" style={{ padding: "24px", textAlign: "center", color: "var(--text-muted)" }}>
        <p style={{ margin: "24px 0" }}>No significant player complaints or friction detected.</p>
      </div>
    );
  }

  const handleSnippetClick = (c: ComplaintCategoryData) => {
    if (onViewSnippets) {
      onViewSnippets(c);
    } else {
      setActiveComplaint(c);
    }
  };

  return (
    <div className="card" style={{ padding: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
        <div>
          <span style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
            Player Complaint Breakdown & Pain Points
          </span>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", margin: "4px 0 0" }}>
            Zero-shot multi-label classification over negative reviews
          </p>
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table className="table" style={{ width: "100%", textAlign: "left", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-subtle)", fontSize: "11px", textTransform: "uppercase", color: "var(--text-muted)" }}>
              <th style={{ padding: "10px 12px" }}>Issue Category</th>
              <th style={{ padding: "10px 12px" }}>Negative Share %</th>
              <th style={{ padding: "10px 12px" }}>Severity</th>
              <th style={{ padding: "10px 12px", textAlign: "right" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {complaints.map((c, i) => {
              const sevBadge =
                c.severity === "high"
                  ? "badge-pill--danger"
                  : c.severity === "moderate"
                  ? "badge-pill--warning"
                  : "badge-pill--accent";

              return (
                <tr
                  key={c.category || i}
                  style={{
                    borderBottom: "1px solid var(--border-subtle)",
                    fontSize: "13px",
                    transition: "background var(--transition-fast)",
                  }}
                >
                  <td style={{ padding: "14px 12px", fontWeight: 600, color: "var(--text-primary)" }}>
                    {c.category}
                  </td>
                  <td style={{ padding: "14px 12px", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    {c.volume_pct.toFixed(1)}%
                  </td>
                  <td style={{ padding: "14px 12px" }}>
                    <span className={`badge-pill ${sevBadge}`} style={{ textTransform: "capitalize" }}>
                      {c.severity}
                    </span>
                  </td>
                  <td style={{ padding: "14px 12px", textAlign: "right" }}>
                    <button
                      onClick={() => handleSnippetClick(c)}
                      className="btn btn--secondary"
                      style={{ padding: "4px 10px", fontSize: "11px" }}
                    >
                      View Snippets 💬
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {activeComplaint && (
        <SnippetModal
          isOpen={true}
          categoryTitle={activeComplaint.category}
          snippets={activeComplaint.representative_snippets}
          onClose={() => setActiveComplaint(null)}
        />
      )}
    </div>
  );
}
