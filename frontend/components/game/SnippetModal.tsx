"use client";

import React, { useEffect } from "react";

export interface SnippetModalProps {
  isOpen: boolean;
  categoryTitle: string;
  snippets: string[];
  onClose: () => void;
}

export function SnippetModal({
  isOpen,
  categoryTitle,
  snippets,
  onClose,
}: SnippetModalProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(11, 20, 32, 0.75)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
        padding: "20px",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--bg-surface-raised)",
          border: "1px solid var(--border-strong)",
          borderRadius: "16px",
          width: "100%",
          maxWidth: "600px",
          boxShadow: "0 12px 36px rgba(0,0,0,0.5)",
          padding: "24px",
          position: "relative",
          animation: "modalFadeIn 0.2s ease-out",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "12px" }}>
          <div>
            <span style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--danger)" }}>
              Representative Player Feedback
            </span>
            <h3 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
              {categoryTitle}
            </h3>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-muted)",
              fontSize: "20px",
              cursor: "pointer",
              padding: "4px 8px",
              borderRadius: "4px",
            }}
          >
            ✕
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "12px", maxHeight: "400px", overflowY: "auto" }}>
          {snippets && snippets.length > 0 ? (
            snippets.map((snip, i) => (
              <div
                key={i}
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "8px",
                  padding: "14px 16px",
                  fontSize: "13px",
                  lineHeight: "1.6",
                  color: "var(--text-secondary)",
                  fontStyle: "italic",
                  position: "relative",
                }}
              >
                &ldquo;{snip}&rdquo;
              </div>
            ))
          ) : (
            <p style={{ color: "var(--text-muted)", fontSize: "13px" }}>No specific snippets recorded for this complaint.</p>
          )}
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "20px" }}>
          <button
            onClick={onClose}
            className="btn btn--secondary"
            style={{ padding: "8px 18px", fontSize: "13px" }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
