"use client";

import React from "react";
import { useUserMode } from "@/lib/UserModeContext";

export function UserModeToggle({ className }: { className?: string }) {
  const { mode, setMode } = useUserMode();

  return (
    <div
      className={`user-mode-toggle ${className || ""}`}
      style={{
        display: "inline-flex",
        alignItems: "center",
        backgroundColor: "var(--bg-surface)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "999px",
        padding: "3px",
        gap: "2px",
      }}
      role="radiogroup"
      aria-label="SteamIQ Audience Mode Toggle"
    >
      <button
        type="button"
        role="radio"
        aria-checked={mode === "player"}
        onClick={() => setMode("player")}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "6px",
          padding: "5px 12px",
          borderRadius: "999px",
          border: "none",
          cursor: "pointer",
          fontSize: "12px",
          fontWeight: 600,
          transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
          backgroundColor: mode === "player" ? "var(--accent-primary)" : "transparent",
          color: mode === "player" ? "var(--bg-base)" : "var(--text-secondary)",
        }}
      >
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "15px", fontVariationSettings: mode === "player" ? "'FILL' 1" : "'FILL' 0" }}
        >
          sports_esports
        </span>
        <span>Player</span>
      </button>

      <button
        type="button"
        role="radio"
        aria-checked={mode === "developer"}
        onClick={() => setMode("developer")}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "6px",
          padding: "5px 12px",
          borderRadius: "999px",
          border: "none",
          cursor: "pointer",
          fontSize: "12px",
          fontWeight: 600,
          transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
          backgroundColor: mode === "developer" ? "var(--accent-primary)" : "transparent",
          color: mode === "developer" ? "var(--bg-base)" : "var(--text-secondary)",
        }}
      >
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "15px", fontVariationSettings: mode === "developer" ? "'FILL' 1" : "'FILL' 0" }}
        >
          terminal
        </span>
        <span>Developer</span>
      </button>
    </div>
  );
}
