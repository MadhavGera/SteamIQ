"use client";

import React, { useState, useEffect, useTransition } from "react";
import {
  api,
  AssistantAskResponse,
  AssistantHealthResponse,
  GameSummary,
} from "@/lib/api";
import { useUserMode } from "@/lib/UserModeContext";

export default function AskSteamIQPage() {
  const { mode, isPlayer } = useUserMode();

  const [query, setQuery] = useState("");
  const [selectedGame, setSelectedGame] = useState<GameSummary | null>(null);
  const [gameSearchQuery, setGameSearchQuery] = useState("");
  const [gameSearchResults, setGameSearchResults] = useState<GameSummary[]>([]);
  const [showGamePicker, setShowGamePicker] = useState(false);

  const [response, setResponse] = useState<AssistantAskResponse | null>(null);
  const [health, setHealth] = useState<AssistantHealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [, startTransition] = useTransition();

  // Load Assistant health
  useEffect(() => {
    let isMounted = true;
    api
      .assistantHealth()
      .then((h) => {
        if (isMounted) setHealth(h);
      })
      .catch(() => {
        if (isMounted) {
          setHealth({
            status: "standby",
            is_ready: false,
            model_name: "assistant_intent_main",
            run_id: null,
            tau: 0.85,
          });
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Search games for the optional typeahead picker
  useEffect(() => {
    if (!gameSearchQuery.trim()) {
      setGameSearchResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const res = await api.searchGames(gameSearchQuery);
        setGameSearchResults(res?.games || []);
      } catch {
        setGameSearchResults([]);
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [gameSearchQuery]);

  const handleAsk = async (queryToAsk?: string) => {
    const q = (queryToAsk || query).trim();
    if (!q) return;

    setIsLoading(true);
    setError(null);

    try {
      const res = await api.assistantAsk({
        query: q,
        user_mode: mode,
        game_app_id: selectedGame?.app_id || null,
      });
      startTransition(() => {
        setResponse(res);
        setIsLoading(false);
      });
    } catch (err: unknown) {
      startTransition(() => {
        const msg = err instanceof Error ? err.message : "Failed to answer query";
        setError(msg);
        setIsLoading(false);
      });
    }
  };

  const samplePrompts = isPlayer
    ? [
        "What is Terraria about?",
        "What do players complain about in Cyberpunk 2077?",
        "Why do players love Hollow Knight?",
        "Recommend games similar to Hades",
        "Should I buy Baldur's Gate 3 now or wait for a discount?",
      ]
    : [
        "What is the market opportunity for a Roguelike Deckbuilder?",
        "What were the feature drivers of success for Elden Ring?",
        "How did recent updates impact player reception for No Man's Sky?",
        "What are competitor price points and discounts in the RPG genre?",
        "What are the top complaint themes for indie survival games?",
      ];

  const formatIntentLabel = (intent: string): string => {
    return intent
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
  };

  return (
    <div className="container" style={{ paddingBottom: "80px", maxWidth: "960px", margin: "0 auto" }}>
      {/* Header Banner */}
      <div style={{ padding: "32px 0 20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
          <span
            className="material-symbols-outlined"
            style={{ fontSize: "24px", color: "var(--accent-primary)" }}
          >
            chat_spark
          </span>
          <span
            className="badge-pill"
            style={{
              background: isPlayer ? "rgba(99, 102, 241, 0.15)" : "rgba(16, 185, 129, 0.15)",
              color: isPlayer ? "#818cf8" : "#34d399",
              borderColor: isPlayer ? "rgba(99, 102, 241, 0.3)" : "rgba(16, 185, 129, 0.3)",
              fontWeight: 600,
            }}
          >
            {isPlayer ? "Player Discovery Mode" : "Developer Intelligence Mode"}
          </span>
          {health?.is_ready && (
            <span className="badge-pill badge-pill--success" style={{ fontSize: "11px" }}>
              Model Online (tau = {health.tau})
            </span>
          )}
        </div>
        <h1 style={{ fontSize: "32px", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
          Ask SteamIQ
        </h1>
        <p style={{ fontSize: "15px", color: "var(--text-secondary)", marginTop: "6px" }}>
          Learned, calibrated query routing with extractive passage retrieval. Zero hallucinated free-text;
          answers are executed strictly against validated analytical data marts.
        </p>
      </div>

      {/* Query Bar & Game Selector */}
      <div
        className="card"
        style={{
          background: "var(--bg-surface-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "12px",
          padding: "20px",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.2)",
        }}
      >
        {/* Optional Game Scope Badge / Selector */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "14px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "13px", color: "var(--text-muted)", fontWeight: 500 }}>
            Target Game Context:
          </span>
          {selectedGame ? (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "4px 10px",
                background: "var(--bg-surface-high)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "6px",
                fontSize: "13px",
                color: "var(--text-primary)",
              }}
            >
              <strong>{selectedGame.name}</strong>
              <button
                onClick={() => setSelectedGame(null)}
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--text-muted)",
                  padding: "0 2px",
                  fontSize: "14px",
                }}
                title="Remove game filter"
              >
                ✕
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowGamePicker(!showGamePicker)}
              className="btn-secondary"
              style={{ fontSize: "12px", padding: "4px 10px" }}
            >
              + Tag Specific Game (Optional)
            </button>
          )}

          {/* Typeahead Dropdown */}
          {showGamePicker && !selectedGame && (
            <div style={{ width: "100%", marginTop: "8px", position: "relative" }}>
              <input
                type="text"
                placeholder="Type to search games (e.g. Portal, Terraria)..."
                value={gameSearchQuery}
                onChange={(e) => setGameSearchQuery(e.target.value)}
                autoFocus
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  background: "var(--bg-surface-low)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "6px",
                  color: "var(--text-primary)",
                  fontSize: "13px",
                }}
              />
              {gameSearchResults.length > 0 && (
                <div
                  style={{
                    position: "absolute",
                    top: "100%",
                    left: 0,
                    right: 0,
                    zIndex: 20,
                    background: "var(--bg-surface-high)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "6px",
                    marginTop: "4px",
                    maxHeight: "220px",
                    overflowY: "auto",
                    boxShadow: "0 8px 24px rgba(0, 0, 0, 0.4)",
                  }}
                >
                  {gameSearchResults.map((g) => (
                    <div
                      key={g.app_id}
                      onClick={() => {
                        setSelectedGame(g);
                        setShowGamePicker(false);
                        setGameSearchQuery("");
                      }}
                      style={{
                        padding: "8px 12px",
                        cursor: "pointer",
                        borderBottom: "1px solid var(--border-subtle)",
                        fontSize: "13px",
                        display: "flex",
                        justifyContent: "space-between",
                        color: "var(--text-primary)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = "var(--bg-surface-elevated)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "transparent";
                      }}
                    >
                      <span>{g.name}</span>
                      <span style={{ color: "var(--text-muted)", fontSize: "11px" }}>#{g.app_id}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Query Input */}
        <div style={{ display: "flex", gap: "10px" }}>
          <input
            type="text"
            placeholder={
              isPlayer
                ? "Ask about game features, sentiment, complaints, or recommendations..."
                : "Ask about market sizing, success drivers, update impacts, or competitors..."
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !isLoading) {
                handleAsk();
              }
            }}
            disabled={isLoading}
            style={{
              flex: 1,
              padding: "12px 16px",
              background: "var(--bg-surface-low)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "8px",
              color: "var(--text-primary)",
              fontSize: "15px",
            }}
          />
          <button
            onClick={() => handleAsk()}
            disabled={isLoading || !query.trim()}
            className="btn-primary"
            style={{
              padding: "12px 24px",
              borderRadius: "8px",
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            {isLoading ? (
              <span>Evaluating...</span>
            ) : (
              <>
                <span>Ask</span>
                <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                  send
                </span>
              </>
            )}
          </button>
        </div>

        {/* Suggested Prompts */}
        <div style={{ marginTop: "14px" }}>
          <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "8px" }}>
            Suggested Questions ({isPlayer ? "Player Discovery" : "Developer Intelligence"}):
          </span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {samplePrompts.map((p, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setQuery(p);
                  handleAsk(p);
                }}
                className="badge-pill badge-pill--interactive"
                style={{
                  fontSize: "12px",
                  padding: "4px 10px",
                  cursor: "pointer",
                  background: "var(--bg-surface-high)",
                  border: "1px solid var(--border-subtle)",
                  color: "var(--text-secondary)",
                }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div
          style={{
            marginTop: "20px",
            padding: "16px",
            background: "rgba(239, 68, 68, 0.1)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: "8px",
            color: "#f87171",
            fontSize: "14px",
          }}
        >
          <strong>Query Evaluation Failed:</strong> {error}
        </div>
      )}

      {/* Response Card */}
      {response && (
        <div
          className="card"
          style={{
            marginTop: "24px",
            background: "var(--bg-surface-elevated)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "12px",
            padding: "24px",
            boxShadow: "0 6px 24px rgba(0, 0, 0, 0.25)",
          }}
        >
          {/* Metadata Top Row */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "16px",
              flexWrap: "wrap",
              gap: "10px",
              borderBottom: "1px solid var(--border-subtle)",
              paddingBottom: "12px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span
                className="badge-pill"
                style={{
                  background: "rgba(99, 102, 241, 0.2)",
                  color: "#a5b4fc",
                  border: "1px solid rgba(99, 102, 241, 0.3)",
                  fontWeight: 600,
                  fontSize: "13px",
                }}
              >
                Intent: {formatIntentLabel(response.intent)}
              </span>

              <span
                className="badge-pill"
                style={{
                  background:
                    response.confidence >= 0.85
                      ? "rgba(16, 185, 129, 0.15)"
                      : "rgba(245, 158, 11, 0.15)",
                  color: response.confidence >= 0.85 ? "#34d399" : "#fbbf24",
                  border: `1px solid ${
                    response.confidence >= 0.85
                      ? "rgba(16, 185, 129, 0.3)"
                      : "rgba(245, 158, 11, 0.3)"
                  }`,
                  fontSize: "12px",
                }}
              >
                Confidence: {(response.confidence * 100).toFixed(1)}%
              </span>

              {response.resolved_game && (
                <span className="badge-pill badge-pill--neutral" style={{ fontSize: "12px" }}>
                  Game: {response.resolved_game.title} (#{response.resolved_game.app_id})
                </span>
              )}
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px" }}>
                Mode: {response.user_mode}
              </span>
              {response.provenance.cached && (
                <span
                  className="badge-pill"
                  style={{
                    background: "rgba(59, 130, 246, 0.15)",
                    color: "#60a5fa",
                    fontSize: "11px",
                  }}
                >
                  ⚡ Cache Hit
                </span>
              )}
            </div>
          </div>

          {/* Abstained Warning */}
          {response.abstained && (
            <div
              style={{
                marginBottom: "16px",
                padding: "12px 16px",
                background: "rgba(245, 158, 11, 0.1)",
                border: "1px solid rgba(245, 158, 11, 0.3)",
                borderRadius: "8px",
                color: "#fbbf24",
                fontSize: "13px",
              }}
            >
              <strong>Selective Abstention Active:</strong> Calibrated router confidence (
              {(response.confidence * 100).toFixed(1)}%) was below the precision threshold tau.
              Displaying fallback extractive review passages below.
            </div>
          )}

          {/* Answer Body */}
          <div
            style={{
              fontSize: "15px",
              lineHeight: 1.6,
              color: "var(--text-primary)",
              whiteSpace: "pre-line",
            }}
          >
            {response.answer}
          </div>

          {/* Extractive Review Passages Card (if present) */}
          {response.retrieval_passages && response.retrieval_passages.length > 0 && (
            <div
              style={{
                marginTop: "24px",
                padding: "16px",
                background: "var(--bg-surface-low)",
                borderRadius: "8px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <h4
                style={{
                  fontSize: "14px",
                  fontWeight: 600,
                  color: "var(--text-secondary)",
                  marginBottom: "12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                  format_quote
                </span>
                Extractive Review Snippets (Passage Index)
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {response.retrieval_passages.map((p, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: "10px 14px",
                      background: "var(--bg-surface-elevated)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "6px",
                      fontSize: "13px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginBottom: "4px",
                        fontSize: "11px",
                        color: "var(--text-muted)",
                      }}
                    >
                      <span>
                        <strong>{p.title}</strong> — Field: <code>{p.field}</code>
                      </span>
                      <span>Score: {p.score.toFixed(3)}</span>
                    </div>
                    <p style={{ margin: 0, fontStyle: "italic", color: "var(--text-primary)" }}>
                      &ldquo;{p.text}&rdquo;
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Full Provenance Footer */}
          <div
            style={{
              marginTop: "24px",
              paddingTop: "14px",
              borderTop: "1px solid var(--border-subtle)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "8px",
              fontSize: "12px",
              color: "var(--text-muted)",
            }}
          >
            <div>
              <span>Tables: </span>
              {response.provenance.data_tables_read.length > 0 ? (
                response.provenance.data_tables_read.map((t, idx) => (
                  <code key={idx} style={{ marginRight: "6px" }}>
                    {t}
                  </code>
                ))
              ) : (
                <span>None (refused / abstained)</span>
              )}
            </div>

            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              <span>Run: <code>{response.provenance.model_run_id || "ephemeral"}</code></span>
              <span>Latency: <strong>{response.provenance.took_ms.toFixed(1)}ms</strong></span>
              <span
                style={{
                  color: "var(--text-secondary)",
                  background: "var(--bg-surface-low)",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  fontSize: "11px",
                }}
              >
                🔒 Zero LLM Free-Text
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
