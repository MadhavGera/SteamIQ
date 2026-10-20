"use client";

import React, { useState, useEffect, useTransition } from "react";
import {
  api,
  AssistantAskResponse,
  AssistantHealthResponse,
  GameSummary,
} from "@/lib/api";
import { useUserMode } from "@/lib/UserModeContext";
import { CandidateGameCard } from "@/components/assistant/CandidateGameCard";
import { AnalysisOverviewGrid } from "@/components/assistant/AnalysisOverviewGrid";
import { MLDiagnosticsPanel } from "@/components/assistant/MLDiagnosticsPanel";

function renderInlineMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function FormattedAnswer({ content }: { content: string }) {
  const paragraphs = content.split("\n\n");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
      {paragraphs.map((p, idx) => {
        const trimmed = p.trim();
        if (trimmed.startsWith("### ")) {
          return (
            <h3
              key={idx}
              style={{
                fontSize: "18px",
                fontWeight: 700,
                color: "var(--accent-primary)",
                margin: "4px 0 0",
              }}
            >
              {trimmed.replace(/^###\s+/, "")}
            </h3>
          );
        }
        if (trimmed.startsWith("- ")) {
          const items = trimmed.split("\n").filter((l) => l.trim().startsWith("- "));
          return (
            <ul
              key={idx}
              style={{
                margin: 0,
                paddingLeft: "20px",
                display: "flex",
                flexDirection: "column",
                gap: "6px",
                fontSize: "14px",
                lineHeight: 1.6,
                color: "var(--text-primary)",
              }}
            >
              {items.map((item, iIdx) => (
                <li
                  key={iIdx}
                  dangerouslySetInnerHTML={{
                    __html: renderInlineMarkdown(item.replace(/^-\s+/, "")),
                  }}
                />
              ))}
            </ul>
          );
        }
        return (
          <p
            key={idx}
            style={{
              margin: 0,
              fontSize: "14px",
              lineHeight: 1.65,
              color: "var(--text-primary)",
            }}
            dangerouslySetInnerHTML={{ __html: renderInlineMarkdown(trimmed) }}
          />
        );
      })}
    </div>
  );
}

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
            // HONEST-FALLBACK: Default tau fallback threshold when API is uncontactable
            tau: 0.5,
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
        // HONEST-FALLBACK: Empty array when games search response is missing
        setGameSearchResults(res?.games ?? []);
      } catch {
        setGameSearchResults([]);
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [gameSearchQuery]);

  const handleAsk = async (queryToAsk?: string) => {
    // HONEST-FALLBACK: Fallback to query input state when queryToAsk argument is omitted
    const q = (queryToAsk ?? query).trim();
    if (!q) return;

    setIsLoading(true);
    setError(null);

    try {
      const res = await api.assistantAsk({
        query: q,
        user_mode: mode,
        game_app_id: selectedGame?.app_id ?? null, // HONEST-FALLBACK: Null app_id when no game context selected
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
        "similar game to elden ring",
        "Why do players love Hollow Knight?",
        "What do players complain about in Cyberpunk 2077?",
        "Should I buy Baldur's Gate 3 now or wait for a discount?",
        "What is Terraria about?",
      ]
    : [
        "What is the market opportunity for a Roguelike Deckbuilder?",
        "similar game to elden ring",
        "What were the feature drivers of success for Elden Ring?",
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
    <div className="container" style={{ paddingBottom: "80px", maxWidth: "1100px", margin: "0 auto" }}>
      {/* Header Banner */}
      <div style={{ padding: "32px 0 20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px", flexWrap: "wrap" }}>
          <span
            className="material-symbols-outlined"
            style={{ fontSize: "24px", color: "var(--accent-primary)" }}
          >
            psychology
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
            {isPlayer ? "SteamIQ | Player mode" : "SteamIQ | Developer mode"}
          </span>
          <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.08em", color: "var(--text-muted)", textTransform: "uppercase" }}>
            GAME INTELLIGENCE · ANALYTICS · RECOMMENDATIONS
          </span>
          {health?.is_ready && (
            <span className="badge-pill badge-pill--success" style={{ fontSize: "11px", marginLeft: "auto" }}>
              Calibrated Router Online (&tau; = {health.tau})
            </span>
          )}
        </div>
        <h1 style={{ fontSize: "32px", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
          Ask SteamIQ
        </h1>
        <p style={{ fontSize: "14px", color: "var(--text-secondary)", marginTop: "6px" }}>
          Dual-stage intelligence powered by Two-Tower neural collaborative filtering and calibrated intent routing.
          Answers and recommendations are backed by pre-materialized analytical data marts.
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
                placeholder="Type to search games (e.g. Elden Ring, Hollow Knight, Cyberpunk)..."
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
                ? 'Try "similar game to elden ring", "why do players love hollow knight"...'
                : 'Try "What is the market opportunity for a Roguelike Deckbuilder?", "similar game to elden ring"...'
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
              cursor: isLoading || !query.trim() ? "not-allowed" : "pointer",
            }}
          >
            {isLoading ? (
              <span>Evaluating...</span>
            ) : (
              <>
                <span>Analyze</span>
                <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                  arrow_forward
                </span>
              </>
            )}
          </button>
        </div>

        {/* Suggested Prompts */}
        <div style={{ marginTop: "14px" }}>
          <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "8px" }}>
            Suggested Queries:
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

      {/* Loading Skeleton */}
      {isLoading && (
        <div
          className="card"
          style={{
            marginTop: "24px",
            background: "var(--bg-surface-elevated)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "12px",
            padding: "32px",
            textAlign: "center",
          }}
        >
          <div
            style={{
              display: "inline-block",
              width: "36px",
              height: "36px",
              border: "3px solid rgba(0, 210, 255, 0.2)",
              borderTopColor: "var(--accent-primary)",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
              marginBottom: "12px",
            }}
          />
          <div style={{ fontSize: "15px", fontWeight: 600, color: "var(--text-primary)" }}>
            Running Neural Inference & Candidate Retrieval...
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
            Calibrating intent confidence and querying in-memory Two-Tower embedding space.
          </div>
        </div>
      )}

      {/* Error Banner */}
      {error && !isLoading && (
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

      {/* Response Centerpiece (Answer-First Primary View) */}
      {response && !isLoading && (
        <div style={{ marginTop: "24px", display: "flex", flexDirection: "column", gap: "20px" }}>
          {/* Quick Routing Status Bar */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "10px",
              padding: "4px 0",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span className="badge-pill" style={{ background: "rgba(99, 102, 241, 0.15)", color: "#a5b4fc" }}>
                Intent: {formatIntentLabel(response.intent)}
              </span>
              <span
                className="badge-pill"
                style={{
                  background: response.confidence >= response.tau ? "rgba(16, 185, 129, 0.15)" : "rgba(245, 158, 11, 0.15)",
                  color: response.confidence >= response.tau ? "#34d399" : "#fbbf24",
                }}
              >
                {(response.confidence * 100).toFixed(1)}% Confidence
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span className="badge-pill badge-pill--neutral" style={{ fontSize: "11px" }}>
                ⚡ {response.provenance.took_ms < 0.1 && response.provenance.took_ms > 0 ? "< 0.1 ms" : `${response.provenance.took_ms.toFixed(1)} ms`}
              </span>
            </div>
          </div>

          {/* 4 Overview KPI Cards */}
          <AnalysisOverviewGrid
            marketSummary={response.market_summary}
            intent={response.intent}
            confidence={response.confidence}
            tau={response.tau}
          />

          {/* Resolved Game Context Banner (if specific game bound) */}
          {response.resolved_game && (
            <div
              className="card"
              style={{
                background: "rgba(0, 210, 255, 0.05)",
                border: "1px solid rgba(0, 210, 255, 0.25)",
                borderRadius: "10px",
                padding: "14px 18px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "10px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)", fontSize: "20px" }}>
                  sports_esports
                </span>
                <div>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)", display: "block", textTransform: "uppercase" }}>
                    Subject Game
                  </span>
                  <strong style={{ fontSize: "16px", color: "var(--text-primary)" }}>
                    {response.resolved_game.title}
                  </strong>
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                {response.resolved_game.positive_ratio !== null && response.resolved_game.positive_ratio !== undefined && (
                  <span
                    style={{
                      background: "rgba(16, 185, 129, 0.2)",
                      color: "#34d399",
                      border: "1px solid rgba(16, 185, 129, 0.4)",
                      padding: "2px 8px",
                      borderRadius: "4px",
                      fontSize: "12px",
                      fontWeight: 600,
                    }}
                  >
                    {response.resolved_game.positive_ratio}% Positive
                  </span>
                )}
                {response.resolved_game.price_usd && (
                  <span
                    style={{
                      background: "var(--bg-surface-elevated)",
                      color: "var(--text-primary)",
                      border: "1px solid var(--border-subtle)",
                      padding: "2px 8px",
                      borderRadius: "4px",
                      fontSize: "12px",
                      fontWeight: 600,
                    }}
                  >
                    {response.resolved_game.price_usd.startsWith("$")
                      ? response.resolved_game.price_usd
                      : `$${response.resolved_game.price_usd}`}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Primary Answer & Key Findings Card */}
          <div
            className="card"
            style={{
              background: "var(--bg-surface-elevated)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "12px",
              padding: "24px 28px",
              boxShadow: "0 4px 20px rgba(0, 0, 0, 0.25)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "12px" }}>
              <span className="material-symbols-outlined" style={{ fontSize: "20px", color: "var(--accent-primary)" }}>
                analytics
              </span>
              <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                Key Findings & Analytical Synthesis
              </h3>
            </div>
            <FormattedAnswer content={response.answer} />
          </div>

          {/* Candidates Grid (Structured recommendations / market peer exemplars) */}
          {response.candidates && response.candidates.length > 0 && (
            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "14px",
                }}
              >
                <h3 style={{ fontSize: "17px", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                  Catalog Peers & Market Benchmarks ({response.candidates.length} titles)
                </h3>
                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                  Neural collaborative filtering & mart similarity
                </span>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(310px, 1fr))",
                  gap: "18px",
                }}
              >
                {response.candidates.map((c) => (
                  <CandidateGameCard key={c.app_id} candidate={c} />
                ))}
              </div>
            </div>
          )}

          {/* Extractive Snippets (when available) */}
          {response.retrieval_passages && response.retrieval_passages.length > 0 && (
            <div
              className="card"
              style={{
                background: "var(--bg-surface-low)",
                borderRadius: "10px",
                border: "1px solid var(--border-subtle)",
                padding: "16px 20px",
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
                Extractive Review Intelligence Snippets
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

          {/* Collapsible ML Diagnostics & Architecture Drawer */}
          <MLDiagnosticsPanel response={response} defaultExpanded={false} />

          {/* Persistent Lineage & SLA Footer */}
          <div
            style={{
              paddingTop: "16px",
              borderTop: "1px solid var(--border-subtle)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "10px",
              fontSize: "12px",
              color: "var(--text-muted)",
            }}
          >
            <div>
              <span>Marts Read: </span>
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

            <div style={{ display: "flex", gap: "14px", alignItems: "center" }}>
              <span>
                Run ID: <code>{response.provenance.model_run_id ?? "ephemeral"}</code> {/* HONEST-FALLBACK: Fallback model run id to ephemeral */}
              </span>
              <span>
                Total Latency: <strong>{response.provenance.took_ms < 0.1 && response.provenance.took_ms > 0 ? "< 0.1 ms" : `${response.provenance.took_ms.toFixed(1)} ms`}</strong>
              </span>
              <span
                style={{
                  color: "var(--text-secondary)",
                  background: "var(--bg-surface-low)",
                  padding: "2px 8px",
                  borderRadius: "4px",
                  fontSize: "11px",
                }}
              >
                🔒 Zero LLM Hallucinations
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
