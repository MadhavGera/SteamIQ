"use client";

import React, { useEffect, useState } from "react";
import {
  api,
  type DimensionMatchScore,
  type GameMatchProfile,
  type GameMatchResult,
  type UserMatchPreferences,
} from "@/lib/api";

interface MatchTabProps {
  appId: number;
  gameTitle: string;
}

const PRESETS: Array<{ label: string; icon: string; prefs: UserMatchPreferences }> = [
  {
    label: "Hardcore Challenger",
    icon: "swords",
    prefs: { difficulty: 9.0, combat: 8.5, story_weight: 5.0, exploration: 7.0, multiplayer: 3.0, session_length: 8.0 },
  },
  {
    label: "Cozy Story Lover",
    icon: "menu_book",
    prefs: { difficulty: 2.0, combat: 1.0, story_weight: 9.5, exploration: 7.0, multiplayer: 0.0, session_length: 5.0 },
  },
  {
    label: "Open-World Explorer",
    icon: "explore",
    prefs: { difficulty: 5.0, combat: 6.0, story_weight: 7.0, exploration: 9.5, multiplayer: 2.0, session_length: 8.5 },
  },
  {
    label: "Solo Action Gamer",
    icon: "bolt",
    prefs: { difficulty: 7.0, combat: 9.0, story_weight: 4.0, exploration: 5.0, multiplayer: 0.0, session_length: 6.0 },
  },
];

export function MatchTab({ appId, gameTitle }: MatchTabProps) {
  const [profile, setProfile] = useState<GameMatchProfile | null>(null);
  const [prefs, setPrefs] = useState<UserMatchPreferences>({
    difficulty: 5.0,
    story_weight: 5.0,
    exploration: 5.0,
    combat: 5.0,
    multiplayer: 0.0,
    session_length: 5.0,
  });
  const [matchResult, setMatchResult] = useState<GameMatchResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [evaluating, setEvaluating] = useState<boolean>(false);

  // 1. Initial Load: Fetch Profile
  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      setLoading(true);
      try {
        const p = await api.getMatchProfile(appId);
        if (isMounted) {
          setProfile(p);
          // Set initial preferences to match baseline or defaults
          const initialPrefs: UserMatchPreferences = {
            difficulty: p.difficulty,
            story_weight: p.story_weight,
            exploration: p.exploration,
            combat: p.combat,
            multiplayer: p.multiplayer,
            session_length: p.session_length,
          };
          setPrefs(initialPrefs);
          const res = await api.matchGame(appId, initialPrefs);
          if (isMounted) setMatchResult(res);
        }
      } catch (err) {
        console.error("Failed to load match profile:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, [appId]);

  // 2. Preference Change Handler
  const handlePrefChange = async (key: keyof UserMatchPreferences, value: number) => {
    const updated = { ...prefs, [key]: value };
    setPrefs(updated);
    setEvaluating(true);
    try {
      const res = await api.matchGame(appId, updated);
      setMatchResult(res);
    } catch (err) {
      console.error("Match calculation failed:", err);
    } finally {
      setEvaluating(false);
    }
  };

  const applyPreset = async (presetPrefs: UserMatchPreferences) => {
    setPrefs(presetPrefs);
    setEvaluating(true);
    try {
      const res = await api.matchGame(appId, presetPrefs);
      setMatchResult(res);
    } catch (err) {
      console.error("Match calculation failed:", err);
    } finally {
      setEvaluating(false);
    }
  };

  if (loading) {
    return (
      <div className="card" style={{ padding: "64px 24px", textAlign: "center", marginTop: "24px" }}>
        <div className="loading-bars" style={{ margin: "0 auto 16px" }}>
          <span />
          <span />
          <span />
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
          Loading game intensity DNA for {gameTitle}...
        </p>
      </div>
    );
  }

  const matchPct = matchResult?.overall_match_pct ?? 75.0;
  const verdict = matchResult?.match_verdict ?? "Strong Match";

  const getVerdictColor = (score: number) => {
    if (score >= 85) return "var(--success)";
    if (score >= 70) return "var(--accent-primary)";
    if (score >= 50) return "var(--warning)";
    return "var(--danger)";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "24px" }}>
      {/* ── 1. Hero Match Dial & Verdict ── */}
      <div
        className="card card--raised"
        style={{
          background: "linear-gradient(135deg, rgba(94, 194, 240, 0.08) 0%, rgba(19, 31, 46, 0.95) 100%)",
          border: `1px solid ${getVerdictColor(matchPct)}40`,
          padding: "32px",
        }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "24px" }}>
          <div style={{ maxWidth: "600px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
              <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)", fontSize: "20px" }}>
                auto_awesome
              </span>
              <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.08em", color: "var(--accent-primary)", textTransform: "uppercase" }}>
                PLAYER TASTE ALIGNMENT
              </span>
            </div>
            <h2 style={{ fontSize: "26px", fontWeight: 800, margin: "0 0 10px", color: "var(--text-primary)" }}>
              Is {gameTitle} for You?
            </h2>
            <p style={{ fontSize: "14px", color: "var(--text-secondary)", lineHeight: "1.6", margin: 0 }}>
              Adjust the 6 gameplay intensity sliders below to match your playstyle. SteamIQ compares your preferences against {gameTitle}&apos;s verified NLP topic taxonomy and store tags.
            </p>

            {profile?.profile_summary?.primary_traits && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginTop: "16px" }}>
                {profile.profile_summary.primary_traits.map((trait, idx) => (
                  <span
                    key={idx}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "4px",
                      padding: "4px 10px",
                      borderRadius: "6px",
                      backgroundColor: "var(--bg-surface)",
                      border: "1px solid var(--border-subtle)",
                      fontSize: "12px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "14px", color: "var(--accent-light)" }}>
                      verified
                    </span>
                    {trait}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Match Score Badge Dial */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "24px 36px",
              backgroundColor: "var(--bg-base)",
              borderRadius: "16px",
              border: `1px solid ${getVerdictColor(matchPct)}`,
              minWidth: "180px",
              textAlign: "center",
              boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
            }}
          >
            <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.05em" }}>
              MATCH SCORE
            </span>
            <div style={{ fontSize: "44px", fontWeight: 800, color: getVerdictColor(matchPct), lineHeight: "1.1", margin: "4px 0" }}>
              {Math.round(matchPct)}%
            </div>
            <div
              style={{
                fontSize: "13px",
                fontWeight: 700,
                color: getVerdictColor(matchPct),
                backgroundColor: `${getVerdictColor(matchPct)}15`,
                padding: "3px 10px",
                borderRadius: "999px",
                marginTop: "4px",
              }}
            >
              {verdict}
            </div>
            {evaluating && (
              <span style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "6px" }}>
                Recalculating...
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ── 2. Playstyle Presets ── */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
          <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "var(--text-muted)" }}>
            tune
          </span>
          <span style={{ fontSize: "12px", fontWeight: 700, letterSpacing: "0.05em", color: "var(--text-muted)", textTransform: "uppercase" }}>
            QUICK PRESETS
          </span>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => applyPreset(p.prefs)}
              className="btn btn--secondary"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 16px",
                fontSize: "13px",
                borderRadius: "8px",
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "var(--accent-primary)" }}>
                {p.icon}
              </span>
              <span>{p.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── 3. Interactive Dimension Sliders & Game Alignment ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "20px" }}>
        {/* Slider Controls */}
        <div className="card" style={{ padding: "24px" }}>
          <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)", fontSize: "20px" }}>
              sliders
            </span>
            <span>Your Taste Preferences (0 – 10)</span>
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {[
              { key: "difficulty", label: "Challenge & Difficulty", icon: "psychology", desc: "0 = Relaxing / Cozy → 10 = Brutal Souls-like" },
              { key: "story_weight", label: "Narrative & Story Depth", icon: "auto_stories", desc: "0 = Pure Gameplay → 10 = Deep Lore / Visual Novel" },
              { key: "exploration", label: "World Exploration", icon: "explore", desc: "0 = Linear / Level-based → 10 = Massive Open World" },
              { key: "combat", label: "Combat Intensity", icon: "sports_martial_arts", desc: "0 = Puzzle / Peaceful → 10 = Fast-Paced Action / FPS" },
              { key: "multiplayer", label: "Multiplayer Focus", icon: "groups", desc: "0 = Strictly Solo → 10 = Heavy Co-op / Competitive" },
              { key: "session_length", label: "Session Length & Scope", icon: "schedule", desc: "0 = Short / Bite-sized → 10 = Epic 40h+ Campaign" },
            ].map((dim) => {
              const currentVal = prefs[dim.key as keyof UserMatchPreferences] ?? 5.0;
              return (
                <div key={dim.key} style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <label
                      htmlFor={`slider-${dim.key}`}
                      style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "6px" }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "var(--accent-secondary)" }}>
                        {dim.icon}
                      </span>
                      {dim.label}
                    </label>
                    <span style={{ fontSize: "14px", fontWeight: 700, color: "var(--accent-primary)", fontFamily: "var(--font-mono)" }}>
                      {currentVal.toFixed(1)} / 10
                    </span>
                  </div>
                  <input
                    id={`slider-${dim.key}`}
                    type="range"
                    min="0"
                    max="10"
                    step="0.5"
                    value={currentVal}
                    onChange={(e) => handlePrefChange(dim.key as keyof UserMatchPreferences, parseFloat(e.target.value))}
                    style={{
                      width: "100%",
                      accentColor: "var(--accent-primary)",
                      cursor: "pointer",
                      height: "6px",
                      borderRadius: "4px",
                    }}
                  />
                  <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>{dim.desc}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dimension Breakdown & Alignment */}
        <div className="card" style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "20px" }}>
          <h3 style={{ fontSize: "16px", fontWeight: 700, margin: 0, display: "flex", alignItems: "center", gap: "8px" }}>
            <span className="material-symbols-outlined" style={{ color: "var(--success)", fontSize: "20px" }}>
              analytics
            </span>
            <span>Dimension Alignment vs {gameTitle}</span>
          </h3>

          {matchResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              {Object.entries(matchResult.dimension_scores).map(([key, score]: [string, DimensionMatchScore]) => {
                const isClose = score.delta <= 1.5;
                const isFar = score.delta >= 3.5;
                return (
                  <div key={key} style={{ padding: "12px 14px", borderRadius: "8px", backgroundColor: "var(--bg-surface-raised)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                      <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>
                        {score.dimension}
                      </span>
                      <span
                        style={{
                          fontSize: "12px",
                          fontWeight: 700,
                          color: isClose ? "var(--success)" : isFar ? "var(--danger)" : "var(--accent-primary)",
                        }}
                      >
                        {Math.round(score.dimension_match_pct)}% Match (Δ {score.delta.toFixed(1)})
                      </span>
                    </div>

                    {/* Dual Bar Comparison */}
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px", fontSize: "11px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ width: "80px", color: "var(--text-muted)" }}>Game DNA:</span>
                        <div style={{ flex: 1, height: "6px", backgroundColor: "var(--border-subtle)", borderRadius: "3px", overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${score.game_intensity * 10}%`,
                              height: "100%",
                              backgroundColor: "var(--accent-secondary)",
                              borderRadius: "3px",
                            }}
                          />
                        </div>
                        <span style={{ width: "35px", textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                          {score.game_intensity.toFixed(1)}
                        </span>
                      </div>

                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ width: "80px", color: "var(--text-muted)" }}>Your Taste:</span>
                        <div style={{ flex: 1, height: "6px", backgroundColor: "var(--border-subtle)", borderRadius: "3px", overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${score.user_preference * 10}%`,
                              height: "100%",
                              backgroundColor: isClose ? "var(--success)" : isFar ? "var(--danger)" : "var(--accent-primary)",
                              borderRadius: "3px",
                            }}
                          />
                        </div>
                        <span style={{ width: "35px", textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--text-primary)", fontWeight: 700 }}>
                          {score.user_preference.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Highlights & Friction Points */}
          {matchResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: "12px", borderTop: "1px solid var(--border-subtle)", paddingTop: "16px" }}>
              {matchResult.alignment_highlights.length > 0 && (
                <div>
                  <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--success)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    WHAT YOU&apos;LL LOVE
                  </span>
                  <ul style={{ margin: "6px 0 0", paddingLeft: "18px", fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
                    {matchResult.alignment_highlights.map((h, idx) => (
                      <li key={idx}>{h}</li>
                    ))}
                  </ul>
                </div>
              )}

              {matchResult.friction_points.length > 0 && (
                <div>
                  <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--warning)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    THINGS TO NOTE
                  </span>
                  <ul style={{ margin: "6px 0 0", paddingLeft: "18px", fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
                    {matchResult.friction_points.map((f, idx) => (
                      <li key={idx}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── 4. Privacy & Integrity Notice ── */}
      <div
        style={{
          padding: "12px 18px",
          borderRadius: "8px",
          backgroundColor: "var(--bg-surface)",
          border: "1px solid var(--border-subtle)",
          fontSize: "12px",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "var(--accent-primary)" }}>
          lock
        </span>
        <span>
          <strong>Grounded Telemetry:</strong> Match scores are calculated in-memory against pre-materialized game profile metrics. Your slider preferences are never tracked or saved to the server.
        </span>
      </div>
    </div>
  );
}
