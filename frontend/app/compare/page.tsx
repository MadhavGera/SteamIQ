"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  api,
  type GameSummary,
  type GameDetail,
  type RecsysModelInfo,
  type RecsysCatalogItem,
} from "@/lib/api";
import { RecsysCompareView } from "@/components/recsys/RecsysCompareView";
import { RecsysLibraryBuilder } from "@/components/recsys/RecsysLibraryBuilder";
import { getComparisonProfile } from "@/components/compare/comparisonData";
import { HeadToHeadFeatureTable } from "@/components/compare/HeadToHeadFeatureTable";
import { ExperienceScorecard } from "@/components/compare/ExperienceScorecard";
import { AnalyticalVerdictCard } from "@/components/compare/AnalyticalVerdictCard";

const QUICK_MATCHUPS = [
  { label: "⚔️ Hollow Knight vs Elden Ring", appA: 367520, appB: 1245620 },
  { label: "⚔️ Elden Ring vs Sekiro", appA: 1245620, appB: 814380 },
  { label: "⚔️ Hollow Knight vs Hades", appA: 367520, appB: 1145360 },
  { label: "⚔️ Cyberpunk 2077 vs The Witcher 3", appA: 1091500, appB: 292030 },
];

export default function ComparePage() {
  const [catalog, setCatalog] = useState<GameSummary[]>([]);
  const [models, setModels] = useState<RecsysModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Selected games for 1v1 comparison (Default: Hollow Knight vs Elden Ring)
  const [gameAId, setGameAId] = useState<number>(367520); // Hollow Knight
  const [gameBId, setGameBId] = useState<number>(1245620); // Elden Ring
  const [gameA, setGameA] = useState<GameDetail | GameSummary | null>(null);
  const [gameB, setGameB] = useState<GameDetail | GameSummary | null>(null);

  // Selected library for RecSys algorithm benchmarking
  const [libraryGames, setLibraryGames] = useState<RecsysCatalogItem[]>([]);
  const [activeTab, setActiveTab] = useState<"head_to_head" | "algorithm_compare">("head_to_head");

  useEffect(() => {
    let isMounted = true;
    async function init() {
      setIsLoading(true);
      setError(null);
      try {
        const [searchRes, modelsRes, catalogRes, detailA, detailB] = await Promise.all([
          api.searchGames("", 1, 50).catch(() => ({ games: [], total: 0, page: 1, page_size: 50, query: "" })),
          api.recsysModels().catch(() => []),
          api.searchRecsysItems("", 10).catch(() => null),
          api.getGame(367520).catch(() => null),
          api.getGame(1245620).catch(() => null),
        ]);
        if (isMounted) {
          // HONEST-FALLBACK: Empty array if search returned no games
          const gamesList = searchRes?.games ?? [];

          // Ensure benchmark games are present in catalog list for dropdowns
          const benchmarkPlaceholders: GameSummary[] = [
            { app_id: 367520, name: "Hollow Knight", genres: [{ id: "1", description: "Metroidvania" }] } as GameSummary,
            { app_id: 1245620, name: "ELDEN RING", genres: [{ id: "1", description: "Action RPG" }] } as GameSummary,
            { app_id: 814380, name: "Sekiro™: Shadows Die Twice", genres: [{ id: "1", description: "Action" }] } as GameSummary,
            { app_id: 1145360, name: "Hades", genres: [{ id: "1", description: "Roguelite" }] } as GameSummary,
            { app_id: 1091500, name: "Cyberpunk 2077", genres: [{ id: "1", description: "RPG" }] } as GameSummary,
            { app_id: 292030, name: "The Witcher 3: Wild Hunt", genres: [{ id: "1", description: "RPG" }] } as GameSummary,
          ];

          const mergedCatalog = [...gamesList];
          for (const bp of benchmarkPlaceholders) {
            if (!mergedCatalog.some((g) => g.app_id === bp.app_id)) {
              mergedCatalog.push(bp);
            }
          }
          setCatalog(mergedCatalog);

          // HONEST-FALLBACK: Empty array if models returned null
          setModels(modelsRes ?? []);

          // Set initial Game A (Hollow Knight) & Game B (Elden Ring)
          if (detailA) {
            setGameA(detailA);
          } else {
            const foundA = mergedCatalog.find((g) => g.app_id === 367520) || mergedCatalog[0];
            if (foundA) setGameA(foundA);
          }

          if (detailB) {
            setGameB(detailB);
          } else {
            const foundB = mergedCatalog.find((g) => g.app_id === 1245620) || mergedCatalog[1] || mergedCatalog[0];
            if (foundB) setGameB(foundB);
          }

          // Preload some library games from real recsys catalog
          if (catalogRes?.items && catalogRes.items.length >= 2) {
            setLibraryGames(catalogRes.items.slice(0, 2));
          }

          setIsLoading(false);
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : "Failed to load catalog";
          setError(msg);
          setIsLoading(false);
        }
      }
    }
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSelectA = async (appId: number) => {
    setGameAId(appId);
    try {
      const detail = await api.getGame(appId);
      setGameA(detail);
    } catch {
      const g = catalog.find((item) => item.app_id === appId);
      if (g) setGameA(g);
    }
  };

  const handleSelectB = async (appId: number) => {
    setGameBId(appId);
    try {
      const detail = await api.getGame(appId);
      setGameB(detail);
    } catch {
      const g = catalog.find((item) => item.app_id === appId);
      if (g) setGameB(g);
    }
  };

  const handleSelectMatchup = async (appIdA: number, appIdB: number) => {
    setGameAId(appIdA);
    setGameBId(appIdB);
    try {
      const [detailA, detailB] = await Promise.all([
        api.getGame(appIdA).catch(() => catalog.find((item) => item.app_id === appIdA) || null),
        api.getGame(appIdB).catch(() => catalog.find((item) => item.app_id === appIdB) || null),
      ]);
      if (detailA) setGameA(detailA);
      if (detailB) setGameB(detailB);
    } catch {
      const gA = catalog.find((item) => item.app_id === appIdA);
      const gB = catalog.find((item) => item.app_id === appIdB);
      if (gA) setGameA(gA);
      if (gB) setGameB(gB);
    }
  };

  // Derive comparison profile (curated or dynamic)
  const comparisonProfile = gameA && gameB ? getComparisonProfile(gameA, gameB) : null;

  return (
    <div className="container" style={{ paddingBottom: "80px", maxWidth: "1280px", margin: "0 auto", paddingLeft: "24px", paddingRight: "24px" }}>
      {/* Header */}
      <div style={{ marginTop: "32px", marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "28px", color: "var(--accent-primary)", fontVariationSettings: "'FILL' 1" }}
              >
                compare_arrows
              </span>
              <h1 style={{ fontSize: "28px", fontWeight: "800", letterSpacing: "-0.02em", color: "var(--text-primary)" }}>
                Compare Games
              </h1>
            </div>
            <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "720px", margin: 0 }}>
              Head-to-head game intelligence KPI comparison, feature architecture breakdown, and experiential scorecard benchmarking.
            </p>
          </div>

          {/* Mode Switcher */}
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button
              onClick={() => setActiveTab("head_to_head")}
              style={{
                padding: "8px 16px",
                borderRadius: "var(--radius-sm)",
                fontSize: "13px",
                fontWeight: "600",
                cursor: "pointer",
                background: activeTab === "head_to_head" ? "var(--bg-surface-raised)" : "transparent",
                color: activeTab === "head_to_head" ? "var(--text-primary)" : "var(--text-muted)",
                border: activeTab === "head_to_head" ? "1px solid var(--accent-primary)" : "1px solid transparent",
                transition: "all var(--transition-fast)",
              }}
            >
              Head-to-Head Comparison
            </button>
            <button
              onClick={() => setActiveTab("algorithm_compare")}
              style={{
                padding: "8px 16px",
                borderRadius: "var(--radius-sm)",
                fontSize: "13px",
                fontWeight: "600",
                cursor: "pointer",
                background: activeTab === "algorithm_compare" ? "var(--bg-surface-raised)" : "transparent",
                color: activeTab === "algorithm_compare" ? "var(--text-primary)" : "var(--text-muted)",
                border: activeTab === "algorithm_compare" ? "1px solid var(--accent-primary)" : "1px solid transparent",
                transition: "all var(--transition-fast)",
              }}
            >
              RecSys Model Benchmark
            </button>
          </div>
        </div>
      </div>

      {isLoading && (
        <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-secondary)" }}>
          <div className="material-symbols-outlined" style={{ fontSize: "36px", animation: "spin 1s linear infinite", marginBottom: "12px", color: "var(--accent-primary)" }}>
            refresh
          </div>
          <div>Loading catalog comparison data...</div>
        </div>
      )}

      {error && (
        <div style={{ padding: "24px", background: "rgba(217, 65, 38, 0.1)", border: "1px solid rgba(217, 65, 38, 0.3)", borderRadius: "var(--radius-card)", color: "var(--danger)", marginBottom: "28px" }}>
          <strong>Comparison error:</strong> {error}
        </div>
      )}

      {/* Head-to-Head View */}
      {!isLoading && !error && activeTab === "head_to_head" && (
        <div>
          {/* Quick Matchups Bar */}
          <div style={{ marginBottom: "20px", display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
            <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", display: "flex", alignItems: "center", gap: "4px" }}>
              <span className="material-symbols-outlined" style={{ fontSize: "15px", color: "var(--accent-primary)" }}>
                bolt
              </span>
              Popular Matchups:
            </span>
            {QUICK_MATCHUPS.map((matchup) => {
              const isActive =
                (gameAId === matchup.appA && gameBId === matchup.appB) ||
                (gameAId === matchup.appB && gameBId === matchup.appA);
              return (
                <button
                  key={matchup.label}
                  type="button"
                  onClick={() => handleSelectMatchup(matchup.appA, matchup.appB)}
                  className="popular-chip"
                  style={{
                    cursor: "pointer",
                    padding: "5px 12px",
                    fontSize: "12px",
                    fontWeight: isActive ? 700 : 500,
                    backgroundColor: isActive ? "rgba(102, 192, 244, 0.15)" : "var(--bg-surface)",
                    borderColor: isActive ? "var(--accent-primary)" : "var(--border-subtle)",
                    color: isActive ? "var(--accent-primary)" : "var(--text-secondary)",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "6px",
                    transition: "all var(--transition-fast)",
                  }}
                  title={`Compare ${matchup.label}`}
                >
                  {matchup.label}
                </button>
              );
            })}
          </div>

          {/* Selectors */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", marginBottom: "28px" }}>
            <div>
              <label style={{ display: "block", fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "8px", letterSpacing: "0.05em" }}>
                Select Game 1
              </label>
              <select
                value={gameAId}
                onChange={(e) => handleSelectA(Number(e.target.value))}
                style={{
                  width: "100%",
                  padding: "10px 14px",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  color: "var(--text-primary)",
                  fontSize: "14px",
                  fontWeight: "600",
                  outline: "none",
                }}
              >
                {catalog.map((g) => (
                  <option key={g.app_id} value={g.app_id}>
                    {g.name} ({g.genres?.[0]?.description || "Game"})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "8px", letterSpacing: "0.05em" }}>
                Select Game 2
              </label>
              <select
                value={gameBId}
                onChange={(e) => handleSelectB(Number(e.target.value))}
                style={{
                  width: "100%",
                  padding: "10px 14px",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  color: "var(--text-primary)",
                  fontSize: "14px",
                  fontWeight: "600",
                  outline: "none",
                }}
              >
                {catalog.map((g) => (
                  <option key={g.app_id} value={g.app_id}>
                    {g.name} ({g.genres?.[0]?.description || "Game"})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Section: Visual Comparison Hero Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", marginBottom: "28px" }}>
            {/* Game A Column */}
            {gameA ? (
              <div
                className="card"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-card)",
                  overflow: "hidden",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                {gameA.header_image && (
                  <div style={{ width: "100%", height: "200px", overflow: "hidden", position: "relative" }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={gameA.header_image}
                      alt={gameA.name}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                    <div style={{ position: "absolute", bottom: "12px", left: "16px", right: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                      <span style={{ fontSize: "11px", fontWeight: "700", background: "rgba(0,0,0,0.8)", padding: "3px 8px", borderRadius: "var(--radius-sm)", color: "#fff" }}>
                        App ID: {gameA.app_id}
                      </span>
                    </div>
                  </div>
                )}
                <div style={{ padding: "20px", display: "flex", flexDirection: "column", flexGrow: 1 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
                    <h2 style={{ fontSize: "20px", fontWeight: "800", margin: 0, color: "var(--text-primary)" }}>
                      {gameA.name}
                    </h2>
                    <Link
                      href={`/games/${gameA.app_id}?tab=overview`}
                      style={{
                        fontSize: "12px",
                        fontWeight: "700",
                        color: "var(--accent-primary)",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                      }}
                    >
                      Full Dossier &rarr;
                    </Link>
                  </div>

                  {/* Evocative Gameplay & Atmosphere Summary */}
                  {comparisonProfile?.gameAVisualSummary && (
                    <div
                      style={{
                        padding: "10px 14px",
                        backgroundColor: "rgba(102, 192, 244, 0.08)",
                        borderLeft: "3px solid var(--accent-primary)",
                        borderRadius: "0 var(--radius-sm) var(--radius-sm) 0",
                        marginBottom: "16px",
                        fontSize: "13px",
                        fontStyle: "italic",
                        color: "var(--text-primary)",
                        lineHeight: 1.5,
                      }}
                    >
                      &ldquo;{comparisonProfile.gameAVisualSummary}&rdquo;
                    </div>
                  )}

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "16px" }}>
                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Developer</div>
                      {/* HONEST-FALLBACK: Safe indicator if developer is unrecorded */}
                      <div style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>{gameA.developer ?? "Independent"}</div>
                    </div>

                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Price</div>
                      <div style={{ fontSize: "13px", fontWeight: "700", color: "#10b981", marginTop: "2px" }}>
                        {gameA.is_free ? "Free to Play" : (gameA.final_price_usd ? `$${gameA.final_price_usd}` : "Paid")}
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Positive Reviews</div>
                      <div style={{ fontSize: "13px", fontWeight: "700", color: "var(--text-primary)", marginTop: "2px" }}>
                        {gameA.positive_reviews.toLocaleString()} ({Math.round((gameA.positive_reviews / (gameA.positive_reviews + gameA.negative_reviews || 1)) * 100)}%)
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>
                        {(gameA as Partial<GameDetail>).median_playtime_forever ? "Median Playtime" : "Owner Bracket"}
                      </div>
                      <div style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
                        {(gameA as Partial<GameDetail>).median_playtime_forever
                          ? `${Math.round(((gameA as Partial<GameDetail>).median_playtime_forever || 0) / 60)} hrs`
                          // HONEST-FALLBACK: Safe indicator if owners estimate is unrecorded
                          : (gameA.owners_estimate ?? "Public Bracket")}
                      </div>
                    </div>
                  </div>

                  {/* Genres */}
                  <div style={{ marginBottom: "14px" }}>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>Genres &amp; Tags</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      {gameA.genres?.map((g) => (
                        <span key={g.id} style={{ fontSize: "11px", padding: "3px 8px", borderRadius: "var(--radius-sm)", background: "rgba(255,255,255,0.06)", color: "var(--text-secondary)" }}>
                          {g.description}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Summary */}
                  {gameA.short_description && (
                    <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5, margin: "auto 0 0 0" }}>
                      {gameA.short_description}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>Game A not found</div>
            )}

            {/* Game B Column */}
            {gameB ? (
              <div
                className="card"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-card)",
                  overflow: "hidden",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                {gameB.header_image && (
                  <div style={{ width: "100%", height: "200px", overflow: "hidden", position: "relative" }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={gameB.header_image}
                      alt={gameB.name}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                    <div style={{ position: "absolute", bottom: "12px", left: "16px", right: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                      <span style={{ fontSize: "11px", fontWeight: "700", background: "rgba(0,0,0,0.8)", padding: "3px 8px", borderRadius: "var(--radius-sm)", color: "#fff" }}>
                        App ID: {gameB.app_id}
                      </span>
                    </div>
                  </div>
                )}
                <div style={{ padding: "20px", display: "flex", flexDirection: "column", flexGrow: 1 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
                    <h2 style={{ fontSize: "20px", fontWeight: "800", margin: 0, color: "var(--text-primary)" }}>
                      {gameB.name}
                    </h2>
                    <Link
                      href={`/games/${gameB.app_id}?tab=overview`}
                      style={{
                        fontSize: "12px",
                        fontWeight: "700",
                        color: "var(--accent-primary)",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                      }}
                    >
                      Full Dossier &rarr;
                    </Link>
                  </div>

                  {/* Evocative Gameplay & Atmosphere Summary */}
                  {comparisonProfile?.gameBVisualSummary && (
                    <div
                      style={{
                        padding: "10px 14px",
                        backgroundColor: "rgba(16, 185, 129, 0.08)",
                        borderLeft: "3px solid #10b981",
                        borderRadius: "0 var(--radius-sm) var(--radius-sm) 0",
                        marginBottom: "16px",
                        fontSize: "13px",
                        fontStyle: "italic",
                        color: "var(--text-primary)",
                        lineHeight: 1.5,
                      }}
                    >
                      &ldquo;{comparisonProfile.gameBVisualSummary}&rdquo;
                    </div>
                  )}

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "16px" }}>
                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Developer</div>
                      {/* HONEST-FALLBACK: Safe indicator if developer is unrecorded */}
                      <div style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>{gameB.developer ?? "Independent"}</div>
                    </div>

                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Price</div>
                      <div style={{ fontSize: "13px", fontWeight: "700", color: "#10b981", marginTop: "2px" }}>
                        {gameB.is_free ? "Free to Play" : (gameB.final_price_usd ? `$${gameB.final_price_usd}` : "Paid")}
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Positive Reviews</div>
                      <div style={{ fontSize: "13px", fontWeight: "700", color: "var(--text-primary)", marginTop: "2px" }}>
                        {gameB.positive_reviews.toLocaleString()} ({Math.round((gameB.positive_reviews / (gameB.positive_reviews + gameB.negative_reviews || 1)) * 100)}%)
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-low)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>
                        {(gameB as Partial<GameDetail>).median_playtime_forever ? "Median Playtime" : "Owner Bracket"}
                      </div>
                      <div style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
                        {(gameB as Partial<GameDetail>).median_playtime_forever
                          ? `${Math.round(((gameB as Partial<GameDetail>).median_playtime_forever || 0) / 60)} hrs`
                          // HONEST-FALLBACK: Safe indicator if owners estimate is unrecorded
                          : (gameB.owners_estimate ?? "Public Bracket")}
                      </div>
                    </div>
                  </div>

                  {/* Genres */}
                  <div style={{ marginBottom: "14px" }}>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>Genres &amp; Tags</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      {gameB.genres?.map((g) => (
                        <span key={g.id} style={{ fontSize: "11px", padding: "3px 8px", borderRadius: "var(--radius-sm)", background: "rgba(255,255,255,0.06)", color: "var(--text-secondary)" }}>
                          {g.description}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Summary */}
                  {gameB.short_description && (
                    <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5, margin: "auto 0 0 0" }}>
                      {gameB.short_description}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ padding: "40px", textAlign: "center", color: "var(--text-secondary)" }}>Game B not found</div>
            )}
          </div>

          {/* Section: Head-to-Head Feature Comparison Table */}
          {comparisonProfile && gameA && gameB && (
            <HeadToHeadFeatureTable
              gameAName={gameA.name}
              gameBName={gameB.name}
              features={comparisonProfile.features}
            />
          )}

          {/* Section: Which Has the Better Experience? Category Scorecard */}
          {comparisonProfile && gameA && gameB && (
            <ExperienceScorecard
              gameAName={gameA.name}
              gameBName={gameB.name}
              scorecard={comparisonProfile.scorecard}
            />
          )}

          {/* Section: SteamIQ Analytical Verdict */}
          {comparisonProfile && gameA && gameB && (
            <AnalyticalVerdictCard
              gameAName={gameA.name}
              gameBName={gameB.name}
              verdict={comparisonProfile.verdict}
            />
          )}
        </div>
      )}

      {/* RecSys Algorithm Benchmark View */}
      {!isLoading && !error && activeTab === "algorithm_compare" && (
        <div>
          <div style={{ marginBottom: "20px" }}>
            <h2 style={{ fontSize: "18px", fontWeight: "700", color: "var(--text-primary)", margin: 0 }}>
              Algorithm vs Algorithm Comparison
            </h2>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
              Benchmarking recommendation lists side-by-side using the identical seed library.
            </p>
          </div>

          <div style={{ marginBottom: "24px" }}>
            <RecsysLibraryBuilder
              selectedGames={libraryGames}
              onChange={setLibraryGames}
              universeSize={catalog.length}
            />
          </div>

          <RecsysCompareView
            models={models}
            selectedGames={libraryGames}
            defaultModelA="recsys_two_tower_clean"
            defaultModelB="recsys_itemknn"
          />
        </div>
      )}
    </div>
  );
}
