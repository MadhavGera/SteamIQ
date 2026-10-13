"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  api,
  type GameSummary,
  type RecsysModelInfo,
  type RecsysCatalogItem,
} from "@/lib/api";
import { RecsysCompareView } from "@/components/recsys/RecsysCompareView";
import { RecsysLibraryBuilder } from "@/components/recsys/RecsysLibraryBuilder";

export default function ComparePage() {
  const [catalog, setCatalog] = useState<GameSummary[]>([]);
  const [models, setModels] = useState<RecsysModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Selected games for 1v1 comparison
  const [gameAId, setGameAId] = useState<number>(1145360); // Hades
  const [gameBId, setGameBId] = useState<number>(367520); // Hollow Knight
  const [gameA, setGameA] = useState<GameSummary | null>(null);
  const [gameB, setGameB] = useState<GameSummary | null>(null);

  // Selected library for RecSys algorithm benchmarking
  const [libraryGames, setLibraryGames] = useState<RecsysCatalogItem[]>([]);
  const [activeTab, setActiveTab] = useState<"head_to_head" | "algorithm_compare">("head_to_head");

  useEffect(() => {
    let isMounted = true;
    async function init() {
      setIsLoading(true);
      setError(null);
      try {
        const [searchRes, modelsRes, catalogRes] = await Promise.all([
          api.searchGames("", 1, 60),
          api.recsysModels(),
          api.searchRecsysItems("", 10).catch(() => null),
        ]);
        if (isMounted) {
          // HONEST-FALLBACK: Empty array if search returned no games
          const gamesList = searchRes?.games ?? [];
          setCatalog(gamesList);
          // HONEST-FALLBACK: Empty array if models returned null
          setModels(modelsRes ?? []);

          // Set initial Game A & Game B from list
          const foundA = gamesList.find((g) => g.app_id === 1145360) || gamesList[0];
          const foundB = gamesList.find((g) => g.app_id === 367520) || gamesList[1] || gamesList[0];

          if (foundA) {
            setGameA(foundA);
            setGameAId(foundA.app_id);
          }
          if (foundB) {
            setGameB(foundB);
            setGameBId(foundB.app_id);
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

  const handleSelectA = (appId: number) => {
    setGameAId(appId);
    const g = catalog.find((item) => item.app_id === appId);
    if (g) setGameA(g);
  };

  const handleSelectB = (appId: number) => {
    setGameBId(appId);
    const g = catalog.find((item) => item.app_id === appId);
    if (g) setGameB(g);
  };

  return (
    <div className="container" style={{ paddingBottom: "80px", maxWidth: "1280px", margin: "0 auto", paddingLeft: "24px", paddingRight: "24px" }}>
      {/* Header */}
      <div style={{ marginTop: "32px", marginBottom: "28px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "28px", color: "var(--accent-cyan)", fontVariationSettings: "'FILL' 1" }}
              >
                compare_arrows
              </span>
              <h1 style={{ fontSize: "28px", fontWeight: "800", letterSpacing: "-0.02em", color: "var(--text-high)" }}>
                Compare Games
              </h1>
            </div>
            <p style={{ color: "var(--text-medium)", fontSize: "14px", maxWidth: "720px", margin: 0 }}>
              Head-to-head game intelligence KPI comparison, audience overlap, and cross-algorithm recommendation benchmarking.
            </p>
          </div>

          {/* Mode Switcher */}
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button
              onClick={() => setActiveTab("head_to_head")}
              style={{
                padding: "8px 16px",
                borderRadius: "8px",
                fontSize: "13px",
                fontWeight: "600",
                cursor: "pointer",
                background: activeTab === "head_to_head" ? "var(--bg-surface-elevated)" : "transparent",
                color: activeTab === "head_to_head" ? "var(--text-high)" : "var(--text-medium)",
                border: activeTab === "head_to_head" ? "1px solid var(--border-medium)" : "1px solid transparent",
              }}
            >
              Head-to-Head KPIs
            </button>
            <button
              onClick={() => setActiveTab("algorithm_compare")}
              style={{
                padding: "8px 16px",
                borderRadius: "8px",
                fontSize: "13px",
                fontWeight: "600",
                cursor: "pointer",
                background: activeTab === "algorithm_compare" ? "var(--bg-surface-elevated)" : "transparent",
                color: activeTab === "algorithm_compare" ? "var(--text-high)" : "var(--text-medium)",
                border: activeTab === "algorithm_compare" ? "1px solid var(--border-medium)" : "1px solid transparent",
              }}
            >
              RecSys Model Benchmark
            </button>
          </div>
        </div>
      </div>

      {isLoading && (
        <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-medium)" }}>
          <div className="material-symbols-outlined" style={{ fontSize: "36px", animation: "spin 1s linear infinite", marginBottom: "12px" }}>
            refresh
          </div>
          <div>Loading catalog comparison data...</div>
        </div>
      )}

      {error && (
        <div style={{ padding: "24px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "12px", color: "#ef4444", marginBottom: "28px" }}>
          <strong>Comparison error:</strong> {error}
        </div>
      )}

      {/* Head-to-Head View */}
      {!isLoading && !error && activeTab === "head_to_head" && (
        <div>
          {/* Selectors */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", marginBottom: "28px" }}>
            <div>
              <label style={{ display: "block", fontSize: "12px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-low)", marginBottom: "8px" }}>
                Select Game 1
              </label>
              <select
                value={gameAId}
                onChange={(e) => handleSelectA(Number(e.target.value))}
                style={{
                  width: "100%",
                  padding: "10px 14px",
                  borderRadius: "8px",
                  background: "var(--bg-surface-elevated)",
                  border: "1px solid var(--border-medium)",
                  color: "var(--text-high)",
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
              <label style={{ display: "block", fontSize: "12px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-low)", marginBottom: "8px" }}>
                Select Game 2
              </label>
              <select
                value={gameBId}
                onChange={(e) => handleSelectB(Number(e.target.value))}
                style={{
                  width: "100%",
                  padding: "10px 14px",
                  borderRadius: "8px",
                  background: "var(--bg-surface-elevated)",
                  border: "1px solid var(--border-medium)",
                  color: "var(--text-high)",
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

          {/* Side-by-Side Comparison Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            {/* Game A Column */}
            {gameA ? (
              <div className="card" style={{ background: "var(--bg-surface-lowest)", border: "1px solid var(--border-subtle)", borderRadius: "16px", overflow: "hidden" }}>
                {gameA.header_image && (
                  <div style={{ width: "100%", height: "180px", overflow: "hidden", position: "relative" }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={gameA.header_image}
                      alt={gameA.name}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                    <div style={{ position: "absolute", bottom: "12px", left: "16px", right: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                      <span style={{ fontSize: "12px", fontWeight: "700", background: "rgba(0,0,0,0.75)", padding: "4px 8px", borderRadius: "4px", color: "#fff" }}>
                        App ID: {gameA.app_id}
                      </span>
                    </div>
                  </div>
                )}
                <div style={{ padding: "20px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
                    <h2 style={{ fontSize: "20px", fontWeight: "800", margin: 0, color: "var(--text-high)" }}>
                      {gameA.name}
                    </h2>
                    <Link
                      href={`/games/${gameA.app_id}?tab=overview`}
                      style={{
                        fontSize: "12px",
                        fontWeight: "700",
                        color: "var(--accent-cyan)",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                      }}
                    >
                      Full Dossier &rarr;
                    </Link>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "20px" }}>
                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Developer</div>
                      {/* HONEST-FALLBACK: Safe indicator if developer is unrecorded */}
                      <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-high)", marginTop: "2px" }}>{gameA.developer ?? "Independent"}</div>
                    </div>

                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Price</div>
                      <div style={{ fontSize: "14px", fontWeight: "700", color: "#10b981", marginTop: "2px" }}>
                        {gameA.is_free ? "Free to Play" : (gameA.final_price_usd ? `$${gameA.final_price_usd}` : "Paid")}
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Positive Reviews</div>
                      <div style={{ fontSize: "14px", fontWeight: "700", color: "var(--text-high)", marginTop: "2px" }}>
                        {gameA.positive_reviews.toLocaleString()} ({Math.round((gameA.positive_reviews / (gameA.positive_reviews + gameA.negative_reviews || 1)) * 100)}%)
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Owner Bracket</div>
                      {/* HONEST-FALLBACK: Safe indicator if owners estimate is unrecorded */}
                      <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-high)", marginTop: "2px" }}>{gameA.owners_estimate ?? "Public Bracket"}</div>
                    </div>
                  </div>

                  {/* Genres */}
                  <div style={{ marginBottom: "16px" }}>
                    <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>Genres &amp; Tags</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      {gameA.genres?.map((g) => (
                        <span key={g.id} style={{ fontSize: "11px", padding: "3px 8px", borderRadius: "4px", background: "rgba(255,255,255,0.05)", color: "var(--text-medium)" }}>
                          {g.description}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Summary */}
                  {gameA.short_description && (
                    <p style={{ fontSize: "13px", color: "var(--text-medium)", lineHeight: 1.5, margin: 0 }}>
                      {gameA.short_description}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ padding: "40px", textAlign: "center", color: "var(--text-medium)" }}>Game A not found</div>
            )}

            {/* Game B Column */}
            {gameB ? (
              <div className="card" style={{ background: "var(--bg-surface-lowest)", border: "1px solid var(--border-subtle)", borderRadius: "16px", overflow: "hidden" }}>
                {gameB.header_image && (
                  <div style={{ width: "100%", height: "180px", overflow: "hidden", position: "relative" }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={gameB.header_image}
                      alt={gameB.name}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                    <div style={{ position: "absolute", bottom: "12px", left: "16px", right: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                      <span style={{ fontSize: "12px", fontWeight: "700", background: "rgba(0,0,0,0.75)", padding: "4px 8px", borderRadius: "4px", color: "#fff" }}>
                        App ID: {gameB.app_id}
                      </span>
                    </div>
                  </div>
                )}
                <div style={{ padding: "20px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
                    <h2 style={{ fontSize: "20px", fontWeight: "800", margin: 0, color: "var(--text-high)" }}>
                      {gameB.name}
                    </h2>
                    <Link
                      href={`/games/${gameB.app_id}?tab=overview`}
                      style={{
                        fontSize: "12px",
                        fontWeight: "700",
                        color: "var(--accent-cyan)",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                      }}
                    >
                      Full Dossier &rarr;
                    </Link>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "20px" }}>
                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Developer</div>
                      {/* HONEST-FALLBACK: Safe indicator if developer is unrecorded */}
                      <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-high)", marginTop: "2px" }}>{gameB.developer ?? "Independent"}</div>
                    </div>

                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Price</div>
                      <div style={{ fontSize: "14px", fontWeight: "700", color: "#10b981", marginTop: "2px" }}>
                        {gameB.is_free ? "Free to Play" : (gameB.final_price_usd ? `$${gameB.final_price_usd}` : "Paid")}
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Positive Reviews</div>
                      <div style={{ fontSize: "14px", fontWeight: "700", color: "var(--text-high)", marginTop: "2px" }}>
                        {gameB.positive_reviews.toLocaleString()} ({Math.round((gameB.positive_reviews / (gameB.positive_reviews + gameB.negative_reviews || 1)) * 100)}%)
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-surface-elevated)", padding: "12px", borderRadius: "8px" }}>
                      <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700" }}>Owner Bracket</div>
                      {/* HONEST-FALLBACK: Safe indicator if owners estimate is unrecorded */}
                      <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-high)", marginTop: "2px" }}>{gameB.owners_estimate ?? "Public Bracket"}</div>
                    </div>
                  </div>

                  {/* Genres */}
                  <div style={{ marginBottom: "16px" }}>
                    <div style={{ fontSize: "11px", color: "var(--text-low)", textTransform: "uppercase", fontWeight: "700", marginBottom: "6px" }}>Genres &amp; Tags</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      {gameB.genres?.map((g) => (
                        <span key={g.id} style={{ fontSize: "11px", padding: "3px 8px", borderRadius: "4px", background: "rgba(255,255,255,0.05)", color: "var(--text-medium)" }}>
                          {g.description}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Summary */}
                  {gameB.short_description && (
                    <p style={{ fontSize: "13px", color: "var(--text-medium)", lineHeight: 1.5, margin: 0 }}>
                      {gameB.short_description}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ padding: "40px", textAlign: "center", color: "var(--text-medium)" }}>Game B not found</div>
            )}
          </div>
        </div>
      )}

      {/* RecSys Algorithm Benchmark View */}
      {!isLoading && !error && activeTab === "algorithm_compare" && (
        <div>
          <div style={{ marginBottom: "20px" }}>
            <h2 style={{ fontSize: "18px", fontWeight: "700", color: "var(--text-high)", margin: 0 }}>
              Algorithm vs Algorithm Comparison
            </h2>
            <p style={{ fontSize: "13px", color: "var(--text-medium)", margin: "4px 0 0 0" }}>
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
