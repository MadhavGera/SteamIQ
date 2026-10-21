"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { UserModeToggle } from "@/components/UserModeToggle";
import { useUserMode } from "@/lib/UserModeContext";

export function TopNavBar() {
  const pathname = usePathname();
  const { isDeveloper } = useUserMode();
  const [devMenuOpen, setDevMenuOpen] = useState(false);
  const devMenuRef = useRef<HTMLDivElement>(null);

  const isSearchLanding = pathname === "/dashboard";
  const isForYou = pathname === "/";
  const isAsk = pathname.startsWith("/ask");
  const isModelLab = pathname.startsWith("/model-lab");
  const isEmbeddingMap = pathname.startsWith("/embedding-map");
  const isCompare = pathname.startsWith("/compare");
  const isMarketExplorer = pathname.startsWith("/market-explorer");
  const isGameIntelligence = pathname.startsWith("/games");

  const isDevSectionActive = isMarketExplorer || isModelLab || isEmbeddingMap;

  // Close dropdown on outside click or escape key
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (devMenuRef.current && !devMenuRef.current.contains(event.target as Node)) {
        setDevMenuOpen(false);
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setDevMenuOpen(false);
      }
    }

    if (devMenuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("keydown", handleKeyDown);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [devMenuOpen]);

  // Close dropdown on route change
  useEffect(() => {
    setDevMenuOpen(false);
  }, [pathname]);

  return (
    <nav className="stitch-nav">
      <div className="stitch-nav__inner">
        {/* Brand / Logo */}
        <div className="stitch-nav__brand">
          <Link href="/" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div className="stitch-nav__logo-mark">
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "16px", color: "var(--bg-surface-lowest)", fontVariationSettings: "'FILL' 1" }}
              >
                auto_awesome
              </span>
            </div>
            <span className="stitch-nav__logo-text">SteamIQ</span>
          </Link>
        </div>

        {/* Dynamic Nav Links */}
        <div className="stitch-nav__links">
          {/* 1. Search / Landing */}
          <Link
            href="/dashboard"
            className={`stitch-nav__link ${isSearchLanding ? "stitch-nav__link--active" : ""}`}
            id="nav-search-landing"
          >
            Search/Landing
          </Link>

          {/* 2. Game Intelligence */}
          <Link
            href={isGameIntelligence ? pathname : "/games/1145360?tab=overview"}
            className={`stitch-nav__link ${isGameIntelligence ? "stitch-nav__link--active" : ""}`}
            id="nav-game-intelligence"
          >
            Game Intelligence
          </Link>

          {/* 3. Compare Games */}
          <Link
            href="/compare"
            className={`stitch-nav__link ${isCompare ? "stitch-nav__link--active" : ""}`}
            id="nav-compare"
          >
            Compare Games
          </Link>

          {/* 4. Ask SteamIQ with AI Badge Pill */}
          <Link
            href="/ask"
            className={`stitch-nav__link ${isAsk ? "stitch-nav__link--active" : ""}`}
            id="nav-ask"
          >
            <span>Ask SteamIQ</span>
            <span className="stitch-nav__badge-ai">AI</span>
          </Link>

          {/* 5. RecSys Feed (Two-Tower Recommendation Stream) */}
          <Link
            href="/"
            className={`stitch-nav__link ${isForYou ? "stitch-nav__link--active" : ""}`}
            id="nav-recsys-feed"
          >
            RecSys Feed
          </Link>

          {/* 6. Developer Tools Dropdown (Dev Mode Only - keeps navbar sleek & uncluttered) */}
          {isDeveloper && (
            <div className="stitch-nav__dropdown" ref={devMenuRef}>
              <button
                type="button"
                onClick={() => setDevMenuOpen((prev) => !prev)}
                className={`stitch-nav__link stitch-nav__dropdown-toggle ${isDevSectionActive ? "stitch-nav__link--active" : ""}`}
                id="nav-dev-tools-toggle"
                aria-expanded={devMenuOpen}
                aria-haspopup="true"
              >
                <span className="material-symbols-outlined" style={{ fontSize: "15px" }}>
                  terminal
                </span>
                <span>Dev Tools</span>
                <span
                  className="material-symbols-outlined"
                  style={{
                    fontSize: "16px",
                    transform: devMenuOpen ? "rotate(180deg)" : "none",
                    transition: "transform 0.2s ease",
                  }}
                >
                  expand_more
                </span>
              </button>

              {devMenuOpen && (
                <div className="stitch-nav__dropdown-menu" role="menu">
                  <Link
                    href="/market-explorer"
                    className={`stitch-nav__dropdown-item ${isMarketExplorer ? "stitch-nav__dropdown-item--active" : ""}`}
                    role="menuitem"
                    id="nav-market-explorer"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "var(--accent-cyan)" }}>
                      insights
                    </span>
                    <div>
                      <div style={{ fontWeight: 600 }}>Market Explorer</div>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        Demand matrix & trends
                      </div>
                    </div>
                  </Link>

                  <Link
                    href="/model-lab"
                    className={`stitch-nav__dropdown-item ${isModelLab ? "stitch-nav__dropdown-item--active" : ""}`}
                    role="menuitem"
                    id="nav-model-lab"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "var(--accent-primary)" }}>
                      science
                    </span>
                    <div>
                      <div style={{ fontWeight: 600 }}>Model Lab</div>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        Benchmarks & ablations
                      </div>
                    </div>
                  </Link>

                  <Link
                    href="/embedding-map"
                    className={`stitch-nav__dropdown-item ${isEmbeddingMap ? "stitch-nav__dropdown-item--active" : ""}`}
                    role="menuitem"
                    id="nav-embedding-map"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "#10b981" }}>
                      hub
                    </span>
                    <div>
                      <div style={{ fontWeight: 600 }}>Embedding Map</div>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        2D UMAP latent space
                      </div>
                    </div>
                  </Link>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Controls & Mode Toggle */}
        <div className="stitch-nav__actions" style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <UserModeToggle />
          <button className="btn-signin">
            Sign In
          </button>
        </div>
      </div>
    </nav>
  );
}
