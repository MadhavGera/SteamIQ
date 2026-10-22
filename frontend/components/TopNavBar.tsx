"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { UserModeToggle } from "@/components/UserModeToggle";
import { useUserMode } from "@/lib/UserModeContext";

export function TopNavBar() {
  const pathname = usePathname();
  const { isDeveloper, hydrated } = useUserMode();
  const [devMenuOpen, setDevMenuOpen] = useState(false);
  const devMenuRef = useRef<HTMLDivElement>(null);
  const toggleButtonRef = useRef<HTMLButtonElement>(null);

  const isSearchLanding = pathname === "/dashboard";
  const isForYou = pathname === "/";
  const isAsk = pathname.startsWith("/ask");
  const isModelLab = pathname.startsWith("/model-lab");
  const isEmbeddingMap = pathname.startsWith("/embedding-map");
  const isCompare = pathname.startsWith("/compare");
  const isMarketExplorer = pathname.startsWith("/market-explorer");
  const isMonitoring = pathname.startsWith("/monitoring");
  const isGameIntelligence = pathname.startsWith("/games");

  const isDevSectionActive =
    isMarketExplorer || isModelLab || isEmbeddingMap || isMonitoring;

  // Close dropdown on outside click or keyboard navigation
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (devMenuRef.current && !devMenuRef.current.contains(event.target as Node)) {
        setDevMenuOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setDevMenuOpen(false);
        toggleButtonRef.current?.focus();
        return;
      }
      if (!devMenuRef.current) return;
      const items = Array.from(
        devMenuRef.current.querySelectorAll<HTMLAnchorElement>('a[role="menuitem"]')
      );
      if (items.length === 0) return;
      const currentIndex = items.indexOf(document.activeElement as HTMLAnchorElement);

      if (event.key === "ArrowDown") {
        event.preventDefault();
        const nextIndex = currentIndex === -1 || currentIndex === items.length - 1 ? 0 : currentIndex + 1;
        items[nextIndex]?.focus();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        const prevIndex = currentIndex <= 0 ? items.length - 1 : currentIndex - 1;
        items[prevIndex]?.focus();
      } else if (event.key === "Home") {
        event.preventDefault();
        items[0]?.focus();
      } else if (event.key === "End") {
        event.preventDefault();
        items[items.length - 1]?.focus();
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
          {/* 1. Catalog (formerly Search/Landing) */}
          <Link
            href="/dashboard"
            className={`stitch-nav__link ${isSearchLanding ? "stitch-nav__link--active" : ""}`}
            id="nav-search-landing"
          >
            Catalog
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

          {/* 5. For You (Recommender Stream) */}
          <Link
            href="/"
            className={`stitch-nav__link ${isForYou ? "stitch-nav__link--active" : ""}`}
            id="nav-recsys-feed"
          >
            For You
          </Link>

          {/* 6. Developer Tools Dropdown (Rendered strictly when hydrated && isDeveloper) */}
          {hydrated && isDeveloper && (
            <div className="stitch-nav__dropdown" ref={devMenuRef}>
              <button
                ref={toggleButtonRef}
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
                        Demand matrix &amp; trends
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
                        Benchmarks &amp; ablations
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

                  <Link
                    href="/monitoring"
                    className={`stitch-nav__dropdown-item ${isMonitoring ? "stitch-nav__dropdown-item--active" : ""}`}
                    role="menuitem"
                    id="nav-monitoring"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "#f59e0b" }}>
                      monitoring
                    </span>
                    <div>
                      <div style={{ fontWeight: 600 }}>Monitoring</div>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        Drift, gates &amp; model cards
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
        </div>
      </div>
    </nav>
  );
}
