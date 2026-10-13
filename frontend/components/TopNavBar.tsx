"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { UserModeToggle } from "@/components/UserModeToggle";
import { useUserMode } from "@/lib/UserModeContext";

export function TopNavBar() {
  const pathname = usePathname();
  const { isDeveloper } = useUserMode();

  const isSearchLanding = pathname === "/dashboard";
  const isForYou = pathname === "/";
  const isAsk = pathname.startsWith("/ask");
  const isModelLab = pathname.startsWith("/model-lab");
  const isEmbeddingMap = pathname.startsWith("/embedding-map");
  const isCompare = pathname.startsWith("/compare");
  const isMarketExplorer = pathname.startsWith("/market-explorer");
  const isGameIntelligence = pathname.startsWith("/games");

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

          {/* 4. Market Explorer (Developer Mode Only per Roadmap §2) */}
          {isDeveloper && (
            <Link
              href="/market-explorer"
              className={`stitch-nav__link ${isMarketExplorer ? "stitch-nav__link--active" : ""}`}
              id="nav-market-explorer"
            >
              Market Explorer
            </Link>
          )}

          {/* 5. Ask SteamIQ with AI Badge Pill */}
          <Link
            href="/ask"
            className={`stitch-nav__link ${isAsk ? "stitch-nav__link--active" : ""}`}
            id="nav-ask"
          >
            <span>Ask SteamIQ</span>
            <span className="stitch-nav__badge-ai">AI</span>
          </Link>

          {/* Developer Tools (Dev Mode Only) */}
          {isDeveloper && (
            <>
              <Link
                href="/"
                className={`stitch-nav__link ${isForYou ? "stitch-nav__link--active" : ""}`}
                id="nav-for-you"
                title="RecSys For You Feed"
              >
                For You
              </Link>
              <Link
                href="/model-lab"
                className={`stitch-nav__link ${isModelLab ? "stitch-nav__link--active" : ""}`}
                id="nav-model-lab"
                title="ML / Model Evaluation Lab"
              >
                Model Lab
              </Link>
              <Link
                href="/embedding-map"
                className={`stitch-nav__link ${isEmbeddingMap ? "stitch-nav__link--active" : ""}`}
                id="nav-embedding-map"
                title="UMAP Semantic Embedding Map"
              >
                Embedding Map
              </Link>
            </>
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
