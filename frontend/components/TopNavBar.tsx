"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function TopNavBar() {
  const pathname = usePathname();

  const isSearchLanding = pathname === "/";
  const isGameIntelligence = pathname.startsWith("/games");
  const isCompare = pathname.startsWith("/compare");
  const isMarket = pathname.startsWith("/market");
  const isAsk = pathname.startsWith("/ask");

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
                speed
              </span>
            </div>
            <span className="stitch-nav__logo-text">SteamIQ</span>
          </Link>
        </div>

        {/* Dynamic Nav Links */}
        <div className="stitch-nav__links">
          <Link
            href="/"
            className={`stitch-nav__link ${isSearchLanding ? "stitch-nav__link--active" : ""}`}
          >
            Search/Landing
          </Link>
          <Link
            href={isGameIntelligence ? pathname : "/games/1145360?tab=overview"}
            className={`stitch-nav__link ${isGameIntelligence ? "stitch-nav__link--active" : ""}`}
          >
            Game Intelligence
          </Link>
          <span
            className={`stitch-nav__link ${isCompare ? "stitch-nav__link--active" : ""}`}
            style={{ opacity: 0.5, cursor: "not-allowed" }}
          >
            Compare Games
          </span>
          <span
            className={`stitch-nav__link ${isMarket ? "stitch-nav__link--active" : ""}`}
            style={{ opacity: 0.5, cursor: "not-allowed" }}
          >
            Market Explorer
          </span>
          <span
            className={`stitch-nav__link ${isAsk ? "stitch-nav__link--active" : ""}`}
            style={{ opacity: 0.5, cursor: "not-allowed" }}
          >
            Ask SteamIQ
            <span className="stitch-nav__badge-ai">AI</span>
          </span>
        </div>

        {/* Action Button */}
        <div className="stitch-nav__actions">
          <button className="btn-signin">
            Sign In
          </button>
        </div>
      </div>
    </nav>
  );
}
