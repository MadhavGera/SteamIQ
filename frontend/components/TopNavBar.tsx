"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { UserModeToggle } from "@/components/UserModeToggle";

export function TopNavBar() {
  const pathname = usePathname();

  const isForYou = pathname === "/";
  const isModelLab = pathname.startsWith("/model-lab");
  const isEmbeddingMap = pathname.startsWith("/embedding-map");
  const isDashboard = pathname.startsWith("/dashboard");
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
          <Link
            href="/"
            className={`stitch-nav__link ${isForYou ? "stitch-nav__link--active" : ""}`}
            id="nav-for-you"
          >
            For You
          </Link>
          <Link
            href="/model-lab"
            className={`stitch-nav__link ${isModelLab ? "stitch-nav__link--active" : ""}`}
            id="nav-model-lab"
          >
            Model Lab
          </Link>
          <Link
            href="/embedding-map"
            className={`stitch-nav__link ${isEmbeddingMap ? "stitch-nav__link--active" : ""}`}
            id="nav-embedding-map"
          >
            Embedding Map
          </Link>
          <Link
            href="/dashboard"
            className={`stitch-nav__link ${isDashboard ? "stitch-nav__link--active" : ""}`}
            id="nav-dashboard"
          >
            Dashboard
          </Link>
          <Link
            href={isGameIntelligence ? pathname : "/games/1145360?tab=overview"}
            className={`stitch-nav__link ${isGameIntelligence ? "stitch-nav__link--active" : ""}`}
            id="nav-game-intelligence"
          >
            Game Intelligence
          </Link>
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
