import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "SteamIQ — Steam Game Intelligence & Predictive Analytics Platform",
    template: "%s | SteamIQ",
  },
  description:
    "AI-powered Steam game intelligence. Turn Steam reviews, player telemetry, and market signals into explainable decisions.",
  keywords: [
    "Steam",
    "game analytics",
    "indie developer tools",
    "game intelligence",
    "steam sentiment",
    "game success prediction",
  ],
  openGraph: {
    title: "SteamIQ — Game Intelligence Platform",
    description:
      "Turn Steam reviews, player data, and market signals into explainable decisions.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        {/* Stitch TopNavBar (64px) */}
        <nav className="stitch-nav">
          <div className="stitch-nav__inner">
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

            <div className="stitch-nav__links">
              <Link href="/" className="stitch-nav__link stitch-nav__link--active">
                Search/Landing
              </Link>
              <Link href="/games/1145360?tab=overview" className="stitch-nav__link">
                Game Intelligence
              </Link>
              <span className="stitch-nav__link" style={{ opacity: 0.5, cursor: "not-allowed" }}>
                Compare Games
              </span>
              <span className="stitch-nav__link" style={{ opacity: 0.5, cursor: "not-allowed" }}>
                Market Explorer
              </span>
              <span className="stitch-nav__link" style={{ opacity: 0.5, cursor: "not-allowed" }}>
                Ask SteamIQ
                <span className="stitch-nav__badge-ai">AI</span>
              </span>
            </div>

            <div className="stitch-nav__actions">
              <div className="status-pill">
                <span className="status-dot"></span>
                <span>MODEL ONLINE</span>
              </div>
              <button className="btn-signin">
                Sign In
              </button>
            </div>
          </div>
        </nav>

        <div style={{ paddingTop: "64px", minHeight: "calc(100vh - 80px)" }}>
          {children}
        </div>

        <footer className="footer">
          <div className="container">
            <div className="footer__inner">
              <div>
                <strong>SteamIQ</strong> · Advanced Steam Intelligence &amp; Predictive Analytics
              </div>
              <div style={{ color: "var(--accent-light)", fontFamily: "var(--font-mono)", fontSize: "12px" }}>
                Stitch Design System · Phase 2 NLP Core
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
