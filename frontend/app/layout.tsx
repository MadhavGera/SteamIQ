import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "SteamIQ — Steam Game Intelligence & Analytics Platform",
    template: "%s | SteamIQ",
  },
  description:
    "AI-powered Steam game intelligence. Sentiment analysis, competitor discovery, " +
    "and ML-powered success prediction for indie developers and publishers.",
  keywords: ["Steam", "game analytics", "indie developer tools", "game intelligence", "steam sentiment", "game success prediction"],
  openGraph: {
    title: "SteamIQ — Game Intelligence Platform",
    description: "AI-powered Steam game analytics for developers, publishers, and analysts.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        <nav className="navbar">
          <div className="container">
            <div className="navbar__inner">
              <div className="navbar__brand">
                <Link href="/" className="navbar__logo">
                  <svg
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    style={{ color: "var(--accent-primary)" }}
                  >
                    <path
                      d="M12 2L2 7L12 12L22 7L12 2Z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M2 17L12 22L22 17"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M2 12L12 17L22 12"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span>SteamIQ</span>
                </Link>
              </div>

              <div className="navbar__nav">
                <Link href="/" className="navbar__link navbar__link--active">
                  Search / Landing
                </Link>
                <Link href="/games/1145360" className="navbar__link">
                  Game Intelligence
                </Link>
                <span className="navbar__link" style={{ opacity: 0.6, cursor: "not-allowed" }}>
                  Compare Games
                </span>
                <span className="navbar__link" style={{ opacity: 0.6, cursor: "not-allowed" }}>
                  Market Explorer
                </span>
                <span className="navbar__link" style={{ opacity: 0.6, cursor: "not-allowed" }}>
                  Ask SteamIQ
                </span>
              </div>
            </div>
          </div>
        </nav>

        <main>{children}</main>

        <footer className="footer">
          <div className="container">
            <div className="footer__inner">
              <div>
                <strong>SteamIQ</strong> · Advanced Steam Intelligence &amp; Predictive Analytics
              </div>
              <div>
                <span style={{ color: "var(--accent-primary)", fontFamily: "var(--font-mono)" }}>
                  ADR 0001 Architecture · Phase 1 Foundation
                </span>
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
