import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "SteamIQ — Game Intelligence Platform",
    template: "%s | SteamIQ",
  },
  description:
    "AI-powered Steam game analytics. Sentiment analysis, competitor discovery, " +
    "success prediction, and actionable intelligence for indie developers and publishers.",
  keywords: ["Steam", "game analytics", "indie developer tools", "game intelligence"],
  openGraph: {
    title: "SteamIQ — Game Intelligence Platform",
    description: "AI-powered Steam game analytics for developers and publishers.",
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
              <a href="/" className="navbar__logo">
                <div className="navbar__logo-icon">⚡</div>
                <span>
                  <span className="navbar__logo-gradient">SteamIQ</span>
                </span>
                <span className="navbar__tagline">Game Intelligence</span>
              </a>
            </div>
          </div>
        </nav>

        <main>{children}</main>

        <footer className="footer">
          <div className="container">
            <p>
              SteamIQ · Game data from Steam &amp; SteamSpy ·{" "}
              <span style={{ color: "var(--color-accent-primary)" }}>Phase 1</span>
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
