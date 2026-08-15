import type { Metadata } from "next";
import "./globals.css";
import { TopNavBar } from "@/components/TopNavBar";
import { UserModeProvider } from "@/lib/UserModeContext";

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
        <UserModeProvider>
          <TopNavBar />

          <div style={{ paddingTop: "64px", minHeight: "calc(100vh - 80px)" }}>
            {children}
          </div>

          <footer className="footer">
            <div className="container">
              <div className="footer__inner">
                <div>
                  <strong>SteamIQ</strong> · Advanced Steam Intelligence &amp; Predictive Analytics
                </div>
              </div>
            </div>
          </footer>
        </UserModeProvider>
      </body>
    </html>
  );
}
