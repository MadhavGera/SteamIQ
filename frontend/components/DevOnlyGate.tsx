"use client";

import React from "react";
import Link from "next/link";
import { useUserMode } from "@/lib/UserModeContext";

interface DevOnlyGateProps {
  children: React.ReactNode;
  pageTitle?: string;
}

export function DevOnlyGate({ children, pageTitle = "Developer Tool" }: DevOnlyGateProps) {
  const { isDeveloper, hydrated, setMode } = useUserMode();

  if (!hydrated) {
    return (
      <main className="container" style={{ padding: "48px 24px" }}>
        <div
          className="card"
          style={{
            maxWidth: "560px",
            margin: "40px auto",
            textAlign: "center",
            padding: "48px 24px",
            borderRadius: "12px",
          }}
        >
          <div className="skeleton" style={{ height: "32px", width: "160px", margin: "0 auto 16px" }} />
          <div className="skeleton" style={{ height: "16px", width: "280px", margin: "0 auto" }} />
        </div>
      </main>
    );
  }

  if (!isDeveloper) {
    return (
      <main className="container" style={{ padding: "48px 24px" }}>
        <div
          className="card"
          style={{
            maxWidth: "600px",
            margin: "40px auto",
            textAlign: "center",
            padding: "48px 32px",
            border: "1px solid var(--border-color, rgba(255,255,255,0.08))",
            borderRadius: "12px",
            background: "var(--card-bg, #1e2330)",
          }}
        >
          <div
            style={{
              width: "56px",
              height: "56px",
              borderRadius: "50%",
              background: "rgba(239, 68, 68, 0.12)",
              color: "#ef4444",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 20px",
              fontSize: "28px",
            }}
          >
            🔒
          </div>
          <h1 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "12px", color: "#f8fafc" }}>
            Developer Mode Required
          </h1>
          <p style={{ color: "#94a3b8", fontSize: "14px", lineHeight: "1.6", marginBottom: "24px" }}>
            {pageTitle} contains internal model evaluations, raw feature diagnostics, and latency benchmarks intended for developers and ML engineers.
          </p>
          <div style={{ display: "flex", justifyContent: "center", gap: "12px", flexWrap: "wrap" }}>
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => setMode("developer")}
            >
              Switch to Developer Mode
            </button>
            <Link href="/" className="btn btn--secondary">
              Return to Catalog
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return <>{children}</>;
}
