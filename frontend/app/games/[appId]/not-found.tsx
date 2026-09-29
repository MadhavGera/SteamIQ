"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { getApiBaseUrl } from "@/lib/api";

export default function NotFound() {
  const pathname = usePathname();
  const appIdMatch = pathname.match(/\/games\/(\d+)/);
  const appId = appIdMatch ? appIdMatch[1] : null;

  const [status, setStatus] = useState<"idle" | "triggering" | "polling" | "error">("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (status === "polling" && jobId) {
      const apiUrl = getApiBaseUrl();
      intervalId = setInterval(async () => {
        try {
          const res = await fetch(`${apiUrl}/api/v1/ingestion/status/${jobId}`);
          if (!res.ok) {
            throw new Error("Failed to check status");
          }
          const data = await res.json();
          if (data.status === "completed") {
            clearInterval(intervalId);
            setStatus("idle");
            window.location.reload();
          } else if (data.status === "failed") {
            clearInterval(intervalId);
            setStatus("error");
            setErrorMessage(data.error_message || "Ingestion failed.");
          }
        } catch (err) {
          console.error("Polling error:", err);
          clearInterval(intervalId);
          setStatus("error");
          setErrorMessage("Failed to check analysis status.");
        }
      }, 3000);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [status, jobId]);

  const handleAnalyze = async () => {
    if (!appId) return;
    setStatus("triggering");
    setErrorMessage(null);
    try {
      const apiUrl = getApiBaseUrl();
      const res = await fetch(`${apiUrl}/api/v1/ingestion/trigger`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ app_id: parseInt(appId) })
      });
      
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const errorMsg = data.error?.message || data.detail || (res.status === 404 ? "Game not found on Steam." : "Failed to trigger analysis");
        throw new Error(errorMsg);
      }
      
      setJobId(data.job_id);
      setStatus("polling");
    } catch (err) {
      console.error(err);
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Failed to trigger analysis.");
    }
  };

  return (
    <div className="container">
      <div className="empty-state" style={{ paddingTop: 120 }}>
        <div className="empty-state__icon">🎮</div>
        <h1 className="empty-state__title">Game not indexed</h1>
        <p className="empty-state__body" style={{ marginBottom: 24 }}>
          This game hasn&apos;t been fully analyzed by SteamIQ yet.
        </p>

        {status === "idle" && (
          <button 
            onClick={handleAnalyze}
            className="btn btn-primary"
            style={{ marginBottom: 16 }}
          >
            Analyze Game (Takes ~1 min)
          </button>
        )}

        {status === "triggering" && (
          <p style={{ color: "var(--color-text-secondary)" }}>Starting analysis...</p>
        )}

        {status === "polling" && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
            <div className="spinner" style={{ width: 24, height: 24, border: "3px solid var(--color-border)", borderTopColor: "var(--color-accent-primary)", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            <p style={{ color: "var(--color-text-secondary)" }}>
              Analysis in progress... this may take up to a minute.
            </p>
          </div>
        )}

        {status === "error" && (
          <div style={{ marginTop: 16, color: "var(--color-negative)" }}>
            <p style={{ marginBottom: 12 }}>{errorMessage}</p>
            <button onClick={handleAnalyze} className="btn btn-secondary">
              Try Again
            </button>
          </div>
        )}

        <div style={{ marginTop: 32 }}>
          <Link
            href="/"
            style={{
              display: "inline-block",
              color: "var(--color-accent-primary)",
              textDecoration: "underline",
              fontSize: "0.9rem",
            }}
          >
            ← Back to search
          </Link>
        </div>
      </div>
    </div>
  );
}
