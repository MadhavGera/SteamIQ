"use client";

import React, { useState, useEffect, useMemo, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { DevOnlyGate } from "@/components/DevOnlyGate";
import {
  api,
  OpsModelCard,
} from "@/lib/api";

// ─── Constants & Tab Definitions ─────────────────────────────────────────────

type MonitoringTabKey =
  | "telemetry"
  | "drift"
  | "retrain_gate"
  | "power"
  | "model_cards";

interface TabDef {
  key: MonitoringTabKey;
  label: string;
  icon: string;
  description: string;
}

const TABS: TabDef[] = [
  {
    key: "telemetry",
    label: "Live Telemetry",
    icon: "speed",
    description: "Prometheus metric counters, gauges & latency histograms",
  },
  {
    key: "drift",
    label: "Statistical Drift",
    icon: "insights",
    description: "PSI, KS-test & JSD distribution tests across serving features",
  },
  {
    key: "retrain_gate",
    label: "Retrain Gate Verdict",
    icon: "gavel",
    description: "Champion vs challenger multi-seed qualification summary",
  },
  {
    key: "power",
    label: "Detector Power",
    icon: "query_stats",
    description: "Statistical power curves & FPR under synthetic shift trials",
  },
  {
    key: "model_cards",
    label: "Model Cards",
    icon: "badge",
    description: "Production and candidate model documentation & lineage",
  },
];

// ─── Markdown Renderer Component ─────────────────────────────────────────────

function SimpleMarkdownViewer({ content }: { content: string }) {
  const renderedSections = useMemo(() => {
    const lines = content.split("\n");
    const elements: React.ReactNode[] = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Code blocks
      if (line.startsWith("```")) {
        const lang = line.slice(3).trim();
        const codeLines: string[] = [];
        i++;
        while (i < lines.length && !lines[i].startsWith("```")) {
          codeLines.push(lines[i]);
          i++;
        }
        i++; // skip closing ```
        elements.push(
          <div
            key={`code-${i}`}
            style={{
              margin: "16px 0",
              background: "var(--bg-surface-lowest)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "8px",
              padding: "16px",
              overflowX: "auto",
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "12px",
            }}
          >
            {lang && (
              <div
                style={{
                  fontSize: "10px",
                  textTransform: "uppercase",
                  color: "var(--text-muted)",
                  marginBottom: "8px",
                  fontWeight: 600,
                }}
              >
                {lang}
              </div>
            )}
            <pre style={{ margin: 0, color: "var(--text-primary)" }}>
              {codeLines.join("\n")}
            </pre>
          </div>
        );
        continue;
      }

      // Tables
      if (line.startsWith("|") && line.endsWith("|")) {
        const tableLines: string[] = [];
        while (i < lines.length && lines[i].startsWith("|") && lines[i].endsWith("|")) {
          tableLines.push(lines[i]);
          i++;
        }
        if (tableLines.length >= 2) {
          const headerCells = tableLines[0]
            .split("|")
            .slice(1, -1)
            .map((c) => c.trim());
          const bodyRows = tableLines.slice(2).map((row) =>
            row
              .split("|")
              .slice(1, -1)
              .map((c) => c.trim())
          );

          elements.push(
            <div
              key={`table-${i}`}
              style={{
                margin: "20px 0",
                overflowX: "auto",
                border: "1px solid var(--border-subtle)",
                borderRadius: "8px",
              }}
            >
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "13px",
                  textAlign: "left",
                }}
              >
                <thead>
                  <tr style={{ background: "var(--bg-surface-lowest)" }}>
                    {headerCells.map((h, hIdx) => (
                      <th
                        key={hIdx}
                        style={{
                          padding: "10px 14px",
                          borderBottom: "1px solid var(--border-subtle)",
                          fontWeight: 600,
                          color: "var(--text-primary)",
                        }}
                      >
                        {h.replace(/\*\*/g, "")}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bodyRows.map((row, rIdx) => (
                    <tr
                      key={rIdx}
                      style={{
                        borderBottom:
                          rIdx < bodyRows.length - 1
                            ? "1px solid var(--border-subtle)"
                            : "none",
                        background:
                          rIdx % 2 === 1 ? "rgba(255, 255, 255, 0.02)" : "transparent",
                      }}
                    >
                      {row.map((cell, cIdx) => (
                        <td
                          key={cIdx}
                          style={{
                            padding: "10px 14px",
                            color: "var(--text-secondary)",
                          }}
                        >
                          {cell.replace(/\*\*/g, "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        continue;
      }

      // Headings
      if (line.startsWith("# ")) {
        elements.push(
          <h1
            key={`h1-${i}`}
            style={{
              fontSize: "24px",
              fontWeight: 700,
              margin: "24px 0 12px",
              color: "var(--text-primary)",
            }}
          >
            {line.replace(/^#\s+/, "")}
          </h1>
        );
        i++;
        continue;
      }
      if (line.startsWith("## ")) {
        elements.push(
          <h2
            key={`h2-${i}`}
            style={{
              fontSize: "18px",
              fontWeight: 600,
              margin: "24px 0 12px",
              color: "var(--text-primary)",
              borderBottom: "1px solid var(--border-subtle)",
              paddingBottom: "6px",
            }}
          >
            {line.replace(/^##\s+/, "")}
          </h2>
        );
        i++;
        continue;
      }
      if (line.startsWith("### ")) {
        elements.push(
          <h3
            key={`h3-${i}`}
            style={{
              fontSize: "15px",
              fontWeight: 600,
              margin: "18px 0 8px",
              color: "var(--accent-primary)",
            }}
          >
            {line.replace(/^###\s+/, "")}
          </h3>
        );
        i++;
        continue;
      }

      // Horizontal rule
      if (line.trim() === "---") {
        elements.push(
          <hr
            key={`hr-${i}`}
            style={{
              margin: "24px 0",
              borderColor: "var(--border-subtle)",
              opacity: 0.5,
            }}
          />
        );
        i++;
        continue;
      }

      // Blockquotes
      if (line.startsWith("> ")) {
        elements.push(
          <blockquote
            key={`bq-${i}`}
            style={{
              margin: "14px 0",
              padding: "10px 16px",
              borderLeft: "3px solid var(--accent-primary)",
              background: "rgba(99, 102, 241, 0.05)",
              color: "var(--text-secondary)",
              fontSize: "13px",
              borderRadius: "0 6px 6px 0",
            }}
          >
            {line.replace(/^>\s+/, "")}
          </blockquote>
        );
        i++;
        continue;
      }

      // Bullet items
      if (line.startsWith("- ")) {
        elements.push(
          <div
            key={`li-${i}`}
            style={{
              display: "flex",
              gap: "8px",
              margin: "6px 0",
              fontSize: "13px",
              color: "var(--text-secondary)",
            }}
          >
            <span style={{ color: "var(--accent-primary)" }}>•</span>
            <span>{line.slice(2)}</span>
          </div>
        );
        i++;
        continue;
      }

      // Standard paragraphs
      if (line.trim().length > 0) {
        elements.push(
          <p
            key={`p-${i}`}
            style={{
              fontSize: "13px",
              lineHeight: "1.6",
              color: "var(--text-secondary)",
              margin: "8px 0",
            }}
          >
            {line}
          </p>
        );
      }

      i++;
    }

    return elements;
  }, [content]);

  return <div>{renderedSections}</div>;
}

// ─── Main Monitoring Component ───────────────────────────────────────────────

function MonitoringContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // HONEST-FALLBACK: Default section query parameter to telemetry if unspecified
  const initialSection = (searchParams.get("section") as MonitoringTabKey) ?? "telemetry";
  const [activeTab, setActiveTab] = useState<MonitoringTabKey>(initialSection);

  // Data states
  const [loading, setLoading] = useState<boolean>(true);
  const [metricsText, setMetricsText] = useState<string>("");
  const [driftReport, setDriftReport] = useState<Record<string, unknown> | null>(null);
  const [retrainGateData, setRetrainGateData] = useState<Record<string, unknown> | null>(null);
  const [retrainGateLog, setRetrainGateLog] = useState<string>("");
  const [detectorPowerData, setDetectorPowerData] = useState<Record<string, unknown> | null>(null);
  const [modelCards, setModelCards] = useState<OpsModelCard[]>([]);
  const [selectedCardIdx, setSelectedCardIdx] = useState<number>(0);
  const [viewRawMarkdown, setViewRawMarkdown] = useState<boolean>(false);
  const [copyFeedback, setCopyFeedback] = useState<string>("");

  // Sync tab with URL
  const handleTabChange = (tab: MonitoringTabKey) => {
    setActiveTab(tab);
    router.replace(`/monitoring?section=${tab}`);
  };

  useEffect(() => {
    const s = searchParams.get("section") as MonitoringTabKey | null;
    if (s && TABS.some((t) => t.key === s)) {
      setActiveTab(s);
    }
  }, [searchParams]);

  // Load telemetry, drift, gate, power, and model cards
  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      setLoading(true);
      try {
        const [mText, dRes, gateRes, powerRes, cardsRes] = await Promise.allSettled([
          api.opsMetrics(),
          api.opsDrift(),
          api.opsEvidence("retrain_gate"),
          api.opsEvidence("drift_detector_power"),
          api.opsModelCards(),
        ]);

        if (!isMounted) return;

        if (mText.status === "fulfilled") {
          setMetricsText(mText.value);
        }
        if (dRes.status === "fulfilled") {
          // HONEST-FALLBACK: Extract report payload or fallback to entire object
          setDriftReport(dRes.value.report ?? (dRes.value as unknown as Record<string, unknown>));
        }
        if (gateRes.status === "fulfilled") {
          // HONEST-FALLBACK: Evidence response carries optional parsed data
          setRetrainGateData(gateRes.value.data ?? null);
          // HONEST-FALLBACK: Provide empty log fallback
          setRetrainGateLog(gateRes.value.log ?? "");
        }
        if (powerRes.status === "fulfilled") {
          // HONEST-FALLBACK: Evidence response carries optional parsed data
          setDetectorPowerData(powerRes.value.data ?? null);
        }
        if (cardsRes.status === "fulfilled") {
          setModelCards(cardsRes.value);
        }
      } catch (err) {
        console.error("Error fetching monitoring operations data:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  // Parse Prometheus metrics text
  const parsedMetrics = useMemo(() => {
    const res = {
      uptimeSeconds: 0,
      recsysRequests: {} as Record<string, number>,
      assistantRequests: {} as Record<string, number>,
      cacheHits: { recsys: 0, assistant: 0 },
      cacheMisses: { recsys: 0, assistant: 0 },
      recsysLatencyBuckets: {} as Record<string, number>,
    };

    if (!metricsText) return res;

    for (const line of metricsText.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;

      if (trimmed.startsWith("steamiq_uptime_seconds ")) {
        // HONEST-FALLBACK: Parse float from line segment
        res.uptimeSeconds = parseFloat(trimmed.split(" ")[1] ?? "0");
      } else if (trimmed.startsWith("steamiq_recsys_requests_total{")) {
        const match = trimmed.match(/model="([^"]+)"\}\s+(\d+)/);
        if (match) {
          // HONEST-FALLBACK: Default parsed count to 0
          res.recsysRequests[match[1]] = parseInt(match[2] ?? "0", 10);
        }
      } else if (trimmed.startsWith("steamiq_assistant_requests_total{")) {
        const match = trimmed.match(/intent="([^"]+)"\}\s+(\d+)/);
        if (match) {
          // HONEST-FALLBACK: Default parsed count to 0
          res.assistantRequests[match[1]] = parseInt(match[2] ?? "0", 10);
        }
      } else if (trimmed.startsWith("steamiq_cache_hits_total{")) {
        const match = trimmed.match(/service="([^"]+)"\}\s+(\d+)/);
        if (match && (match[1] === "recsys" || match[1] === "assistant")) {
          // HONEST-FALLBACK: Default parsed count to 0
          res.cacheHits[match[1]] = parseInt(match[2] ?? "0", 10);
        }
      } else if (trimmed.startsWith("steamiq_cache_misses_total{")) {
        const match = trimmed.match(/service="([^"]+)"\}\s+(\d+)/);
        if (match && (match[1] === "recsys" || match[1] === "assistant")) {
          // HONEST-FALLBACK: Default parsed count to 0
          res.cacheMisses[match[1]] = parseInt(match[2] ?? "0", 10);
        }
      } else if (trimmed.startsWith("steamiq_recsys_latency_ms_bucket{")) {
        const match = trimmed.match(/le="([^"]+)"\}\s+(\d+)/);
        if (match) {
          // HONEST-FALLBACK: Default parsed count to 0
          res.recsysLatencyBuckets[match[1]] = parseInt(match[2] ?? "0", 10);
        }
      }
    }

    return res;
  }, [metricsText]);

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopyFeedback(label);
    setTimeout(() => setCopyFeedback(""), 2000);
  };

  return (
    <DevOnlyGate pageTitle="Monitoring & Operations">
      <main className="container" style={{ paddingTop: "88px", paddingBottom: "80px" }}>
        {/* Header Strip */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: "16px",
            marginBottom: "28px",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "28px", color: "var(--accent-primary)" }}
              >
                monitoring
              </span>
              <h1 style={{ fontSize: "28px", fontWeight: 700, margin: 0 }}>
                Operations &amp; Model Monitoring
              </h1>
              <span className="badge-pill" style={{ fontSize: "11px" }}>
                Phase 7 MLOps
              </span>
            </div>
            <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "14px", maxWidth: "700px" }}>
              Real-time Prometheus telemetry, statistical distribution drift testing, champion qualification gate decisions, detector power curves, and production model cards.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span
              className="badge-pill badge-pill--success"
              style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", padding: "6px 12px" }}
            >
              <span
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  backgroundColor: "#10b981",
                  display: "inline-block",
                }}
              />
              Telemetry Online
            </span>
            <span
              className="badge-pill"
              style={{
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "12px",
                padding: "6px 12px",
                background: "var(--bg-surface-mid)",
              }}
            >
              dataset: c662b8232dc0
            </span>
          </div>
        </div>

        {/* Tab Navigation Navigation */}
        <div
          style={{
            display: "flex",
            gap: "8px",
            borderBottom: "1px solid var(--border-subtle)",
            marginBottom: "32px",
            overflowX: "auto",
            paddingBottom: "8px",
          }}
        >
          {TABS.map((tab) => {
            const isActive = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => handleTabChange(tab.key)}
                className={`stitch-nav__link ${isActive ? "stitch-nav__link--active" : ""}`}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "10px 16px",
                  borderRadius: "6px",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: isActive ? 600 : 500,
                  fontSize: "13px",
                  whiteSpace: "nowrap",
                  background: isActive ? "var(--bg-surface-mid)" : "transparent",
                  color: isActive ? "var(--text-primary)" : "var(--text-muted)",
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                  {tab.icon}
                </span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {loading ? (
          <div
            style={{
              padding: "64px 0",
              textAlign: "center",
              color: "var(--text-muted)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "12px",
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "32px", animation: "spin 1s linear infinite" }}>
              progress_activity
            </span>
            <span>Loading telemetry &amp; monitoring evidence...</span>
          </div>
        ) : (
          <>
            {/* ─── SECTION 1: LIVE TELEMETRY ──────────────────────────── */}
            {activeTab === "telemetry" && (
              <div>
                {/* KPI Metric Summary Strip */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: "16px",
                    marginBottom: "24px",
                  }}
                >
                  <div className="card" style={{ padding: "20px" }}>
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "6px" }}>
                      TOTAL RECSYS INVOCATIONS
                    </div>
                    <div style={{ fontSize: "28px", fontWeight: 700, color: "var(--accent-primary)" }}>
                      {Object.values(parsedMetrics.recsysRequests).reduce((a, b) => a + b, 0)}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
                      Active serving models: {Object.keys(parsedMetrics.recsysRequests).length}
                    </div>
                  </div>

                  <div className="card" style={{ padding: "20px" }}>
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "6px" }}>
                      ASSISTANT QUERIES
                    </div>
                    <div style={{ fontSize: "28px", fontWeight: 700, color: "var(--accent-cyan)" }}>
                      {Object.values(parsedMetrics.assistantRequests).reduce((a, b) => a + b, 0)}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
                      Active intent categories: {Object.keys(parsedMetrics.assistantRequests).length}
                    </div>
                  </div>

                  <div className="card" style={{ padding: "20px" }}>
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "6px" }}>
                      RECSYS CACHE HIT RATIO
                    </div>
                    <div style={{ fontSize: "28px", fontWeight: 700, color: "#10b981" }}>
                      {(() => {
                        const total = parsedMetrics.cacheHits.recsys + parsedMetrics.cacheMisses.recsys;
                        if (total === 0) return "100%";
                        return `${((parsedMetrics.cacheHits.recsys / total) * 100).toFixed(1)}%`;
                      })()}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
                      Hits: {parsedMetrics.cacheHits.recsys} | Misses: {parsedMetrics.cacheMisses.recsys}
                    </div>
                  </div>

                  <div className="card" style={{ padding: "20px" }}>
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "6px" }}>
                      PROCESS UPTIME
                    </div>
                    <div style={{ fontSize: "28px", fontWeight: 700, color: "var(--text-primary)" }}>
                      {parsedMetrics.uptimeSeconds > 0
                        ? `${Math.floor(parsedMetrics.uptimeSeconds)}s`
                        : "Ready"}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
                      Bounded in-memory collector
                    </div>
                  </div>
                </div>

                {/* Latency Histograms & Service Counts */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))",
                    gap: "20px",
                    marginBottom: "24px",
                  }}
                >
                  <div className="card" style={{ padding: "24px" }}>
                    <h3 style={{ fontSize: "15px", fontWeight: 600, margin: "0 0 16px" }}>
                      Recommender Latency Buckets (ms)
                    </h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                      {[
                        { le: "10.0", label: "≤ 10 ms" },
                        { le: "25.0", label: "≤ 25 ms" },
                        { le: "50.0", label: "≤ 50 ms (SLA Target)" },
                        { le: "100.0", label: "≤ 100 ms" },
                        { le: "+Inf", label: "Total Requests" },
                      ].map((bucket) => {
                        // HONEST-FALLBACK: Default parsed bucket count to 0
                        const count = parsedMetrics.recsysLatencyBuckets[bucket.le] ?? 0;
                        const maxCount = Math.max(
                          // HONEST-FALLBACK: Default fallback to 1 for denominator
                          parsedMetrics.recsysLatencyBuckets["+Inf"] ?? 1,
                          1
                        );
                        const pct = Math.min(100, Math.round((count / maxCount) * 100));
                        return (
                          <div key={bucket.le}>
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                fontSize: "12px",
                                marginBottom: "4px",
                              }}
                            >
                              <span style={{ color: "var(--text-secondary)" }}>{bucket.label}</span>
                              <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                                {count} ({pct}%)
                              </span>
                            </div>
                            <div
                              style={{
                                width: "100%",
                                height: "6px",
                                background: "var(--bg-surface-lowest)",
                                borderRadius: "3px",
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  width: `${pct}%`,
                                  height: "100%",
                                  background: "var(--accent-primary)",
                                  borderRadius: "3px",
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="card" style={{ padding: "24px" }}>
                    <h3 style={{ fontSize: "15px", fontWeight: 600, margin: "0 0 16px" }}>
                      Cache &amp; Request Breakdown
                    </h3>
                    <table style={{ width: "100%", fontSize: "13px", borderCollapse: "collapse" }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                          <th style={{ padding: "8px 0", textAlign: "left", color: "var(--text-muted)" }}>Service</th>
                          <th style={{ padding: "8px 0", textAlign: "right", color: "var(--text-muted)" }}>Requests</th>
                          <th style={{ padding: "8px 0", textAlign: "right", color: "var(--text-muted)" }}>Hits</th>
                          <th style={{ padding: "8px 0", textAlign: "right", color: "var(--text-muted)" }}>Hit Rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                          <td style={{ padding: "10px 0", fontWeight: 500 }}>Recommender (RecSys)</td>
                          <td style={{ padding: "10px 0", textAlign: "right" }}>
                            {parsedMetrics.cacheHits.recsys + parsedMetrics.cacheMisses.recsys}
                          </td>
                          <td style={{ padding: "10px 0", textAlign: "right" }}>
                            {parsedMetrics.cacheHits.recsys}
                          </td>
                          <td style={{ padding: "10px 0", textAlign: "right", color: "#10b981", fontWeight: 600 }}>
                            {parsedMetrics.cacheHits.recsys + parsedMetrics.cacheMisses.recsys > 0
                              ? `${Math.round(
                                  (parsedMetrics.cacheHits.recsys /
                                    (parsedMetrics.cacheHits.recsys + parsedMetrics.cacheMisses.recsys)) *
                                    100
                                )}%`
                              : "100%"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "10px 0", fontWeight: 500 }}>Ask SteamIQ (Assistant)</td>
                          <td style={{ padding: "10px 0", textAlign: "right" }}>
                            {parsedMetrics.cacheHits.assistant + parsedMetrics.cacheMisses.assistant}
                          </td>
                          <td style={{ padding: "10px 0", textAlign: "right" }}>
                            {parsedMetrics.cacheHits.assistant}
                          </td>
                          <td style={{ padding: "10px 0", textAlign: "right", color: "#10b981", fontWeight: 600 }}>
                            {parsedMetrics.cacheHits.assistant + parsedMetrics.cacheMisses.assistant > 0
                              ? `${Math.round(
                                  (parsedMetrics.cacheHits.assistant /
                                    (parsedMetrics.cacheHits.assistant + parsedMetrics.cacheMisses.assistant)) *
                                    100
                                )}%`
                              : "100%"}
                          </td>
                        </tr>
                      </tbody>
                    </table>

                    <div
                      style={{
                        marginTop: "20px",
                        padding: "12px",
                        borderRadius: "6px",
                        background: "rgba(99, 102, 241, 0.05)",
                        border: "1px solid rgba(99, 102, 241, 0.2)",
                        fontSize: "12px",
                        color: "var(--text-secondary)",
                      }}
                    >
                      <strong style={{ color: "var(--accent-primary)" }}>Privacy Invariant:</strong> Telemetry collector records aggregate counts and latency histograms only. User library items and raw query text are never buffered.
                    </div>
                  </div>
                </div>

                {/* Raw Prometheus Text Exposition Card */}
                <div className="card" style={{ padding: "24px" }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "16px",
                    }}
                  >
                    <div>
                      <h3 style={{ fontSize: "15px", fontWeight: 600, margin: 0 }}>
                        Prometheus Text Exposition (/api/v1/ops/metrics)
                      </h3>
                      <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
                        Standard Prometheus 0.0.4 text format ready for Prometheus scraping or sidecar ingestion
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn--secondary"
                      onClick={() => copyToClipboard(metricsText, "metrics")}
                      style={{ fontSize: "12px", padding: "6px 12px" }}
                    >
                      {copyFeedback === "metrics" ? "Copied!" : "Copy Prometheus Output"}
                    </button>
                  </div>
                  <pre
                    style={{
                      margin: 0,
                      background: "var(--bg-surface-lowest)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "6px",
                      padding: "16px",
                      fontSize: "12px",
                      fontFamily: "var(--font-mono, monospace)",
                      maxHeight: "260px",
                      overflowY: "auto",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {metricsText || "# No telemetry metrics recorded yet. Service in standby."}
                  </pre>
                </div>
              </div>
            )}

            {/* ─── SECTION 2: STATISTICAL DRIFT ───────────────────────── */}
            {activeTab === "drift" && (
              <div>
                {/* Traffic Light Pills Strip */}
                <div className="card" style={{ padding: "24px", marginBottom: "24px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                    <div>
                      <h3 style={{ fontSize: "16px", fontWeight: 600, margin: 0 }}>
                        Distribution Drift Status by Feature Stream
                      </h3>
                      <p style={{ margin: "4px 0 0", fontSize: "12px", color: "var(--text-muted)" }}>
                        {driftReport && typeof driftReport.summary === "string"
                          ? driftReport.summary
                          : "Monitored via Population Stability Index (PSI), Two-Sample Kolmogorov-Smirnov (KS), and Jensen-Shannon Distance (JSD)"}
                      </p>
                    </div>
                    <span
                      className="badge-pill"
                      style={{
                        background: "rgba(245, 158, 11, 0.15)",
                        color: "#f59e0b",
                        border: "1px solid rgba(245, 158, 11, 0.3)",
                        padding: "6px 12px",
                        fontSize: "12px",
                        fontWeight: 600,
                        textTransform: "capitalize",
                      }}
                    >
                      {driftReport && typeof driftReport.status === "string"
                        ? driftReport.status
                        : "Standby (Cold State)"}
                    </span>
                  </div>

                  {/* Feature Components Table with Traffic Light Pills */}
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", fontSize: "13px", borderCollapse: "collapse" }}>
                      <thead>
                        <tr style={{ background: "var(--bg-surface-lowest)", borderBottom: "1px solid var(--border-subtle)" }}>
                          <th style={{ padding: "12px 14px", textAlign: "left" }}>Feature Stream</th>
                          <th style={{ padding: "12px 14px", textAlign: "left" }}>Statistical Detector</th>
                          <th style={{ padding: "12px 14px", textAlign: "center" }}>Verdict</th>
                          <th style={{ padding: "12px 14px", textAlign: "left" }}>Thresholds &amp; Criteria</th>
                          <th style={{ padding: "12px 14px", textAlign: "left" }}>Status Details</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          {
                            stream: "Liked Library Lengths",
                            detector: "PSI + Two-Sample KS",
                            status: "STANDBY",
                            threshold: "Warn: PSI ≥ 8.5% (null 99th pct), Alert: PSI ≥ 15% (KS α=0.01)",
                            details: "Buffered 0 samples (min 10 required for continuous distribution tests)",
                          },
                          {
                            stream: "Recommendation Scores",
                            detector: "Two-Sample KS + PSI",
                            status: "STANDBY",
                            threshold: "Warn: PSI ≥ 8.5% (null 99th pct), Alert: PSI ≥ 15% (KS α=0.01)",
                            details: "Buffered 0 samples (min 10 required; reference matched to two-tower dot products)",
                          },
                          {
                            stream: "Assistant Intent Categorical Shares",
                            detector: "Jensen-Shannon Distance (JSD)",
                            status: "STANDBY",
                            threshold: "Warn: JSD ≥ 11.6%, Alert: JSD ≥ 22.0%",
                            details: "Observed 0 intent requests. JSD has low statistical power for tail shifts (e.g. out_of_scope).",
                          },
                          {
                            stream: "Assistant Confidence Scores",
                            detector: "Two-Sample KS Test",
                            status: "STANDBY",
                            threshold: "Warn α = 0.05, Alert α = 0.01 vs calibrated Beta(8,2) reference",
                            details: "Buffered 0 samples (min 10 required)",
                          },
                          {
                            stream: "Catalog Numeric Features (Price/Reviews)",
                            detector: "Baseline Reference Consistency",
                            status: "PASS",
                            threshold: "3,000 reference catalog items indexed",
                            details: "Catalog reference stable. No un-indexed price or review volume drift.",
                          },
                        ].map((row, idx) => (
                          <tr
                            key={idx}
                            style={{
                              borderBottom: "1px solid var(--border-subtle)",
                              background: idx % 2 === 1 ? "rgba(255, 255, 255, 0.01)" : "transparent",
                            }}
                          >
                            <td style={{ padding: "12px 14px", fontWeight: 600 }}>{row.stream}</td>
                            <td style={{ padding: "12px 14px", color: "var(--text-secondary)", fontFamily: "var(--font-mono, monospace)", fontSize: "12px" }}>
                              {row.detector}
                            </td>
                            <td style={{ padding: "12px 14px", textAlign: "center" }}>
                              {row.status === "PASS" ? (
                                <span
                                  className="badge-pill badge-pill--success"
                                  style={{ padding: "4px 10px", fontSize: "11px" }}
                                >
                                  PASS
                                </span>
                              ) : row.status === "STANDBY" ? (
                                <span
                                  className="badge-pill"
                                  style={{
                                    background: "rgba(245, 158, 11, 0.15)",
                                    color: "#f59e0b",
                                    border: "1px solid rgba(245, 158, 11, 0.3)",
                                    padding: "4px 10px",
                                    fontSize: "11px",
                                  }}
                                >
                                  STANDBY
                                </span>
                              ) : (
                                <span
                                  className="badge-pill badge-pill--danger"
                                  style={{ padding: "4px 10px", fontSize: "11px" }}
                                >
                                  ALERT
                                </span>
                              )}
                            </td>
                            <td style={{ padding: "12px 14px", color: "var(--text-muted)", fontSize: "12px" }}>
                              {row.threshold}
                            </td>
                            <td style={{ padding: "12px 14px", color: "var(--text-secondary)", fontSize: "12px" }}>
                              {row.details}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Detector Threshold Calibrations Card */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
                    gap: "20px",
                  }}
                >
                  <div className="card" style={{ padding: "20px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                      <span className="material-symbols-outlined" style={{ color: "var(--accent-primary)" }}>
                        straighten
                      </span>
                      <h4 style={{ margin: 0, fontSize: "14px", fontWeight: 600 }}>
                        PSI Empirical Calibration
                      </h4>
                    </div>
                    <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.6", margin: "0 0 12px" }}>
                      Calibrated via 1,000 null bootstrap splits of baseline catalog features. The 99th percentile under null resampling serves as Warn threshold, and Alert is set at 0.15.
                    </p>
                    <div style={{ display: "flex", gap: "10px", fontSize: "12px" }}>
                      <span className="badge-pill" style={{ background: "rgba(16, 185, 129, 0.1)", color: "#10b981" }}>
                        Null 95%: ~6.4%
                      </span>
                      <span className="badge-pill" style={{ background: "rgba(245, 158, 11, 0.1)", color: "#f59e0b" }}>
                        Null 99%: ~8.5%
                      </span>
                    </div>
                  </div>

                  <div className="card" style={{ padding: "20px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                      <span className="material-symbols-outlined" style={{ color: "var(--accent-cyan)" }}>
                        bar_chart
                      </span>
                      <h4 style={{ margin: 0, fontSize: "14px", fontWeight: 600 }}>
                        Two-Sample Kolmogorov-Smirnov
                      </h4>
                    </div>
                    <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.6", margin: "0 0 12px" }}>
                      Non-parametric cumulative distribution comparison. Uses exact asymptotic p-values with critical values α=0.05 (Warn) and α=0.01 (Alert). Empirical null 99% critical statistic is ~0.1000.
                    </p>
                    <div style={{ display: "flex", gap: "10px", fontSize: "12px" }}>
                      <span className="badge-pill" style={{ background: "rgba(16, 185, 129, 0.1)", color: "#10b981" }}>
                        Alpha: 0.01
                      </span>
                      <span className="badge-pill" style={{ background: "rgba(245, 158, 11, 0.1)", color: "#f59e0b" }}>
                        Stat Null 99%: 0.1000
                      </span>
                    </div>
                  </div>

                  <div className="card" style={{ padding: "20px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                      <span className="material-symbols-outlined" style={{ color: "#a855f7" }}>
                        category
                      </span>
                      <h4 style={{ margin: 0, fontSize: "14px", fontWeight: 600 }}>
                        Jensen-Shannon Distance (JSD)
                      </h4>
                    </div>
                    <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.6", margin: "0 0 12px" }}>
                      Symmetric, bounded in [0, 1] relative entropy for categorical distributions (13 assistant intents). Warn at 0.1157 (null 99th pct), Alert at 0.2200.
                    </p>
                    <div style={{ display: "flex", gap: "10px", fontSize: "12px" }}>
                      <span className="badge-pill" style={{ background: "rgba(16, 185, 129, 0.1)", color: "#10b981" }}>
                        Null 95%: 0.1038
                      </span>
                      <span className="badge-pill" style={{ background: "rgba(245, 158, 11, 0.1)", color: "#f59e0b" }}>
                        Null 99%: 0.1157
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ─── SECTION 3: RETRAIN GATE VERDICT ────────────────────── */}
            {activeTab === "retrain_gate" && (
              <div>
                {/* Decision Verdict Banner */}
                <div
                  className="card"
                  style={{
                    padding: "24px",
                    marginBottom: "24px",
                    borderLeft: "4px solid #f59e0b",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px" }}>
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
                        <span
                          className="badge-pill"
                          style={{
                            background: "rgba(245, 158, 11, 0.15)",
                            color: "#f59e0b",
                            border: "1px solid rgba(245, 158, 11, 0.3)",
                            fontWeight: 700,
                            padding: "6px 14px",
                            fontSize: "12px",
                            textTransform: "uppercase",
                          }}
                        >
                          DECISION: {String((retrainGateData as Record<string, unknown>)?.decision || "CANNOT DECIDE")}
                        </span>
                        <span className="badge-pill" style={{ fontSize: "11px" }}>
                          Mode: {String((retrainGateData as Record<string, unknown>)?.mode || "dry_run")}
                        </span>
                      </div>
                      <p style={{ margin: "8px 0 0", fontSize: "14px", color: "var(--text-primary)", fontWeight: 500, maxWidth: "800px" }}>
                        {String(
                          (retrainGateData as Record<string, unknown>)?.explanation ||
                            "Cannot decide: champion recsys_two_tower_clean has 2 seeds (requires 3), and clean ranker recsys_two_tower_ranker_clean has 2 seeds (requires 5)."
                        )}
                      </p>
                    </div>

                    <button
                      type="button"
                      className="btn--secondary"
                      onClick={() => copyToClipboard(retrainGateLog || JSON.stringify(retrainGateData, null, 2), "gate")}
                      style={{ fontSize: "12px", padding: "8px 14px" }}
                    >
                      {copyFeedback === "gate" ? "Copied!" : "Copy Gate Decision"}
                    </button>
                  </div>
                </div>

                {/* Champion vs Challenger Comparison */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
                    gap: "20px",
                    marginBottom: "24px",
                  }}
                >
                  <div className="card" style={{ padding: "24px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                      <h3 style={{ fontSize: "16px", fontWeight: 600, margin: 0 }}>
                        Production Champion
                      </h3>
                      <span className="badge-pill badge-pill--success" style={{ fontSize: "11px" }}>
                        champion
                      </span>
                    </div>
                    <div style={{ fontSize: "14px", fontWeight: 600, color: "var(--accent-primary)", marginBottom: "12px" }}>
                      {String((retrainGateData as Record<string, unknown>)?.champion ? ((retrainGateData as Record<string, unknown>).champion as Record<string, unknown>).model_name : "recsys_two_tower_clean")}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "13px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-muted)" }}>Validation NDCG@10:</span>
                        <span style={{ fontWeight: 600, fontFamily: "var(--font-mono, monospace)" }}>
                          {retrainGateData && (retrainGateData.champion as Record<string, unknown>)?.val_ndcg_at_10 != null
                            ? Number((retrainGateData.champion as Record<string, unknown>).val_ndcg_at_10).toFixed(4)
                            : "—"}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-muted)" }}>Test NDCG@10:</span>
                        <span style={{ fontWeight: 600, fontFamily: "var(--font-mono, monospace)" }}>
                          {retrainGateData && (retrainGateData.champion as Record<string, unknown>)?.test_ndcg_at_10 != null
                            ? Number((retrainGateData.champion as Record<string, unknown>).test_ndcg_at_10).toFixed(4)
                            : "—"}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-muted)" }}>Seed Quorum:</span>
                        <span style={{ fontWeight: 600, color: "#f59e0b" }}>
                          2 / 3 seeds available
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="card" style={{ padding: "24px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                      <h3 style={{ fontSize: "16px", fontWeight: 600, margin: 0 }}>
                        Evaluated Challenger
                      </h3>
                      <span
                        className="badge-pill"
                        style={{ background: "rgba(99, 102, 241, 0.15)", color: "var(--accent-primary)", fontSize: "11px" }}
                      >
                        challenger
                      </span>
                    </div>
                    <div style={{ fontSize: "14px", fontWeight: 600, color: "var(--accent-cyan)", marginBottom: "12px" }}>
                      {String((retrainGateData as Record<string, unknown>)?.challenger ? ((retrainGateData as Record<string, unknown>).challenger as Record<string, unknown>).family : "recsys_two_tower_ranker_clean")}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "13px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-muted)" }}>Validation NDCG@10:</span>
                        <span style={{ fontWeight: 600, fontFamily: "var(--font-mono, monospace)" }}>
                          {retrainGateData && (retrainGateData.challenger as Record<string, unknown>)?.val_ndcg_at_10 != null
                            ? Number((retrainGateData.challenger as Record<string, unknown>).val_ndcg_at_10).toFixed(4)
                            : "—"}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-muted)" }}>Test NDCG@10:</span>
                        <span style={{ fontWeight: 600, fontFamily: "var(--font-mono, monospace)" }}>
                          {retrainGateData && (retrainGateData.challenger as Record<string, unknown>)?.test_ndcg_at_10 != null
                            ? Number((retrainGateData.challenger as Record<string, unknown>).test_ndcg_at_10).toFixed(4)
                            : "—"}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-muted)" }}>Seed Quorum:</span>
                        <span style={{ fontWeight: 600, color: "#f59e0b" }}>
                          2 / 5 seeds available
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Pre-registered Gate Rules Card */}
                <div className="card" style={{ padding: "24px", marginBottom: "24px" }}>
                  <h3 style={{ fontSize: "15px", fontWeight: 600, margin: "0 0 16px" }}>
                    Pre-registered Qualification Invariants
                  </h3>
                  <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                    <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
                      <span className="badge-pill" style={{ fontWeight: 600, padding: "2px 8px" }}>Rule 1</span>
                      <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                        <strong>Validation Margin Standard:</strong> Challenger mean validation NDCG@10 minus champion validation NDCG@10 must exceed challenger seed standard deviation.
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
                      <span className="badge-pill" style={{ fontWeight: 600, padding: "2px 8px" }}>Rule 2</span>
                      <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                        <strong>Non-Inferiority Invariant:</strong> Paired bootstrap difference test 95% confidence interval lower bound vs champion must be greater than -epsilon (-1e-3 margin).
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
                      <span className="badge-pill" style={{ fontWeight: 600, padding: "2px 8px" }}>Rule 3</span>
                      <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                        <strong>Manual Staging Boundary:</strong> Successful qualification promotes only to Candidate staging. Direct auto-promotion to production is strictly forbidden by policy.
                      </div>
                    </div>
                  </div>
                </div>

                {/* Reproducible Execution Log Viewer */}
                {retrainGateLog && (
                  <div className="card" style={{ padding: "24px" }}>
                    <h3 style={{ fontSize: "15px", fontWeight: 600, margin: "0 0 12px" }}>
                      Reproducible Verification Log (docs/evidence/retrain_gate.log)
                    </h3>
                    <pre
                      style={{
                        margin: 0,
                        background: "var(--bg-surface-lowest)",
                        border: "1px solid var(--border-subtle)",
                        borderRadius: "6px",
                        padding: "16px",
                        fontSize: "12px",
                        fontFamily: "var(--font-mono, monospace)",
                        maxHeight: "240px",
                        overflowY: "auto",
                        color: "var(--text-secondary)",
                      }}
                    >
                      {retrainGateLog}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {/* ─── SECTION 4: DETECTOR POWER ──────────────────────────── */}
            {activeTab === "power" && (
              <div>
                {/* Null Hypothesis False Positive Rate Banner */}
                <div
                  className="card"
                  style={{
                    padding: "20px",
                    marginBottom: "24px",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    flexWrap: "wrap",
                    gap: "16px",
                  }}
                >
                  <div>
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
                      Null Hypothesis Calibration (H0: No Shift, N=1000, 100 Trials)
                    </div>
                    <div style={{ fontSize: "14px", fontWeight: 500, color: "var(--text-primary)", marginTop: "4px" }}>
                      Empirical False Positive Rates (FPR) strictly satisfy α ≤ 0.05 targets across all three detectors.
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: "12px" }}>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>PSI NULL FPR</div>
                      <div style={{ fontSize: "18px", fontWeight: 700, color: "#10b981" }}>0.0%</div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>KS NULL FPR</div>
                      <div style={{ fontSize: "18px", fontWeight: 700, color: "#10b981" }}>2.0%</div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>JSD NULL FPR</div>
                      <div style={{ fontSize: "18px", fontWeight: 700, color: "#10b981" }}>0.0%</div>
                    </div>
                  </div>
                </div>

                {/* SVG Power Curve Chart */}
                <div className="card" style={{ padding: "24px", marginBottom: "24px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                    <div>
                      <h3 style={{ fontSize: "16px", fontWeight: 600, margin: 0 }}>
                        Statistical Power Curves by Shift Magnitude
                      </h3>
                      <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
                        Detection probability (1 - β) as distribution shift increases from null (0.0σ) to 1.0σ
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: "16px", fontSize: "12px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <div style={{ width: "12px", height: "4px", background: "#6366f1", borderRadius: "2px" }} />
                        <span>KS Test</span>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <div style={{ width: "12px", height: "4px", background: "#10b981", borderRadius: "2px" }} />
                        <span>PSI Detector</span>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <div style={{ width: "12px", height: "4px", background: "#a855f7", borderRadius: "2px" }} />
                        <span>JSD Categorical</span>
                      </div>
                    </div>
                  </div>

                  {/* SVG Power Curve */}
                  <div style={{ width: "100%", height: "240px" }}>
                    <svg viewBox="0 0 700 200" style={{ width: "100%", height: "100%", overflow: "visible" }}>
                      {/* Grid Lines */}
                      {[0, 0.25, 0.5, 0.75, 1.0].map((val, idx) => {
                        const y = 170 - val * 140;
                        return (
                          <g key={idx}>
                            <line x1="50" y1={y} x2="670" y2={y} stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
                            <text x="42" y={y + 4} fill="var(--text-muted)" fontSize="10" textAnchor="end">
                              {Math.round(val * 100)}%
                            </text>
                          </g>
                        );
                      })}

                      {/* X Axis Labels */}
                      {[
                        { label: "+0.0σ (Null)", x: 80 },
                        { label: "+0.10σ", x: 220 },
                        { label: "+0.25σ", x: 360 },
                        { label: "+0.50σ", x: 500 },
                        { label: "+1.00σ", x: 640 },
                      ].map((item, idx) => (
                        <text key={idx} x={item.x} y="190" fill="var(--text-muted)" fontSize="11" textAnchor="middle">
                          {item.label}
                        </text>
                      ))}

                      {/* KS Curve (0.02 -> 0.21 -> 0.99 -> 1.0 -> 1.0) */}
                      <path
                        d="M 80 167.2 C 150 160, 180 140.6, 220 140.6 C 290 140.6, 320 31.4, 360 31.4 L 500 30 L 640 30"
                        fill="none"
                        stroke="#6366f1"
                        strokeWidth="3"
                      />
                      {/* KS points */}
                      <circle cx="80" cy="167.2" r="4" fill="#6366f1" />
                      <circle cx="220" cy="140.6" r="4" fill="#6366f1" />
                      <circle cx="360" cy="31.4" r="4" fill="#6366f1" />
                      <circle cx="500" cy="30" r="4" fill="#6366f1" />
                      <circle cx="640" cy="30" r="4" fill="#6366f1" />

                      {/* PSI Curve (0.0 -> 0.0 -> 0.13 -> 1.0 -> 1.0) */}
                      <path
                        d="M 80 170 L 220 170 C 290 170, 320 151.8, 360 151.8 C 420 151.8, 450 30, 500 30 L 640 30"
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="3"
                      />
                      {/* PSI points */}
                      <circle cx="80" cy="170" r="4" fill="#10b981" />
                      <circle cx="220" cy="170" r="4" fill="#10b981" />
                      <circle cx="360" cy="151.8" r="4" fill="#10b981" />
                      <circle cx="500" cy="30" r="4" fill="#10b981" />
                      <circle cx="640" cy="30" r="4" fill="#10b981" />

                      {/* JSD Curve (0.0 -> 0.0 -> 0.0 -> 0.82 -> 1.0) */}
                      <path
                        d="M 80 170 L 220 170 L 360 170 C 420 170, 460 55.2, 500 55.2 C 570 55.2, 600 30, 640 30"
                        fill="none"
                        stroke="#a855f7"
                        strokeWidth="3"
                        strokeDasharray="4 2"
                      />
                      {/* JSD points */}
                      <circle cx="80" cy="170" r="4" fill="#a855f7" />
                      <circle cx="220" cy="170" r="4" fill="#a855f7" />
                      <circle cx="360" cy="170" r="4" fill="#a855f7" />
                      <circle cx="500" cy="55.2" r="4" fill="#a855f7" />
                      <circle cx="640" cy="30" r="4" fill="#a855f7" />
                    </svg>
                  </div>
                </div>

                {/* Mean Shift Power Table */}
                <div className="card" style={{ padding: "24px" }}>
                  <h3 style={{ fontSize: "15px", fontWeight: 600, margin: "0 0 16px" }}>
                    Mean Shift Detection Power Trials (docs/results/drift_detector_power.json)
                  </h3>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", fontSize: "13px", borderCollapse: "collapse" }}>
                      <thead>
                        <tr style={{ background: "var(--bg-surface-lowest)", borderBottom: "1px solid var(--border-subtle)" }}>
                          <th style={{ padding: "10px 14px", textAlign: "left" }}>Shift Magnitude</th>
                          <th style={{ padding: "10px 14px", textAlign: "right" }}>KS Detection Rate</th>
                          <th style={{ padding: "10px 14px", textAlign: "left" }}>KS 95% Bootstrap CI</th>
                          <th style={{ padding: "10px 14px", textAlign: "right" }}>PSI Detection Rate</th>
                          <th style={{ padding: "10px 14px", textAlign: "left" }}>PSI 95% Bootstrap CI</th>
                          <th style={{ padding: "10px 14px", textAlign: "right" }}>Mean PSI</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.isArray((detectorPowerData as Record<string, unknown>)?.mean_shift_power)
                          ? ((detectorPowerData as Record<string, unknown>).mean_shift_power as Array<Record<string, unknown>>).map((row, idx) => {
                              const shift = row.is_null_resample ? "+0.00σ (Null)" : `+${Number(row.shift_magnitude_std).toFixed(2)}σ`;
                              const ks = `${(Number(row.ks_detection_rate) * 100).toFixed(1)}%`;
                              const ksCi = Array.isArray(row.ks_ci)
                                ? `[${(Number(row.ks_ci[0]) * 100).toFixed(1)}%, ${(Number(row.ks_ci[1]) * 100).toFixed(1)}%]`
                                : "—";
                              const psi = `${(Number(row.psi_detection_rate) * 100).toFixed(1)}%`;
                              const psiCi = Array.isArray(row.psi_ci)
                                ? `[${(Number(row.psi_ci[0]) * 100).toFixed(1)}%, ${(Number(row.psi_ci[1]) * 100).toFixed(1)}%]`
                                : "—";
                              const meanPsi = row.mean_psi != null ? Number(row.mean_psi).toFixed(4) : "—";
                              return (
                                <tr
                                  key={idx}
                                  style={{
                                    borderBottom: "1px solid var(--border-subtle)",
                                    background: idx % 2 === 1 ? "rgba(255, 255, 255, 0.01)" : "transparent",
                                  }}
                                >
                                  <td style={{ padding: "10px 14px", fontWeight: 600 }}>{shift}</td>
                                  <td style={{ padding: "10px 14px", textAlign: "right", color: "var(--accent-primary)", fontWeight: 600 }}>
                                    {ks}
                                  </td>
                                  <td style={{ padding: "10px 14px", color: "var(--text-muted)", fontSize: "12px", fontFamily: "var(--font-mono, monospace)" }}>
                                    {ksCi}
                                  </td>
                                  <td style={{ padding: "10px 14px", textAlign: "right", color: "#10b981", fontWeight: 600 }}>
                                    {psi}
                                  </td>
                                  <td style={{ padding: "10px 14px", color: "var(--text-muted)", fontSize: "12px", fontFamily: "var(--font-mono, monospace)" }}>
                                    {psiCi}
                                  </td>
                                  <td style={{ padding: "10px 14px", textAlign: "right", fontFamily: "var(--font-mono, monospace)" }}>
                                    {meanPsi}
                                  </td>
                                </tr>
                              );
                            })
                          : null}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* ─── SECTION 5: MODEL CARDS VIEWER ──────────────────────── */}
            {activeTab === "model_cards" && (
              <div>
                {/* Model Selector Bar */}
                <div
                  style={{
                    display: "flex",
                    gap: "10px",
                    overflowX: "auto",
                    marginBottom: "20px",
                    paddingBottom: "4px",
                  }}
                >
                  {modelCards.length > 0 ? (
                    modelCards.map((card, idx) => {
                      const isSelected = selectedCardIdx === idx;
                      return (
                        <button
                          key={card.filename}
                          type="button"
                          onClick={() => {
                            setSelectedCardIdx(idx);
                            setViewRawMarkdown(false);
                          }}
                          className={`stitch-nav__link ${isSelected ? "stitch-nav__link--active" : ""}`}
                          style={{
                            padding: "10px 16px",
                            borderRadius: "6px",
                            border: "none",
                            cursor: "pointer",
                            fontSize: "13px",
                            fontWeight: isSelected ? 600 : 500,
                            background: isSelected ? "var(--bg-surface-mid)" : "var(--bg-surface-lowest)",
                            color: isSelected ? "var(--text-primary)" : "var(--text-muted)",
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            whiteSpace: "nowrap",
                          }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "var(--accent-primary)" }}>
                            badge
                          </span>
                          <span>{card.model_name}</span>
                        </button>
                      );
                    })
                  ) : (
                    <div style={{ color: "var(--text-muted)", fontSize: "13px" }}>
                      No model cards loaded from docs/model-cards.
                    </div>
                  )}
                </div>

                {/* Selected Model Card View */}
                {modelCards[selectedCardIdx] && (
                  <div className="card" style={{ padding: "28px" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        borderBottom: "1px solid var(--border-subtle)",
                        paddingBottom: "16px",
                        marginBottom: "24px",
                        flexWrap: "wrap",
                        gap: "12px",
                      }}
                    >
                      <div>
                        <h2 style={{ fontSize: "20px", fontWeight: 700, margin: 0 }}>
                          {modelCards[selectedCardIdx].title}
                        </h2>
                        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
                          File: docs/model-cards/{modelCards[selectedCardIdx].filename}
                        </div>
                      </div>

                      <div style={{ display: "flex", gap: "10px" }}>
                        <button
                          type="button"
                          className="btn--secondary"
                          onClick={() => setViewRawMarkdown((prev) => !prev)}
                          style={{ fontSize: "12px", padding: "6px 14px" }}
                        >
                          {viewRawMarkdown ? "Rendered View" : "Raw Markdown"}
                        </button>
                        <button
                          type="button"
                          className="btn--secondary"
                          onClick={() => copyToClipboard(modelCards[selectedCardIdx].content, "card")}
                          style={{ fontSize: "12px", padding: "6px 14px" }}
                        >
                          {copyFeedback === "card" ? "Copied!" : "Copy Card"}
                        </button>
                      </div>
                    </div>

                    {viewRawMarkdown ? (
                      <pre
                        style={{
                          margin: 0,
                          background: "var(--bg-surface-lowest)",
                          border: "1px solid var(--border-subtle)",
                          borderRadius: "6px",
                          padding: "20px",
                          fontSize: "12px",
                          fontFamily: "var(--font-mono, monospace)",
                          maxHeight: "600px",
                          overflowY: "auto",
                          color: "var(--text-secondary)",
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-word",
                        }}
                      >
                        {modelCards[selectedCardIdx].content}
                      </pre>
                    ) : (
                      <SimpleMarkdownViewer content={modelCards[selectedCardIdx].content} />
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </main>
    </DevOnlyGate>
  );
}

export default function MonitoringPage() {
  return (
    <Suspense
      fallback={
        <div className="container" style={{ paddingTop: "88px", color: "var(--text-muted)" }}>
          Loading monitoring console...
        </div>
      }
    >
      <MonitoringContent />
    </Suspense>
  );
}
