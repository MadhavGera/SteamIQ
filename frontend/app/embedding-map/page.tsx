"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  api,
  RecsysMapPoint,
  RecsysMapResponse,
  RecsysNeighborItem,
} from "@/lib/api";
import { RecsysEmptyState } from "@/components/recsys/RecsysEmptyState";
import { DevOnlyGate } from "@/components/DevOnlyGate";

// 20 Distinct, harmonious HSL colors for clusters
const CLUSTER_COLORS = [
  "#66c0f4", "#59bf40", "#f59e0b", "#ec4899", "#8b5cf6",
  "#14b8a6", "#f97316", "#06b6d4", "#84cc16", "#a855f7",
  "#3b82f6", "#10b981", "#ef4444", "#eab308", "#6366f1",
  "#d946ef", "#22c55e", "#0ea5e9", "#f43f5e", "#64748b",
];

function getClusterColor(clusterId: number | null): string {
  if (clusterId === null || clusterId === undefined || clusterId < 0) return "#8f98a0";
  return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

function EmbeddingMapContent() {
  const router = useRouter();
  const [mapData, setMapData] = useState<RecsysMapResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedPoint, setSelectedPoint] = useState<RecsysMapPoint | null>(null);
  const [neighbors, setNeighbors] = useState<RecsysNeighborItem[]>([]);
  const [isLoadingNeighbors, setIsLoadingNeighbors] = useState(false);
  const [hoveredPoint, setHoveredPoint] = useState<RecsysMapPoint | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [activeClusterFilter, setActiveClusterFilter] = useState<number | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Pan & Zoom Transform State
  const transformRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  // 1. Fetch 2D UMAP Map Points on Mount
  useEffect(() => {
    let isMounted = true;
    api
      .recsysMap()
      .then((res) => {
        if (isMounted) {
          setMapData(res);
          setIsLoading(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // 2. Fetch nearest neighbors when a point is selected
  useEffect(() => {
    if (!selectedPoint) {
      setNeighbors([]);
      return;
    }

    let isMounted = true;
    setIsLoadingNeighbors(true);
    api
      .recsysNeighbors(selectedPoint.app_id, 8)
      .then((res) => {
        if (isMounted) {
          setNeighbors(res.neighbors);
          setIsLoadingNeighbors(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setIsLoadingNeighbors(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedPoint]);

  // Extract unique cluster legends
  const clusterMap = new Map<number, string>();
  mapData?.points.forEach((p) => {
    if (p.cluster_id !== null && p.cluster_id !== undefined && !clusterMap.has(p.cluster_id)) {
      // HONEST-FALLBACK: Default cluster label if omitted
      clusterMap.set(p.cluster_id, p.cluster_label ?? `Cluster ${p.cluster_id}`);
    }
  });
  const clusters = Array.from(clusterMap.entries()).sort((a, b) => a[0] - b[0]);

  // Normalized coordinate helper
  const normalizeCoords = useCallback((points: RecsysMapPoint[], width: number, height: number) => {
    if (points.length === 0) return [];

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach((p) => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const padding = 60;

    return points.map((p) => ({
      ...p,
      canvasX: padding + ((p.x - minX) / rangeX) * (width - 2 * padding),
      canvasY: padding + ((p.y - minY) / rangeY) * (height - 2 * padding),
    }));
  }, []);

  // Main Render Loop onto Canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mapData || mapData.points.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const { scale, offsetX, offsetY } = transformRef.current;

    // Clear main canvas
    ctx.clearRect(0, 0, width, height);

    // Background grid
    ctx.save();
    ctx.fillStyle = "#16202d";
    ctx.fillRect(0, 0, width, height);

    // Apply pan & zoom
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    const normPoints = normalizeCoords(mapData.points, width, height);

    // Draw points
    normPoints.forEach((p) => {
      const isFiltered = activeClusterFilter !== null && p.cluster_id !== activeClusterFilter;
      const isSelected = selectedPoint?.app_id === p.app_id;
      const isHovered = hoveredPoint?.app_id === p.app_id;

      ctx.beginPath();
      const radius = isSelected ? 8 : isHovered ? 6 : 3.5;
      ctx.arc(p.canvasX, p.canvasY, radius, 0, 2 * Math.PI);

      if (isFiltered) {
        ctx.fillStyle = "rgba(60, 70, 80, 0.2)";
      } else {
        ctx.fillStyle = getClusterColor(p.cluster_id);
      }

      ctx.fill();

      if (isSelected) {
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      } else if (isHovered) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }
    });

    ctx.restore();
  }, [mapData, selectedPoint, hoveredPoint, activeClusterFilter, normalizeCoords]);

  // Redraw when dependencies change
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Handle Canvas Resize
  useEffect(() => {
    function resize() {
      const canvas = canvasRef.current;
      if (canvas && canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = Math.max(560, window.innerHeight - 260);
        drawCanvas();
      }
    }
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [drawCanvas]);

  // Mouse Interactivity: Pan, Zoom, Hover, Click
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const newScale = Math.max(0.5, Math.min(15, transformRef.current.scale * zoomFactor));

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    transformRef.current = {
      scale: newScale,
      offsetX: mouseX - (mouseX - transformRef.current.offsetX) * (newScale / transformRef.current.scale),
      offsetY: mouseY - (mouseY - transformRef.current.offsetY) * (newScale / transformRef.current.scale),
    };
    drawCanvas();
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDraggingRef.current = true;
    dragStartRef.current = {
      x: e.clientX - transformRef.current.offsetX,
      y: e.clientY - transformRef.current.offsetY,
    };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !mapData) return;

    if (isDraggingRef.current) {
      transformRef.current.offsetX = e.clientX - dragStartRef.current.x;
      transformRef.current.offsetY = e.clientY - dragStartRef.current.y;
      drawCanvas();
      return;
    }

    // Hover Detection
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const { scale, offsetX, offsetY } = transformRef.current;
    const normPoints = normalizeCoords(mapData.points, canvas.width, canvas.height);

    let closest: RecsysMapPoint | null = null;
    let minDistance = 12 / scale; // Radius in data space

    normPoints.forEach((p) => {
      const screenX = p.canvasX * scale + offsetX;
      const screenY = p.canvasY * scale + offsetY;
      const dist = Math.hypot(mouseX - screenX, mouseY - screenY);
      if (dist < minDistance * scale) {
        minDistance = dist / scale;
        closest = p;
      }
    });

    if (closest !== hoveredPoint) {
      setHoveredPoint(closest);
      if (closest) {
        setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      } else {
        setTooltipPos(null);
      }
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleClick = () => {
    if (hoveredPoint) {
      setSelectedPoint(hoveredPoint);
    }
  };

  const handleResetZoom = () => {
    transformRef.current = { scale: 1, offsetX: 0, offsetY: 0 };
    drawCanvas();
  };

  // "Add to my library" handler
  const handleAddToLibrary = (appId: number) => {
    if (typeof window !== "undefined") {
      sessionStorage.setItem("recsys_add_game_id", String(appId));
      router.push("/");
    }
  };

  if (!isLoading && (!mapData || mapData.points.length === 0)) {
    return (
      <main className="container" style={{ paddingTop: "88px", paddingBottom: "64px" }}>
        <RecsysEmptyState
          command="make recsys-project"
          title="Item Embedding Map Not Projected"
          message="2D UMAP projection and KMeans semantic clusters have not been materialized. Run the projection script below:"
        />
      </main>
    );
  }

  return (
    <main className="container" style={{ paddingTop: "88px", paddingBottom: "64px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          flexWrap: "wrap",
          gap: "16px",
          marginBottom: "20px",
          borderBottom: "1px solid var(--border-subtle)",
          paddingBottom: "16px",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
            <span className="badge-pill badge-pill--neutral" style={{ color: "var(--accent-primary)" }}>
              2D UMAP Embedding Space
            </span>
            {mapData && (
              <span className="badge-pill badge-pill--neutral" style={{ fontFamily: "var(--font-mono, monospace)" }}>
                {mapData.total_points.toLocaleString()} points • 20 semantic clusters
              </span>
            )}
          </div>
          <h1 style={{ fontSize: "28px", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
            Item Embedding Map
          </h1>
          <p style={{ fontSize: "14px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Interactive 2D UMAP projection of 64-dimensional Two-Tower item representations. Drag to pan, scroll to zoom, click to inspect nearest neighbors.
          </p>
        </div>

        {/* Zoom Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <button
            type="button"
            onClick={handleResetZoom}
            className="btn btn--secondary"
            style={{ fontSize: "12px", padding: "6px 12px" }}
            aria-label="Reset zoom and center view"
          >
            <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
              center_focus_strong
            </span>
            Reset View
          </button>
        </div>
      </div>

      {/* Cluster Legend Filter Pills */}
      {clusters.length > 0 && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "6px",
            marginBottom: "16px",
            maxHeight: "80px",
            overflowY: "auto",
            padding: "4px 0",
          }}
        >
          <button
            type="button"
            onClick={() => setActiveClusterFilter(null)}
            className="badge-pill"
            style={{
              backgroundColor: activeClusterFilter === null ? "var(--accent-primary)" : "var(--bg-surface)",
              color: activeClusterFilter === null ? "var(--bg-surface-lowest)" : "var(--text-secondary)",
              border: "1px solid var(--border-subtle)",
              cursor: "pointer",
            }}
          >
            All Clusters ({clusters.length})
          </button>
          {clusters.map(([id, label]) => {
            const isSelected = activeClusterFilter === id;
            return (
              <button
                key={id}
                type="button"
                onClick={() => setActiveClusterFilter(isSelected ? null : id)}
                className="badge-pill"
                style={{
                  backgroundColor: isSelected ? getClusterColor(id) : "var(--bg-surface)",
                  color: isSelected ? "#0e141b" : "var(--text-primary)",
                  border: `1px solid ${getClusterColor(id)}`,
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    backgroundColor: getClusterColor(id),
                  }}
                />
                <span>{label}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Map Layout Grid: Canvas + Slide-out Details Side Panel */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: selectedPoint ? "1fr 340px" : "1fr",
          gap: "20px",
          alignItems: "start",
        }}
      >
        {/* Canvas Container */}
        <div
          style={{
            position: "relative",
            width: "100%",
            borderRadius: "var(--radius-card)",
            overflow: "hidden",
            border: "1px solid var(--border-subtle)",
            backgroundColor: "var(--bg-surface-low)",
          }}
        >
          {isLoading && (
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: "rgba(22, 32, 45, 0.85)",
                zIndex: 10,
              }}
            >
              <div className="loading-bars" style={{ marginBottom: "12px" }}>
                <span />
                <span />
                <span />
              </div>
              <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                Rendering 3,000 UMAP embedding vectors...
              </p>
            </div>
          )}

          <canvas
            ref={canvasRef}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleClick}
            style={{
              display: "block",
              width: "100%",
              cursor: isDraggingRef.current ? "grabbing" : hoveredPoint ? "pointer" : "grab",
            }}
          />

          {/* Hover Tooltip */}
          {hoveredPoint && tooltipPos && (
            <div
              style={{
                position: "absolute",
                top: tooltipPos.y - 48,
                left: tooltipPos.x + 12,
                backgroundColor: "rgba(14, 20, 27, 0.95)",
                border: "1px solid var(--border-strong)",
                borderRadius: "var(--radius-sm)",
                padding: "6px 10px",
                fontSize: "12px",
                color: "var(--text-primary)",
                pointerEvents: "none",
                zIndex: 30,
                boxShadow: "var(--shadow-dropdown)",
                whiteSpace: "nowrap",
              }}
            >
              <div style={{ fontWeight: 700 }}>{hoveredPoint.title}</div>
              <div style={{ fontSize: "11px", color: getClusterColor(hoveredPoint.cluster_id) }}>
                {hoveredPoint.cluster_label || `Cluster ${hoveredPoint.cluster_id}`}
              </div>
            </div>
          )}
        </div>

        {/* Selected Game Side Panel */}
        {selectedPoint && (
          <aside
            className="card"
            style={{
              padding: "20px",
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border-subtle)",
              display: "flex",
              flexDirection: "column",
              gap: "14px",
            }}
          >
            {/* Header with Close */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
              <div>
                <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px", color: getClusterColor(selectedPoint.cluster_id) }}>
                  {selectedPoint.cluster_label || `Cluster ${selectedPoint.cluster_id}`}
                </span>
                <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
                  {selectedPoint.title}
                </h3>
              </div>
              <button
                onClick={() => setSelectedPoint(null)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-muted)",
                  cursor: "pointer",
                }}
                aria-label="Close side panel"
              >
                <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
                  close
                </span>
              </button>
            </div>

            {/* Game Cover Image */}
            {selectedPoint.header_image_url && (
              <div
                style={{
                  width: "100%",
                  height: "120px",
                  borderRadius: "var(--radius-sm)",
                  overflow: "hidden",
                  backgroundColor: "var(--bg-surface-lowest)",
                }}
              >
                <img
                  src={selectedPoint.header_image_url}
                  alt={selectedPoint.title}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
              </div>
            )}

            {/* Action: Add to Library */}
            <button
              type="button"
              onClick={() => handleAddToLibrary(selectedPoint.app_id)}
              className="btn btn--primary"
              style={{ width: "100%", justifyContent: "center" }}
              aria-label={`Add ${selectedPoint.title} to seed library`}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                add_to_photos
              </span>
              Add to My Library
            </button>

            {/* Tags */}
            {selectedPoint.tags && selectedPoint.tags.length > 0 && (
              <div>
                <div style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "4px" }}>
                  Distinctive Tags
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                  {selectedPoint.tags.map((t) => (
                    <span
                      key={t}
                      style={{
                        fontSize: "11px",
                        padding: "2px 6px",
                        backgroundColor: "var(--bg-surface-low)",
                        borderRadius: "var(--radius-sm)",
                        color: "var(--text-secondary)",
                      }}
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Cosine Nearest Neighbors in Two-Tower Embedding Space */}
            <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "14px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <h4 style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-primary)" }}>
                  Cosine Nearest Neighbors
                </h4>
                <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>Two-Tower 64-d</span>
              </div>

              {isLoadingNeighbors ? (
                <div style={{ padding: "16px 0", textAlign: "center" }}>
                  <div className="loading-bars" style={{ height: "14px", margin: "0 auto" }}>
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              ) : neighbors.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {neighbors.map((nb) => (
                    <div
                      key={nb.app_id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "6px 8px",
                        backgroundColor: "var(--bg-surface-low)",
                        borderRadius: "var(--radius-sm)",
                        fontSize: "12px",
                      }}
                    >
                      <Link
                        href={`/games/${nb.app_id}?tab=overview`}
                        style={{
                          maxWidth: "180px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          color: "var(--text-primary)",
                          fontWeight: 600,
                        }}
                      >
                        {nb.title}
                      </Link>
                      <span
                        style={{
                          fontFamily: "var(--font-mono, monospace)",
                          color: "var(--accent-primary)",
                          fontSize: "11px",
                        }}
                      >
                        {nb.similarity.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>No neighbors found.</div>
              )}
            </div>
          </aside>
        )}
      </div>
    </main>
  );
}

export default function EmbeddingMapPage() {
  return (
    <DevOnlyGate pageTitle="Item Embedding Map">
      <EmbeddingMapContent />
    </DevOnlyGate>
  );
}
