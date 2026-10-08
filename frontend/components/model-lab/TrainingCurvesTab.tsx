"use client";

import React from "react";
import { RecsysEmptyState } from "../recsys/RecsysEmptyState";

interface EpochMetric {
  epoch?: number;
  iteration?: number;
  train_loss?: number;
  val_loss?: number;
  val_ndcg_at_10?: number;
  eval_ndcg?: number;
}

interface TrainingCurvesPayload {
  two_tower_curves?: EpochMetric[];
  ranker_curves?: EpochMetric[];
  metadata?: Record<string, unknown>;
}

interface TrainingCurvesTabProps {
  payload: TrainingCurvesPayload | null;
}

/** Inline SVG line chart for loss / metric per epoch */
function EpochLineChart({
  points,
  title,
  metricKey,
  yLabel,
  color = "var(--accent-primary)",
}: {
  points: EpochMetric[];
  title: string;
  metricKey: keyof EpochMetric;
  yLabel: string;
  color?: string;
}) {
  const validPoints = points.filter((p) => typeof p[metricKey] === "number");
  if (validPoints.length === 0) return null;

  const width = 480;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 35, left: 55 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const values = validPoints.map((p) => p[metricKey] as number);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 1;

  const getX = (i: number) => padding.left + (i / (validPoints.length - 1 || 1)) * plotWidth;
  const getY = (v: number) => padding.top + plotHeight - ((v - minVal) / range) * plotHeight;

  const pathD = validPoints.reduce(
    (acc, p, i) => `${acc} ${i === 0 ? "M" : "L"} ${getX(i)} ${getY(p[metricKey] as number)}`,
    ""
  );

  return (
    <div className="card" style={{ padding: "16px" }}>
      <h4 style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "8px" }}>
        {title}
      </h4>
      <div style={{ overflowX: "auto" }}>
        <svg width={width} height={height} style={{ backgroundColor: "var(--bg-surface-low)", borderRadius: "var(--radius-sm)" }} aria-label={title}>
          {/* Axis lines */}
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotHeight} stroke="var(--border-subtle)" />
          <line x1={padding.left} y1={padding.top + plotHeight} x2={padding.left + plotWidth} y2={padding.top + plotHeight} stroke="var(--border-subtle)" />

          {/* Curve path */}
          <path d={pathD} fill="none" stroke={color} strokeWidth="2" />

          {/* Points */}
          {validPoints.map((p, i) => (
            <circle key={i} cx={getX(i)} cy={getY(p[metricKey] as number)} r="3" fill={color} />
          ))}

          {/* Axis values */}
          <text x={padding.left} y={padding.top + 10} fill="var(--text-muted)" fontSize="10px">
            {maxVal.toFixed(3)}
          </text>
          <text x={padding.left} y={padding.top + plotHeight} fill="var(--text-muted)" fontSize="10px">
            {minVal.toFixed(3)}
          </text>

          {/* Labels */}
          <text x={width / 2} y={height - 6} fill="var(--text-muted)" fontSize="10px" textAnchor="middle">
            Epoch / Step →
          </text>
          <text x={-height / 2} y={14} fill="var(--text-muted)" fontSize="10px" textAnchor="middle" transform="rotate(-90)">
            {yLabel}
          </text>
        </svg>
      </div>
    </div>
  );
}

export function TrainingCurvesTab({ payload }: TrainingCurvesTabProps) {
  const ttCurves = payload?.two_tower_curves || [];
  const rankerCurves = payload?.ranker_curves || [];

  if (ttCurves.length === 0 && rankerCurves.length === 0) {
    return (
      <RecsysEmptyState
        command="make recsys-train-tt"
        title="Training History Curves Not Logged"
        message="Epoch training loss logs and validation convergence curves are not recorded in this build. Run training to generate live curve checkpoints:"
      />
    );
  }

  return (
    <div>
      <div style={{ marginBottom: "20px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
          Model Convergence & Loss Progression
        </h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
          In-sample training loss and out-of-sample validation metrics logged across training epochs.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "16px" }}>
        {ttCurves.length > 0 && (
          <>
            <EpochLineChart
              points={ttCurves}
              title="Two-Tower: Training Loss"
              metricKey="train_loss"
              yLabel="BPR Loss"
              color="var(--accent-primary)"
            />
            <EpochLineChart
              points={ttCurves}
              title="Two-Tower: Validation NDCG@10"
              metricKey="val_ndcg_at_10"
              yLabel="Val NDCG@10"
              color="var(--success)"
            />
          </>
        )}

        {rankerCurves.length > 0 && (
          <EpochLineChart
            points={rankerCurves}
            title="Ranker: Validation NDCG"
            metricKey="eval_ndcg"
            yLabel="NDCG"
            color="var(--warning)"
          />
        )}
      </div>
    </div>
  );
}
