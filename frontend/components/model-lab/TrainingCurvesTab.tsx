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
  learning_rate?: number;
  tau?: number;
  wall_time?: number;
}

interface RawCurvePayload {
  curves?: EpochMetric[];
  history?: EpochMetric[];
  seeds?: Record<string, EpochMetric[]>;
  [key: string]: unknown;
}

interface TrainingCurvesPayload {
  two_tower_curves?: EpochMetric[] | RawCurvePayload;
  ranker_curves?: EpochMetric[] | RawCurvePayload;
  metadata?: Record<string, unknown>;
}

interface TrainingCurvesTabProps {
  payload: TrainingCurvesPayload | null;
}

function parseCurvesData(raw: unknown): { main: EpochMetric[]; seeds: Record<string, EpochMetric[]> } {
  if (!raw) return { main: [], seeds: {} };
  if (Array.isArray(raw)) {
    return { main: raw as EpochMetric[], seeds: {} };
  }
  if (typeof raw === "object") {
    const obj = raw as RawCurvePayload;
    const seeds = (obj.seeds && typeof obj.seeds === "object") ? obj.seeds : {};
    let main: EpochMetric[] = [];
    if (Array.isArray(obj.curves) && obj.curves.length > 0) {
      main = obj.curves;
    } else if (Array.isArray(obj.history) && obj.history.length > 0) {
      main = obj.history;
    } else {
      const seedKeys = Object.keys(seeds);
      if (seedKeys.length > 0) {
        main = seeds[seedKeys[0]] || [];
      }
    }
    return { main, seeds };
  }
  return { main: [], seeds: {} };
}

/** Inline SVG line chart for loss / metric per epoch with multi-seed overlay */
function EpochLineChart({
  points,
  seeds = {},
  title,
  metricKey,
  yLabel,
  color = "var(--accent-primary)",
}: {
  points: EpochMetric[];
  seeds?: Record<string, EpochMetric[]>;
  title: string;
  metricKey: keyof EpochMetric;
  yLabel: string;
  color?: string;
}) {
  const validPoints = points.filter((p) => typeof p[metricKey] === "number");
  const seedEntries = Object.entries(seeds);

  if (validPoints.length === 0 && seedEntries.length === 0) return null;

  const width = 480;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 35, left: 55 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Gather all values across main points and all seed traces for accurate axis scaling
  const allValues: number[] = validPoints.map((p) => p[metricKey] as number);
  seedEntries.forEach(([, sPoints]) => {
    sPoints.forEach((p) => {
      if (typeof p[metricKey] === "number") {
        allValues.push(p[metricKey] as number);
      }
    });
  });

  if (allValues.length === 0) return null;

  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const range = maxVal - minVal || 1;

  const getX = (i: number, len: number) => padding.left + (i / (Math.max(len - 1, 1))) * plotWidth;
  const getY = (v: number) => padding.top + plotHeight - ((v - minVal) / range) * plotHeight;

  const mainPathD = validPoints.reduce(
    (acc, p, i) => `${acc} ${i === 0 ? "M" : "L"} ${getX(i, validPoints.length)} ${getY(p[metricKey] as number)}`,
    ""
  );

  return (
    <div className="card" style={{ padding: "16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "8px" }}>
        <h4 style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-primary)" }}>
          {title}
        </h4>
        {seedEntries.length > 1 && (
          <span style={{ fontSize: "11px", color: "var(--text-secondary)" }}>
            {seedEntries.length} seeds overlaid
          </span>
        )}
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg width={width} height={height} style={{ backgroundColor: "var(--bg-surface-low)", borderRadius: "var(--radius-sm)" }} aria-label={title}>
          {/* Axis lines */}
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotHeight} stroke="var(--border-subtle)" />
          <line x1={padding.left} y1={padding.top + plotHeight} x2={padding.left + plotWidth} y2={padding.top + plotHeight} stroke="var(--border-subtle)" />

          {/* Faint multi-seed traces */}
          {seedEntries.map(([sKey, sPoints]) => {
            const validSeedPoints = sPoints.filter((p) => typeof p[metricKey] === "number");
            if (validSeedPoints.length === 0) return null;
            const seedPathD = validSeedPoints.reduce(
              (acc, p, i) => `${acc} ${i === 0 ? "M" : "L"} ${getX(i, validSeedPoints.length)} ${getY(p[metricKey] as number)}`,
              ""
            );
            return (
              <path
                key={sKey}
                d={seedPathD}
                fill="none"
                stroke={color}
                strokeWidth="1.2"
                strokeOpacity="0.3"
                strokeDasharray="2 2"
              />
            );
          })}

          {/* Main / champion curve path */}
          {validPoints.length > 0 && (
            <path d={mainPathD} fill="none" stroke={color} strokeWidth="2.5" />
          )}

          {/* Points for main curve */}
          {validPoints.map((p, i) => (
            <circle key={i} cx={getX(i, validPoints.length)} cy={getY(p[metricKey] as number)} r="3" fill={color} />
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
  const { main: ttCurves, seeds: ttSeeds } = parseCurvesData(payload?.two_tower_curves);
  const { main: rankerCurves, seeds: rankerSeeds } = parseCurvesData(payload?.ranker_curves);

  if (ttCurves.length === 0 && rankerCurves.length === 0 && Object.keys(ttSeeds).length === 0 && Object.keys(rankerSeeds).length === 0) {
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
          In-sample training loss and out-of-sample validation metrics logged across training epochs. Multi-seed runs display individual seed paths alongside the champion run.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "16px" }}>
        {(ttCurves.length > 0 || Object.keys(ttSeeds).length > 0) && (
          <>
            <EpochLineChart
              points={ttCurves}
              seeds={ttSeeds}
              title="Two-Tower: Training Loss"
              metricKey="train_loss"
              yLabel="Sampled Softmax Loss"
              color="var(--accent-primary)"
            />
            <EpochLineChart
              points={ttCurves}
              seeds={ttSeeds}
              title="Two-Tower: Validation NDCG@10"
              metricKey="val_ndcg_at_10"
              yLabel="Val NDCG@10"
              color="var(--success)"
            />
          </>
        )}

        {(rankerCurves.length > 0 || Object.keys(rankerSeeds).length > 0) && (
          <EpochLineChart
            points={rankerCurves}
            seeds={rankerSeeds}
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
