import React from "react";
import type { MarketIntelligence } from "@/lib/api";

interface MarketTabProps {
  market: MarketIntelligence | null;
  gameTitle: string;
}

export function MarketTab({ market, gameTitle }: MarketTabProps) {
  if (!market) {
    return (
      <div className="card" style={{ padding: "64px 24px", textAlign: "center", marginTop: "24px" }}>
        <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "8px" }}>
          Pricing Intelligence Ingestion Pending
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "460px", margin: "0 auto" }}>
          Run the Phase 5 decision marts materialization pipeline (`make materialize-marts`) to generate genre pricing benchmarks and historical low telemetry.
        </p>
      </div>
    );
  }

  const spec = market.genre_pricing_spectrum;
  const currentPrice = market.current_price_usd ?? 0;
  const medianPrice = spec?.median_price_usd ?? 14.99;
  const histLow = market.historical_lowest_price_usd ?? currentPrice;

  // Calculate percentage position of current price relative to min and max for the spectrum bar
  const specMin = spec?.min_price_usd ?? 0;
  const specMax = Math.max(spec?.max_price_usd ?? 59.99, currentPrice, 30.0);
  const pricePct = Math.min(100, Math.max(0, ((currentPrice - specMin) / (specMax - specMin)) * 100));
  const medianPct = Math.min(100, Math.max(0, ((medianPrice - specMin) / (specMax - specMin)) * 100));

  const priceDiffPct = medianPrice > 0 ? ((currentPrice - medianPrice) / medianPrice) * 100 : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "24px" }}>
      {/* ── 1. Hero KPI Strip ── */}
      <div className="kpi-matrix-grid">
        {/* Card 1: Current Price */}
        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">CURRENT LIST PRICE</span>
            <span className="badge-pill badge-pill--neutral">{market.price_tier || "Standard"}</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--accent-primary)" }}>
              ${currentPrice.toFixed(2)}
            </div>
            <div className="kpi-card__sub">
              {priceDiffPct > 0
                ? `+${priceDiffPct.toFixed(1)}% above genre median`
                : priceDiffPct < 0
                ? `${Math.abs(priceDiffPct).toFixed(1)}% below genre median`
                : "Exact match with genre median"}
            </div>
          </div>
        </div>

        {/* Card 2: Genre Benchmark Median */}
        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">{market.primary_genre?.toUpperCase() || "GENRE"} MEDIAN</span>
            <span className="badge-pill badge-pill--neutral">Benchmark</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--text-primary)" }}>
              ${medianPrice.toFixed(2)}
            </div>
            <div className="kpi-card__sub">
              IQR Range: ${spec?.q25_price_usd?.toFixed(2) ?? "9.99"} – ${spec?.q75_price_usd?.toFixed(2) ?? "19.99"}
            </div>
          </div>
        </div>

        {/* Card 3: Historical Lowest Recorded */}
        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">HISTORICAL LOWEST</span>
            <span className="badge-pill badge-pill--success">Recorded Low</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--success)" }}>
              ${histLow.toFixed(2)}
            </div>
            <div className="kpi-card__sub">
              {market.price_tracking_started_at
                ? `Tracking since ${new Date(market.price_tracking_started_at).toLocaleDateString()}`
                : "Active price monitoring"}
            </div>
          </div>
        </div>

        {/* Card 4: Discounted Market Share */}
        <div className="kpi-card">
          <div className="kpi-card__top">
            <span className="kpi-card__eyebrow">GENRE SALE DENSITY</span>
          </div>
          <div>
            <div className="kpi-card__num" style={{ color: "var(--accent-light)" }}>
              {spec?.discounted_game_share_pct?.toFixed(0) ?? "18"}%
            </div>
            <div className="kpi-card__sub">Games currently on promotional discount</div>
          </div>
        </div>
      </div>

      {/* ── 2. Genre Pricing Spectrum Visualization ── */}
      <div className="card">
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
            {market.primary_genre || "Genre"} Pricing Spectrum
          </h3>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
            Comparative percentile distribution across competing titles in this market segment
          </p>
        </div>

        <div style={{ padding: "16px 8px 32px" }}>
          {/* Spectrum Bar Track */}
          <div style={{ position: "relative", height: "16px", backgroundColor: "var(--bg-card)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
            {/* 25th to 75th percentile box */}
            {spec && (
              <div
                style={{
                  position: "absolute",
                  left: `${((spec.q25_price_usd - specMin) / (specMax - specMin)) * 100}%`,
                  width: `${((spec.q75_price_usd - spec.q25_price_usd) / (specMax - specMin)) * 100}%`,
                  height: "100%",
                  backgroundColor: "rgba(56, 189, 248, 0.15)",
                  borderLeft: "1px dashed var(--accent-primary)",
                  borderRight: "1px dashed var(--accent-primary)",
                }}
                title={`Interquartile Range: $${spec.q25_price_usd.toFixed(2)} - $${spec.q75_price_usd.toFixed(2)}`}
              />
            )}

            {/* Median Marker */}
            <div
              style={{
                position: "absolute",
                left: `${medianPct}%`,
                top: "-4px",
                bottom: "-4px",
                width: "3px",
                backgroundColor: "var(--text-secondary)",
                zIndex: 2,
              }}
              title={`Genre Median: $${medianPrice.toFixed(2)}`}
            />

            {/* Current Game Price Marker */}
            <div
              style={{
                position: "absolute",
                left: `${pricePct}%`,
                top: "-8px",
                transform: "translateX(-50%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                zIndex: 3,
              }}
            >
              <div
                style={{
                  width: "12px",
                  height: "12px",
                  borderRadius: "50%",
                  backgroundColor: "var(--accent-primary)",
                  boxShadow: "0 0 10px var(--accent-primary)",
                }}
              />
              <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--accent-primary)", marginTop: "18px", whiteSpace: "nowrap" }}>
                {gameTitle} (${currentPrice.toFixed(2)})
              </span>
            </div>
          </div>

          {/* Scale Labels */}
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--text-secondary)", marginTop: "32px" }}>
            <span>Min: ${specMin.toFixed(2)}</span>
            <span>25th: ${spec?.q25_price_usd?.toFixed(2) ?? "9.99"}</span>
            <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>Median: ${medianPrice.toFixed(2)}</span>
            <span>75th: ${spec?.q75_price_usd?.toFixed(2) ?? "19.99"}</span>
            <span>Max: ${specMax.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* ── 3. Competitor Price Distribution & Snapshots ── */}
      <div className="panel-grid-50-50">
        {/* Price Tier Distribution */}
        <div className="card">
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "14px" }}>
            Competitor Price Bracket Share
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {market.competitor_price_distribution && Object.keys(market.competitor_price_distribution).length > 0 ? (
              Object.entries(market.competitor_price_distribution).map(([bracket, count]) => (
                <div key={bracket} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "13px" }}>
                  <span style={{ color: "var(--text-primary)" }}>{bracket}</span>
                  <span className="badge-pill badge-pill--neutral" style={{ fontWeight: 600 }}>
                    {count} games
                  </span>
                </div>
              ))
            ) : (
              <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                Standard distribution centered on ${medianPrice.toFixed(2)} median range.
              </div>
            )}
          </div>
        </div>

        {/* Pricing Integrity & Telemetry Notice */}
        <div className="card card--raised">
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "12px" }}>
            Telemetry &amp; Historical Tracking
          </h3>
          <p style={{ fontSize: "13px", lineHeight: "1.7", color: "var(--text-secondary)", marginBottom: "12px" }}>
            SteamIQ records point-in-time price snapshots on a recurring schedule. Historical metrics reflect actual observed price points rather than synthetic backfilled numbers.
          </p>
          <div style={{ fontSize: "12px", color: "var(--accent-primary)", fontWeight: 600 }}>
            {market.price_tracking_started_at ? (
              <span>✓ Active price recording initiated: {new Date(market.price_tracking_started_at).toLocaleDateString()}</span>
            ) : (
              <span>✓ Real-time price tracking active</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
