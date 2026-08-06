import type { Metadata } from "next";
import Link from "next/link";
import { api, formatPrice, formatPlaytime, reviewScore, type GameDetail } from "@/lib/api";
import { notFound } from "next/navigation";

// ─── Metadata ─────────────────────────────────────────────────────────────────

export async function generateMetadata({
  params,
}: {
  params: { appId: string };
}): Promise<Metadata> {
  try {
    const game = await api.getGame(Number(params.appId));
    return {
      title: game.name,
      description: game.short_description ?? `Game intelligence for ${game.name} on Steam.`,
    };
  } catch {
    return { title: "Game Not Found" };
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ReviewBadge({ positive, negative }: { positive: number; negative: number }) {
  const score = reviewScore(positive, negative);
  if (!score) return null;

  let cls = "review-badge--mixed";
  if (score.pct >= 80) cls = "review-badge--positive";
  else if (score.pct < 50) cls = "review-badge--negative";

  return (
    <span className={`review-badge ${cls}`} style={{ fontSize: "0.85rem", padding: "5px 14px" }}>
      👍 {score.pct}% &nbsp;·&nbsp; {score.label}
    </span>
  );
}

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__value">{value}</div>
      {sub && <div className="stat-card__sub">{sub}</div>}
    </div>
  );
}

// Strip HTML tags from Steam descriptions
function stripHtml(html: string): string {
  return html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default async function GameDetailPage({
  params,
}: {
  params: { appId: string };
}) {
  const appId = Number(params.appId);

  if (isNaN(appId)) notFound();

  let game: GameDetail;
  try {
    game = await api.getGame(appId);
  } catch {
    notFound();
  }

  const score     = reviewScore(game.positive_reviews, game.negative_reviews);
  const priceStr  = formatPrice(game.final_price_usd, game.is_free);
  const totalReviews = game.positive_reviews + game.negative_reviews;
  const description = game.description ? stripHtml(game.description) : null;

  const topTags = game.tags
    ? Object.entries(game.tags)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 15)
        .map(([tag]) => tag)
    : [];

  const platforms = [
    game.platform_windows && "Windows",
    game.platform_mac && "macOS",
    game.platform_linux && "Linux",
  ].filter(Boolean) as string[];

  return (
    <>
      {/* Hero */}
      <section
        className="detail-hero"
        style={{ position: "relative", paddingTop: 40 }}
      >
        {game.header_image && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={game.header_image}
            alt=""
            className="detail-hero__bg"
            aria-hidden="true"
          />
        )}
        <div className="detail-hero__overlay" />

        <div className="container" style={{ width: "100%", position: "relative", zIndex: 1 }}>
          <a href="/" className="back-link">← Back to search</a>

          <div className="detail-hero__grid">
            {game.header_image && (
              <div className="detail-hero__cover">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={game.header_image} alt={`${game.name} cover`} />
              </div>
            )}

            <div>
              <h1 className="detail-hero__title">{game.name}</h1>

              <p className="detail-hero__byline">
                {[game.developer, game.publisher]
                  .filter(Boolean)
                  .join(" · ")}
                {game.release_date && ` · ${game.release_date}`}
              </p>

              <div className="detail-hero__tags">
                {game.genres?.slice(0, 5).map((g) => (
                  <span key={g.id} className="genre-chip">{g.description}</span>
                ))}
              </div>

              <div className="detail-hero__price-row">
                <span className={`detail-price ${game.is_free ? "free" : ""}`}>
                  {priceStr}
                </span>
                {game.discount_pct > 0 && (
                  <span
                    style={{
                      background: "var(--color-accent-yellow)",
                      color: "#000",
                      fontWeight: 700,
                      fontSize: "0.8rem",
                      padding: "2px 8px",
                      borderRadius: 4,
                    }}
                  >
                    -{game.discount_pct}%
                  </span>
                )}
                <ReviewBadge
                  positive={game.positive_reviews}
                  negative={game.negative_reviews}
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats bar */}
      <div className="container">
        <div className="stats-bar">
          <StatCard
            label="Review Score"
            value={score ? `${score.pct}%` : "N/A"}
            sub={score?.label}
          />
          <StatCard
            label="Total Reviews"
            value={totalReviews > 0 ? totalReviews.toLocaleString() : "N/A"}
            sub={`${game.positive_reviews.toLocaleString()} positive`}
          />
          <StatCard
            label="Owners"
            value={game.owners_estimate ?? "N/A"}
          />
          <StatCard
            label="Avg Playtime"
            value={formatPlaytime(game.average_playtime_forever)}
          />
          {game.metacritic_score && (
            <StatCard
              label="Metacritic"
              value={game.metacritic_score}
              sub="Critic score"
            />
          )}
          <StatCard
            label="App ID"
            value={game.app_id}
            sub={
              <a
                href={`https://store.steampowered.com/app/${game.app_id}`}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "var(--color-accent-primary)", textDecoration: "none" }}
              >
                View on Steam ↗
              </a>
            }
          />
        </div>

        {/* Content grid */}
        <div className="content-grid">
          {/* Main: description */}
          <div>
            {description && (
              <div className="content-panel" style={{ marginBottom: 20 }}>
                <h2 className="content-panel__title">About</h2>
                <p className="description-text">{description}</p>
              </div>
            )}

            {/* NLP Intelligence — Phase 2+ placeholder */}
            <div
              className="content-panel"
              style={{
                background: "repeating-linear-gradient(-45deg, transparent, transparent 8px, rgba(99,148,255,0.02) 8px, rgba(99,148,255,0.02) 16px)",
                border: "1px dashed var(--color-border-strong)",
              }}
            >
              <h2 className="content-panel__title">🔬 Review Intelligence</h2>
              <div style={{ textAlign: "center", padding: "32px 0", color: "var(--color-text-muted)" }}>
                <div style={{ fontSize: "2rem", marginBottom: 12 }}>🚧</div>
                <p style={{ fontWeight: 600, color: "var(--color-text-secondary)", marginBottom: 8 }}>
                  Phase 2 — NLP Core
                </p>
                <p style={{ fontSize: "0.85rem", maxWidth: 400, margin: "0 auto", lineHeight: 1.7 }}>
                  Sentiment analysis, topic discovery, complaint detection, and review
                  summarization will appear here once the NLP pipeline runs for this game.
                </p>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Tags */}
            {topTags.length > 0 && (
              <div className="content-panel">
                <h2 className="content-panel__title">Tags</h2>
                <div className="tag-cloud">
                  {topTags.map((tag) => (
                    <span key={tag} className="tag-chip">{tag}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Platforms */}
            {platforms.length > 0 && (
              <div className="content-panel">
                <h2 className="content-panel__title">Platforms</h2>
                <div className="platform-badges">
                  {platforms.map((p) => (
                    <span key={p} className="platform-badge">
                      {p === "Windows" ? "🪟" : p === "macOS" ? "🍎" : "🐧"} {p}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Competitor Similarity — Phase 3+ placeholder */}
            <div
              className="content-panel"
              style={{ border: "1px dashed var(--color-border-strong)" }}
            >
              <h2 className="content-panel__title">🔗 Similar Games</h2>
              <div style={{ textAlign: "center", padding: "20px 0", color: "var(--color-text-muted)", fontSize: "0.8rem" }}>
                Phase 3 — Similarity + Competitors
              </div>
            </div>

            {/* Success prediction — Phase 4+ placeholder */}
            <div
              className="content-panel"
              style={{ border: "1px dashed var(--color-border-strong)" }}
            >
              <h2 className="content-panel__title">📈 Success Score</h2>
              <div style={{ textAlign: "center", padding: "20px 0", color: "var(--color-text-muted)", fontSize: "0.8rem" }}>
                Phase 4 — Predictive ML
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
