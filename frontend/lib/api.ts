/**
 * Centralized API client for the SteamIQ backend.
 *
 * ADR 0001, Decision 7: NEXT_PUBLIC_API_URL is ALWAYS read from env.
 * Never hardcode localhost or any port number in this file or anywhere else.
 *
 * Usage:
 *   import { api } from "@/lib/api";
 *   const result = await api.searchGames("hollow knight");
 */

// ─── Types matching the backend ApiResponse envelope ─────────────────────────

export interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown> | null;
  } | null;
  meta: {
    version: string;
    took_ms: number | null;
    page?: number | null;
    page_size?: number | null;
    total?: number | null;
  } | null;
}

export interface GameSummary {
  app_id: number;
  name: string;
  short_description: string | null;
  header_image: string | null;
  developer: string | null;
  publisher: string | null;
  release_date: string | null;
  is_free: boolean;
  final_price_usd: string | null;
  discount_pct: number;
  positive_reviews: number;
  negative_reviews: number;
  owners_estimate: string | null;
  genres: Array<{ id: string; description: string }> | null;
  is_ingested?: boolean;
}

export interface GameDetail extends GameSummary {
  description: string | null;
  website: string | null;
  coming_soon: boolean;
  categories: Array<{ id: string; description: string }> | null;
  tags: Record<string, number> | null;
  price_usd: string | null;
  platform_windows: boolean;
  platform_mac: boolean;
  platform_linux: boolean;
  review_score: number | null;
  review_score_desc: string | null;
  average_playtime_forever: number;
  median_playtime_forever: number;
  metacritic_score: number | null;
  // Decision & Mart Intelligence (Phase 5)
  primary_genre: string | null;
  revenue_tier: string | null;
  estimated_gross_revenue_usd: number | string | null;
  success_score: number | string | null;
  net_sentiment_pct: number | string | null;
  peak_ccu_24h: number | null;
  executive_brief: {
    market_position?: string;
    growth_vector?: string;
  } | null;
  top_strengths: string[] | null;
  top_complaints: Array<{
    category: string;
    volume_pct: number;
    severity: string;
  }> | null;
  shap_values: Record<string, number> | null;
  model_run_id: string | null;
}

export interface GameSearchResult {
  games: GameSummary[];
  total: number;
  page: number;
  page_size: number;
  query: string;
}

// ─── Market & Pricing Intelligence Types (Phase 5 — Tab 4) ────────────────────

export interface GenrePricingSpectrum {
  genre: string;
  median_price_usd: number;
  q25_price_usd: number;
  q75_price_usd: number;
  min_price_usd: number;
  max_price_usd: number;
  discounted_game_share_pct: number;
}

export interface PriceSnapshotItem {
  recorded_at: string;
  price_usd: number | null;
  final_price_usd: number | null;
  discount_pct: number;
}

export interface MarketIntelligence {
  app_id: number;
  primary_genre: string | null;
  current_price_usd: number | null;
  price_tier: string;
  historical_lowest_price_usd: number | null;
  price_tracking_started_at: string | null;
  genre_pricing_spectrum: GenrePricingSpectrum;
  competitor_price_distribution: Record<string, number>;
  recent_price_snapshots: PriceSnapshotItem[];
}

// ─── Update Impact Tracker Types (Phase 5 — Tab 6) ────────────────────────────

export interface UpdateImpactItem {
  id: number;
  app_id: number;
  patch_name: string;
  patch_version: string | null;
  patch_date: string;
  is_inferred: boolean;
  window_days: number;
  pre_sentiment_positive_pct: number | null;
  post_sentiment_positive_pct: number | null;
  sentiment_delta_pct: number | null;
  observed_sentiment_verdict: string;
  pre_avg_ccu: number | null;
  post_avg_ccu: number | null;
  ccu_change_pct: number | null;
  pre_complaint_distribution: Record<string, number> | null;
  post_complaint_distribution: Record<string, number> | null;
  top_resolved_complaints: Array<{
    category: string;
    pre_volume_pct: number;
    post_volume_pct: number;
    delta_pct: number;
  }> | null;
  top_emerging_complaints: Array<{
    category: string;
    pre_volume_pct: number;
    post_volume_pct: number;
    delta_pct: number;
  }> | null;
  correlation_summary: string;
  materialized_at: string;
}

export interface UpdateImpactFeed {
  app_id: number;
  total_updates: number;
  latest_verdict: string | null;
  average_sentiment_delta: number | null;
  updates: UpdateImpactItem[];
}

// ─── Recommendation Center Types (Phase 5 — Tab 7) ────────────────────────────

export interface RecommendationItem {
  id: number;
  model_run_id: string | null;
  recommendation_type: string;
  domain: string;
  priority_rank: number;
  title: string;
  impact_level: "High" | "Medium" | "Low";
  difficulty_level: "High" | "Medium" | "Low";
  confidence_score: number;
  rationale: string;
  evidence_type: string;
  evidence_payload: Record<string, unknown> | null;
  action_items: string[] | null;
  generated_at: string;
}

export interface RecommendationBundle {
  app_id: number;
  total_recommendations: number;
  generated_at: string;
  model_run_id: string | null;
  priority_action_summary: string | null;
  recommendations: RecommendationItem[];
}

// ─── Game Match Types (Phase 5 — Player Match Tab) ──────────────────────────

export interface GameMatchProfile {
  app_id: number;
  difficulty: number;
  story_weight: number;
  exploration: number;
  combat: number;
  multiplayer: number;
  session_length: number;
  confidence_score: number;
  profile_summary: {
    primary_traits?: string[];
    tag_highlights?: string[];
    intensity_vector?: Record<string, number>;
  } | null;
  materialized_at: string;
}

export interface UserMatchPreferences {
  difficulty?: number;
  story_weight?: number;
  exploration?: number;
  combat?: number;
  multiplayer?: number;
  session_length?: number;
}

export interface DimensionMatchScore {
  dimension: string;
  user_preference: number;
  game_intensity: number;
  delta: number;
  dimension_match_pct: number;
}

export interface GameMatchResult {
  app_id: number;
  game_name: string;
  overall_match_pct: number;
  match_verdict: string;
  dimension_scores: Record<string, DimensionMatchScore>;
  alignment_highlights: string[];
  friction_points: string[];
  confidence_score: number;
  materialized_at: string;
}

// ─── Review Intelligence Types (Phase 2) ──────────────────────────────────────

export interface SentimentOverviewData {
  positive_pct: number;
  mixed_pct: number;
  negative_pct: number;
  positive_count: number;
  negative_count: number;
  total_count: number;
  sentiment_label: string;
}

export interface MonthlySentimentData {
  month: string;
  positive_reviews: number;
  negative_reviews: number;
  net_positive_pct: number;
}

export interface ReviewTopicData {
  topic_id: number;
  label: string;
  review_count: number;
  sentiment_score: number;
  keywords: string[] | null;
}

export interface LovedFeatureData {
  feature_name: string;
  mention_count: number;
  praise_intensity: number;
}

export interface ComplaintCategoryData {
  category: string;
  volume_pct: number;
  severity: "high" | "moderate" | "low";
  representative_snippets: string[];
}

export interface ReviewSummaryData {
  strengths: string[];
  pain_points: string[];
  feature_requests: string[];
}

export interface ReviewIntelligenceBundle {
  app_id: number;
  sentiment: SentimentOverviewData;
  timeline: MonthlySentimentData[];
  topics: ReviewTopicData[];
  loved_features: LovedFeatureData[];
  complaints: ComplaintCategoryData[];
  summary: ReviewSummaryData;
  is_processed: boolean;
}

// ─── Competitor Intelligence Types (Phase 3) ──────────────────────────────────

export interface CompetitorItem {
  rank: number;
  app_id: number;
  name: string;
  similarity_score: number;
  similarity_pct: number;
  shared_tags: string[];
  price_usd: string | null;
  price_delta_usd: number | null;
  market_presence?: number | null;
  ccu_data_stale?: boolean;
  positive_reviews: number;
  negative_reviews: number;
  review_pct: number | null;
  header_image: string | null;
  genres: Array<{ id: string; description: string }> | null;
}

export interface CompetitorSourceGame {
  app_id: number;
  name: string;
  header_image: string | null;
  final_price_usd: string | null;
  positive_reviews: number;
  negative_reviews: number;
  review_pct: number | null;
  genres: Array<{ id: string; description: string }> | null;
}

export interface CompetitorList {
  app_id: number;
  source_game: CompetitorSourceGame;
  competitors: CompetitorItem[];
  total: number;
  model_name: string;
  is_processed: boolean;
}

// ─── API error class ──────────────────────────────────────────────────────────

export class SteamIQApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly details?: Record<string, unknown> | null,
  ) {
    super(message);
    this.name = "SteamIQApiError";
  }
}

// ─── Base fetch helper ────────────────────────────────────────────────────────

const BASE_URL =
  (typeof window === "undefined"
    ? process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://backend:8000"
    : process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001");

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;

  const resp = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
  });

  const envelope: ApiResponse<T> = await resp.json();

  if (!envelope.success || envelope.error) {
    throw new SteamIQApiError(
      envelope.error?.code ?? "unknown_error",
      envelope.error?.message ?? "An unknown error occurred",
      envelope.error?.details
    );
  }

  return envelope.data as T;
}

// ─── API methods ──────────────────────────────────────────────────────────────

export const api = {
  /**
   * Search games by name.
   * GET /api/v1/games/search?q={query}&page={page}&page_size={pageSize}
   */
  async searchGames(
    query: string,
    page = 1,
    pageSize = 20
  ): Promise<GameSearchResult> {
    const params = new URLSearchParams({
      q: query,
      page: String(page),
      page_size: String(pageSize),
    });
    return apiFetch<GameSearchResult>(`/api/v1/games/search?${params}`);
  },

  /**
   * Get full game detail by Steam app_id.
   * GET /api/v1/games/{appId}
   */
  async getGame(appId: number): Promise<GameDetail> {
    return apiFetch<GameDetail>(`/api/v1/games/${appId}`);
  },

  /**
   * Get Review Intelligence bundle by Steam app_id (Phase 2).
   * GET /api/v1/games/{appId}/reviews
   */
  async getReviews(appId: number): Promise<ReviewIntelligenceBundle> {
    return apiFetch<ReviewIntelligenceBundle>(`/api/v1/games/${appId}/reviews`);
  },

  /**
   * Get Competitor Discovery bundle by Steam app_id (Phase 3).
   * GET /api/v1/games/{appId}/competitors?limit={limit}
   */
  async getCompetitors(appId: number, limit = 10): Promise<CompetitorList> {
    return apiFetch<CompetitorList>(`/api/v1/games/${appId}/competitors?limit=${limit}`);
  },

  /**
   * Get Market & Pricing Intelligence by Steam app_id (Phase 5).
   * GET /api/v1/games/{appId}/market
   */
  async getMarket(appId: number): Promise<MarketIntelligence> {
    return apiFetch<MarketIntelligence>(`/api/v1/games/${appId}/market`);
  },

  /**
   * Get Update Impact Tracker timeline by Steam app_id (Phase 5).
   * GET /api/v1/games/{appId}/updates
   */
  async getUpdates(appId: number): Promise<UpdateImpactFeed> {
    return apiFetch<UpdateImpactFeed>(`/api/v1/games/${appId}/updates`);
  },

  /**
   * Get Hybrid Recommendations bundle by Steam app_id (Phase 5).
   * GET /api/v1/games/{appId}/recommendations
   */
  async getRecommendations(appId: number): Promise<RecommendationBundle> {
    return apiFetch<RecommendationBundle>(`/api/v1/games/${appId}/recommendations`);
  },

  /**
   * Get 6-dimension intensity profile for Game Match (Phase 5 — Roadmap v2 §3).
   * GET /api/v1/games/{appId}/match/profile
   */
  async getMatchProfile(appId: number): Promise<GameMatchProfile> {
    return apiFetch<GameMatchProfile>(`/api/v1/games/${appId}/match/profile`);
  },

  /**
   * Evaluate ephemeral user preferences against game match profile (Phase 5 — Roadmap v2 §3).
   * POST /api/v1/games/{appId}/match
   */
  async matchGame(appId: number, prefs: UserMatchPreferences): Promise<GameMatchResult> {
    return apiFetch<GameMatchResult>(`/api/v1/games/${appId}/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(prefs),
    });
  },

  /**
   * Health check.
   * GET /health
   */
  async health(): Promise<{ status: string; db: string; version: string }> {
    return apiFetch<{ status: string; db: string; version: string }>("/health");
  },
} as const;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Format playtime from minutes to a human-readable string. */
export function formatPlaytime(minutes: number): string {
  if (minutes === 0) return "N/A";
  const hours = Math.round(minutes / 60);
  return hours >= 1 ? `${hours.toLocaleString()}h avg` : `${minutes}m avg`;
}

/** Calculate positive review percentage. */
export function reviewScore(
  positive: number,
  negative: number
): { pct: number; label: string } | null {
  const total = positive + negative;
  if (total === 0) return null;
  const pct = Math.round((positive / total) * 100);
  let label = "Mixed";
  if (pct >= 95 && total >= 500) label = "Overwhelmingly Positive";
  else if (pct >= 80) label = "Very Positive";
  else if (pct >= 70) label = "Mostly Positive";
  else if (pct >= 40) label = "Mixed";
  else if (pct >= 20) label = "Mostly Negative";
  else label = "Overwhelmingly Negative";
  return { pct, label };
}

/** Format USD price from string (e.g. "14.99" → "$14.99"). */
export function formatPrice(
  price: string | null,
  isFree: boolean
): string {
  if (isFree) return "Free";
  if (!price || price === "0.00") return "Free";
  return `$${parseFloat(price).toFixed(2)}`;
}
