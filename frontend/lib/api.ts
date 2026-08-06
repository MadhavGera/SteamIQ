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
}

export interface GameSearchResult {
  games: GameSummary[];
  total: number;
  page: number;
  page_size: number;
  query: string;
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

const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

if (!BASE_URL && typeof window !== "undefined") {
  console.error(
    "[SteamIQ] NEXT_PUBLIC_API_URL is not set. " +
      "Copy .env.example to .env.local and set NEXT_PUBLIC_API_URL."
  );
}

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
