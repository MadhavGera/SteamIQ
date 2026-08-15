import type { Metadata } from "next";
import { notFound } from "next/navigation";
import {
  api,
  type CompetitorList,
  type GameDetail,
  type MarketIntelligence,
  type RecommendationBundle,
  type ReviewIntelligenceBundle,
  type UpdateImpactFeed,
} from "@/lib/api";
import { GameDetailView } from "@/components/game/GameDetailView";

// ─── Metadata ─────────────────────────────────────────────────────────────────

export async function generateMetadata({
  params,
}: {
  params: { appId: string };
}): Promise<Metadata> {
  try {
    const game = await api.getGame(Number(params.appId));
    return {
      title: `${game.name} — Game Intelligence | SteamIQ`,
      description:
        game.short_description ??
        `Game intelligence, review breakdown, and predictive analytics for ${game.name} on Steam.`,
    };
  } catch {
    return { title: `Game ${params.appId} Intelligence — SteamIQ` };
  }
}

// ─── Page Component (Server Component with Client Mode-Toggle IA) ────────────

export default async function GameDetailPage({
  params,
  searchParams,
}: {
  params: { appId: string };
  searchParams?: { tab?: string };
}) {
  const appId = Number(params.appId);
  if (isNaN(appId)) notFound();

  const initialTab = searchParams?.tab ?? "overview";

  let game: GameDetail | null = null;
  let reviewsBundle: ReviewIntelligenceBundle | null = null;
  let competitorData: CompetitorList | null = null;
  let marketData: MarketIntelligence | null = null;
  let updatesData: UpdateImpactFeed | null = null;
  let recommendationsData: RecommendationBundle | null = null;

  const [
    gameRes,
    reviewsRes,
    competitorsRes,
    marketRes,
    updatesRes,
    recommendationsRes,
  ] = await Promise.allSettled([
    api.getGame(appId),
    api.getReviews(appId),
    api.getCompetitors(appId),
    api.getMarket(appId),
    api.getUpdates(appId),
    api.getRecommendations(appId),
  ]);

  if (gameRes.status === "fulfilled") game = gameRes.value;
  if (reviewsRes.status === "fulfilled") reviewsBundle = reviewsRes.value;
  if (competitorsRes.status === "fulfilled") competitorData = competitorsRes.value;
  if (marketRes.status === "fulfilled") marketData = marketRes.value;
  if (updatesRes.status === "fulfilled") updatesData = updatesRes.value;
  if (recommendationsRes.status === "fulfilled") recommendationsData = recommendationsRes.value;

  return (
    <GameDetailView
      appId={appId}
      initialTab={initialTab}
      game={game}
      reviewsBundle={reviewsBundle}
      competitorData={competitorData}
      marketData={marketData}
      updatesData={updatesData}
      recommendationsData={recommendationsData}
    />
  );
}
