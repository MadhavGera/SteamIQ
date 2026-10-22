const http = require("http");
const fs = require("fs");
const path = require("path");

const RESULTS_DIR = path.resolve(__dirname, "../../docs/results");

function loadResultJson(filename, fallback = {}) {
  const filePath = path.join(RESULTS_DIR, filename);
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, "utf-8"));
    }
  } catch (err) {
    console.error(`Failed to read ${filename}:`, err);
  }
  return fallback;
}

const leaderboardJson = loadResultJson("leaderboard.json");
const assistantEvalJson = loadResultJson("assistant_eval.json");
const retrievalLatencyJson = loadResultJson("retrieval_latency.json");
const driftDetectorPowerJson = loadResultJson("drift_detector_power.json");
const monitorModelsJson = loadResultJson("monitor_models.json");
const retrainGateJson = loadResultJson("retrain_gate.json");

const hadesDetail = {
  app_id: 1145360,
  name: "Hades",
  short_description: "Defy the god of the dead as you hack and slash out of the Underworld.",
  description: "Defy the god of the dead as you hack and slash out of the Underworld in this rogue-like dungeon crawler.",
  header_image: "https://cdn.akamai.steamstatic.com/steam/apps/1145360/header.jpg",
  developer: "Supergiant Games",
  publisher: "Supergiant Games",
  release_date: "Sep 17, 2020",
  is_free: false,
  final_price_usd: "24.99",
  price_usd: "24.99",
  discount_pct: 0,
  positive_reviews: 210000,
  negative_reviews: 4000,
  owners_estimate: "5,000,000 - 10,000,000",
  genres: [{ id: "1", description: "Action" }, { id: "23", description: "Indie" }],
  categories: [{ id: "2", description: "Single-player" }],
  tags: { Action: 1500, "Rogue-like": 1400, Indie: 1200 },
  website: "https://www.supergiantgames.com/games/hades",
  coming_soon: false,
  platform_windows: true,
  platform_mac: true,
  platform_linux: false,
  review_score: 9,
  review_score_desc: "Overwhelmingly Positive",
  average_playtime_forever: 1200,
  median_playtime_forever: 600,
  metacritic_score: 93,
  primary_genre: "Action",
  revenue_tier: "Tier 1 ($10M+)",
  estimated_gross_revenue_usd: 50000000,
  success_score: 96,
  net_sentiment_pct: 98,
  peak_ccu_24h: 8500,
  executive_brief: {
    market_position: "Genre-defining action roguelike with exceptional retention.",
    growth_vector: "Franchise sequel and cross-media expansion."
  },
  top_strengths: ["Combat fluidity", "Narrative integration"],
  top_complaints: [{ category: "Repetition", volume_pct: 2.1, severity: "low" }],
  shap_values: { playtime_forever: 0.42, positive_reviews: 0.38 },
  model_run_id: "hades-prod-run-001"
};

const hadesReviews = {
  app_id: 1145360,
  sentiment: {
    positive_pct: 98.1,
    mixed_pct: 0.5,
    negative_pct: 1.4,
    positive_count: 210000,
    negative_count: 4000,
    total_count: 214000,
    sentiment_label: "Overwhelmingly Positive"
  },
  timeline: [
    { month: "2026-09", positive_reviews: 12000, negative_reviews: 200, net_positive_pct: 98.3 }
  ],
  topics: [
    { topic_id: 1, label: "Combat & Gameplay", review_count: 45000, sentiment_score: 0.96, keywords: ["fluid", "responsive", "boons"] }
  ],
  loved_features: [
    { feature_name: "Combat fluidity", mention_count: 24000, praise_intensity: 9.4 }
  ],
  complaints: [
    { category: "Repetition", volume_pct: 2.1, severity: "low", representative_snippets: ["Grinding for darkness can feel repetitive after 100 runs."] }
  ],
  summary: {
    strengths: ["Combat fluidity", "Narrative integration"],
    pain_points: ["Endgame repetition"],
    feature_requests: ["More weapon aspects"]
  },
  is_processed: true
};

const hadesCompetitors = {
  app_id: 1145360,
  source_game: {
    app_id: 1145360,
    name: "Hades",
    header_image: hadesDetail.header_image,
    final_price_usd: "24.99",
    positive_reviews: 210000,
    negative_reviews: 4000,
    review_pct: 98.1,
    genres: hadesDetail.genres
  },
  competitors: [
    {
      rank: 1,
      app_id: 814380,
      name: "Sekiro: Shadows Die Twice",
      similarity_score: 0.85,
      similarity_pct: 85.0,
      shared_tags: ["Action", "Difficult"],
      price_usd: "59.99",
      price_delta_usd: 35.0,
      market_presence: 0.9,
      ccu_data_stale: false,
      positive_reviews: 120000,
      negative_reviews: 6000,
      review_pct: 95.0,
      header_image: hadesDetail.header_image,
      genres: [{ id: "1", description: "Action" }]
    }
  ],
  total: 1,
  model_name: "recsys_two_tower_clean",
  is_processed: true
};

const hadesMarket = {
  app_id: 1145360,
  name: "Hades",
  primary_genre: "Action",
  current_price_usd: 24.99,
  price_tier: "Mid-tier ($20-$40)",
  historical_lowest_price_usd: 12.49,
  price_tracking_started_at: "2020-09-17",
  genre_pricing_spectrum: {
    genre: "Action",
    median_price_usd: 19.99,
    q25_price_usd: 9.99,
    q75_price_usd: 29.99,
    min_price_usd: 0.0,
    max_price_usd: 69.99,
    discounted_game_share_pct: 35.0
  },
  pricing: {
    primary_genre: "Action",
    game_price_usd: 24.99,
    genre_min_price_usd: 0.0,
    genre_p25_price_usd: 9.99,
    genre_median_price_usd: 19.99,
    genre_p75_price_usd: 29.99,
    genre_max_price_usd: 69.99,
    price_vs_median_pct: 25.0,
    position_bracket: "Above Median",
    historical_lowest_price_usd: 12.49,
    historical_lowest_discount_pct: 50.0,
    price_tracking_started_at: "2020-09-17",
    is_free: false
  },
  competitor_price_distribution: { "0-10": 20, "10-20": 40, "20-30": 30, "30+": 10 },
  recent_price_snapshots: [
    { recorded_at: "2026-10-01", price_usd: 24.99, final_price_usd: 24.99, discount_pct: 0 }
  ],
  price_history: [
    { recorded_at: "2026-10-01", price_usd: 24.99, final_price_usd: 24.99, discount_pct: 0 }
  ]
};

const hadesActivity = {
  app_id: 1145360,
  game_name: "Hades",
  current_ccu: 4200,
  peak_ccu_24h: 8500,
  all_time_peak_ccu: 37749,
  owners_estimate: "5,000,000 - 10,000,000",
  average_playtime_forever_hours: 20.0,
  median_playtime_forever_hours: 10.0,
  average_playtime_at_review_hours: 15.0,
  post_review_engagement_delta_hours: 5.0,
  sample_size_reviews: 214000,
  playtime_percentiles: { p25_hours: 5, median_hours: 10, p75_hours: 30, p90_hours: 60 },
  playtime_cohorts: [
    { cohort: "casual", label: "Casual (<10h)", hours_range: "0-10", player_count: 50000, share_pct: 25.0 },
    { cohort: "core", label: "Core (10-50h)", hours_range: "10-50", player_count: 100000, share_pct: 50.0 },
    { cohort: "veteran", label: "Veteran (>50h)", hours_range: "50+", player_count: 50000, share_pct: 25.0 }
  ],
  ccu_history: [
    { snapshot_at: "2026-10-22T00:00:00Z", player_count: 4200, source: "steam_charts" }
  ],
  recent_event_impacts: [
    { patch_name: "v1.0 Launch", patch_date: "2020-09-17", pre_avg_ccu: 12000, post_avg_ccu: 37749, ccu_change_pct: 214.5, verdict: "Positive" }
  ],
  telemetry_mode: "live",
  updated_at: "2026-10-22T00:00:00Z"
};

const hadesUpdates = {
  app_id: 1145360,
  total_updates: 1,
  latest_verdict: "Positive",
  average_sentiment_delta: 1.4,
  updates: [
    {
      id: 1,
      app_id: 1145360,
      patch_name: "v1.0 Launch",
      patch_version: "1.0",
      patch_date: "2020-09-17",
      is_inferred: false,
      window_days: 30,
      pre_sentiment_positive_pct: 97.0,
      post_sentiment_positive_pct: 98.4,
      sentiment_delta_pct: 1.4,
      observed_sentiment_verdict: "Positive",
      pre_avg_ccu: 12000,
      post_avg_ccu: 37749,
      ccu_change_pct: 214.5,
      pre_complaint_distribution: null,
      post_complaint_distribution: null,
      top_resolved_complaints: null,
      top_emerging_complaints: null,
      correlation_summary: "Major player influx and high praise.",
      materialized_at: "2026-10-20T00:00:00Z"
    }
  ]
};

const hadesRecommendations = {
  app_id: 1145360,
  total_recommendations: 1,
  generated_at: "2026-10-20T00:00:00Z",
  model_run_id: "test-run",
  priority_action_summary: "Optimize endgame grind",
  recommendations: [
    {
      id: 1,
      model_run_id: "test-run",
      recommendation_type: "content",
      domain: "gameplay",
      priority_rank: 1,
      title: "Add endgame weapon variants",
      impact_level: "High",
      difficulty_level: "Medium",
      confidence_score: 0.91,
      rationale: "Players cite repetitive grinding.",
      evidence_type: "nlp_aspect",
      evidence_payload: {},
      action_items: ["Prototype two new aspects"],
      generated_at: "2026-10-20T00:00:00Z"
    }
  ]
};

const hadesMatchProfile = {
  app_id: 1145360,
  difficulty: 7.5,
  story_weight: 8.0,
  exploration: 6.5,
  combat: 9.0,
  multiplayer: 0.0,
  session_length: 6.0,
  confidence_score: 0.94,
  profile_summary: {
    primary_traits: ["High replayability", "Fast-paced combat", "Rich narrative"],
    tag_highlights: ["Rogue-like", "Action", "Indie"]
  },
  materialized_at: "2026-10-20T00:00:00Z"
};

const hadesMatchResult = {
  app_id: 1145360,
  game_name: "Hades",
  overall_match_pct: 92.5,
  match_verdict: "Exceptional Match",
  dimension_scores: {
    combat: { dimension: "combat", user_preference: 8.5, game_intensity: 9.0, delta: 0.5, dimension_match_pct: 95.0 },
    difficulty: { dimension: "difficulty", user_preference: 7.0, game_intensity: 7.5, delta: 0.5, dimension_match_pct: 93.0 },
    story_weight: { dimension: "story_weight", user_preference: 8.0, game_intensity: 8.0, delta: 0.0, dimension_match_pct: 100.0 },
    exploration: { dimension: "exploration", user_preference: 6.5, game_intensity: 6.5, delta: 0.0, dimension_match_pct: 100.0 },
    multiplayer: { dimension: "multiplayer", user_preference: 0.0, game_intensity: 0.0, delta: 0.0, dimension_match_pct: 100.0 },
    session_length: { dimension: "session_length", user_preference: 6.0, game_intensity: 6.0, delta: 0.0, dimension_match_pct: 100.0 }
  },
  alignment_highlights: ["Combat intensity aligns with your preference", "Deep narrative integration"],
  friction_points: [],
  confidence_score: 0.94,
  materialized_at: "2026-10-20T00:00:00Z"
};

const marketOpportunitiesMock = [
  {
    id: 1,
    genre_or_tag: "Action Roguelike",
    entity_type: "genre",
    opportunity_score: 85,
    demand_score: 80,
    saturation_score: 30,
    sentiment_gap_score: 25,
    monetization_score: 75,
    game_count: 140,
    median_price_usd: 19.99,
    avg_review_score: 88,
    top_complaint_themes: ["Repetitive runs", "RNG balance"],
    top_loved_themes: ["Combat fluidity", "Replayability"],
    recommended_features: { Progression: "Meta-progression upgrades" },
    materialized_at: "2026-10-20T00:00:00Z"
  }
];

const marketTrendsMock = [
  {
    id: 1,
    app_id: null,
    trend_type: "genre",
    category: "Action Roguelike",
    recorded_date: "2026-10-01",
    metric_name: "player_hours",
    metric_value: 150000,
    change_pct_7d: 5.2,
    change_pct_30d: 12.8,
    metadata_payload: {},
    materialized_at: "2026-10-20T00:00:00Z"
  }
];

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "127.0.0.1:8000"}`);
  const pathname = url.pathname;

  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const ok = (data) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ success: true, data, error: null, meta: { version: "1.0", took_ms: 1.0 } }));
  };

  // Health
  if (pathname === "/health" || pathname === "/api/v1/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", version: "1.0.0" }));
    return;
  }

  // Games endpoints
  if (pathname === "/api/v1/games/1145360" || pathname === "/api/v1/games/367520" || pathname === "/api/v1/games/1245620") {
    ok(hadesDetail);
    return;
  }
  if (pathname === "/api/v1/games/1145360/reviews") {
    ok(hadesReviews);
    return;
  }
  if (pathname === "/api/v1/games/1145360/competitors") {
    ok(hadesCompetitors);
    return;
  }
  if (pathname === "/api/v1/games/1145360/market") {
    ok(hadesMarket);
    return;
  }
  if (pathname === "/api/v1/games/1145360/activity") {
    ok(hadesActivity);
    return;
  }
  if (pathname === "/api/v1/games/1145360/updates") {
    ok(hadesUpdates);
    return;
  }
  if (pathname === "/api/v1/games/1145360/recommendations") {
    ok(hadesRecommendations);
    return;
  }
  if (pathname === "/api/v1/games/1145360/match/profile") {
    ok(hadesMatchProfile);
    return;
  }
  if (pathname === "/api/v1/games/1145360/match") {
    ok(hadesMatchResult);
    return;
  }
  if (pathname === "/api/v1/games" || pathname === "/api/v1/games/search") {
    ok({ games: [hadesDetail], total: 1, page: 1, page_size: 20, query: "" });
    return;
  }

  // Recsys endpoints
  if (pathname === "/api/v1/recsys/health") {
    ok({
      status: "ready",
      is_ready: true,
      champion_model: "recsys_two_tower_clean",
      universe_size: 190,
      loaded_models: ["recsys_two_tower_clean", "recsys_two_tower", "recsys_itemknn", "recsys_als"]
    });
    return;
  }
  if (pathname === "/api/v1/recsys/models") {
    ok([
      { model_name: "recsys_two_tower_clean", display_name: "Two-Tower Recommender (Clean)", family: "two_tower", stage: "champion", deployable: true, is_ready: true, val_ndcg_at_10: 0.1245, latency_p95_ms: 12.18, description: "Learned dense embeddings." },
      { model_name: "recsys_itemknn", display_name: "Item-kNN", family: "itemknn", stage: "candidate", deployable: true, is_ready: true, val_ndcg_at_10: 0.0980, latency_p95_ms: 3.42, description: "Cosine similarity." },
      { model_name: "recsys_als", display_name: "ALS Matrix Factorization", family: "als", stage: "candidate", deployable: true, is_ready: true, val_ndcg_at_10: 0.0890, latency_p95_ms: 4.10, description: "Implicit ALS." }
    ]);
    return;
  }
  if (pathname === "/api/v1/recsys/library/benchmark" || pathname === "/api/v1/recsys/library/trio") {
    ok([
      { app_id: 1245620, title: "ELDEN RING", header_image_url: "https://cdn.akamai.steamstatic.com/steam/apps/1245620/header.jpg", genres: [{ id: "1", description: "Action" }], tags: ["Souls-like", "RPG"] },
      { app_id: 1091500, title: "Cyberpunk 2077", header_image_url: "https://cdn.akamai.steamstatic.com/steam/apps/1091500/header.jpg", genres: [{ id: "1", description: "RPG" }], tags: ["Cyberpunk", "Open World"] },
      { app_id: 814380, title: "Sekiro: Shadows Die Twice", header_image_url: "https://cdn.akamai.steamstatic.com/steam/apps/814380/header.jpg", genres: [{ id: "1", description: "Action" }], tags: ["Souls-like", "Difficult"] }
    ]);
    return;
  }
  if (pathname === "/api/v1/recsys/for-you") {
    ok({
      items: [{ rank: 1, app_id: 814380, item_idx: 42, title: "Sekiro: Shadows Die Twice", header_image_url: "https://cdn.akamai.steamstatic.com/steam/apps/814380/header.jpg", score: 0.892, price_final: "59.99", positive_ratio: 95, user_reviews: 120000, tags: ["Souls-like"], description_short: "Action game", popularity_rank: 15, win: true, mac: false, linux: false, steam_deck: true, because_you_liked: [], reasons: ["Similar game"], candidate_source: "two_tower_clean_v2", similarity_to_seeds: 0.8415, retrieval_timing_ms: 3.42 }],
      total_items: 1,
      model_name: "recsys_two_tower_ranker_clean_v2",
      model_run_id: "ee6d93db-6e51-4548-918f-d5133b214990",
      val_ndcg_at_10: 0.1245,
      dataset_version: "v1.0",
      universe_size: 190,
      unknown_app_ids: [],
      warning: null,
      preview: false,
      took_ms: 12.18,
      debug: { candidate_source_breakdown: { two_tower: 20 }, candidate_retrieval_ms: 6.84, scoring_ms: 4.12, reranking_ms: 1.22, total_pipeline_ms: 12.18 }
    });
    return;
  }
  if (pathname === "/api/v1/recsys/search" || pathname === "/api/v1/recsys/items/search") {
    ok({ items: [{ app_id: 1145360, item_idx: 10, title: "Hades", header_image_url: hadesDetail.header_image, date_release: "2020-09-17", price_final: "24.99", positive_ratio: 98, user_reviews: 210000, tags: ["Action"], description_short: "Action", has_description: true, train_positives: 1500, popularity_rank: 8, win: true, mac: true, linux: false, steam_deck: true }], total: 1, universe_size: 190, query: "" });
    return;
  }
  if (pathname === "/api/v1/recsys/map" || pathname === "/api/v1/recsys/embeddings") {
    ok({
      points: [{ app_id: 1145360, title: "Hades", x: 0.25, y: -0.35, cluster_id: 1, cluster_label: "Action Roguelike", popularity_rank: 8, header_image_url: hadesDetail.header_image, tags: ["Action"] }],
      total_points: 1,
      universe_size: 190
    });
    return;
  }
  if (pathname.includes("/neighbors")) {
    ok({
      neighbors: [{ rank: 1, app_id: 814380, title: "Sekiro: Shadows Die Twice", header_image_url: hadesDetail.header_image, similarity: 0.85, price_final: 59.99, positive_ratio: 95, user_reviews: 120000, tags: ["Action"], popularity_rank: 15 }]
    });
    return;
  }
  if (pathname === "/api/v1/recsys/eval/leaderboard") {
    ok(leaderboardJson);
    return;
  }
  if (pathname === "/api/v1/recsys/eval/runs") {
    ok({ runs: [{ model_run_id: "ee6d93db-6e51-4548-918f-d5133b214990", model_name: "recsys_two_tower_clean", created_at: "2026-10-20T12:00:00Z", dataset_version: "v1.0", status: "completed", metrics: { val_ndcg_at_10: 0.1245, latency_p95_ms: 12.18 } }] });
    return;
  }
  if (pathname.startsWith("/api/v1/recsys/eval/report/") || pathname.startsWith("/api/v1/recsys/reports/")) {
    const reportKey = pathname.split("/").pop();
    const reportData = loadResultJson(`${reportKey}.json`, {});
    ok(reportData);
    return;
  }

  // Assistant endpoints
  if (pathname === "/api/v1/assistant/health") {
    ok({ status: "ready", is_ready: true, tau: 0.63, calibrated: true });
    return;
  }
  if (pathname === "/api/v1/assistant/eval") {
    ok(assistantEvalJson);
    return;
  }
  if (pathname === "/api/v1/assistant/suggested-questions") {
    ok({ questions: ["similar game to elden ring"] });
    return;
  }
  if (pathname === "/api/v1/assistant/ask") {
    ok({ answer: "Hades is a fast-paced roguelike.", intent: "game_info", confidence: 0.95, calibrated: true, retrieved_games: [], latency_ms: 12.0 });
    return;
  }

  // Market endpoints
  if (pathname === "/api/v1/market/opportunities") {
    ok(marketOpportunitiesMock);
    return;
  }
  if (pathname === "/api/v1/market/trends") {
    ok(marketTrendsMock);
    return;
  }

  // Ops endpoints
  if (pathname === "/api/v1/ops/telemetry") {
    ok({ status: "healthy", timestamp: "2026-10-22T12:00:00Z", models: { recsys_two_tower_clean: { status: "healthy", ks_pvalue: 0.42, psi: 0.045, drift_detected: false } } });
    return;
  }
  if (pathname === "/api/v1/ops/drift") {
    ok(driftDetectorPowerJson);
    return;
  }
  if (pathname === "/api/v1/ops/metrics") {
    res.writeHead(200, { "Content-Type": "text/plain; version=0.0.4" });
    res.end("# HELP steamiq_requests_total Total HTTP requests\n# TYPE steamiq_requests_total counter\nsteamiq_requests_total{endpoint=\"/api/v1/recsys/for-you\",status=\"200\"} 1000\n");
    return;
  }
  if (pathname.startsWith("/api/v1/ops/evidence/")) {
    const evName = pathname.split("/").pop();
    const evData = loadResultJson(`${evName}.json`, {});
    ok(evData);
    return;
  }
  if (pathname === "/api/v1/ops/model-cards") {
    ok([{ model_id: "recsys_two_tower_clean", model_name: "Two-Tower Recommender (Clean)", markdown: "# Two-Tower Recommender\n\nDual-encoder architecture." }]);
    return;
  }

  // Catch-all
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ success: false, data: null, error: { code: "NOT_FOUND", message: `Unrecognized route: ${pathname}` }, meta: null }));
});

const PORT = 8000;
server.listen(PORT, "127.0.0.1", () => {
  console.log(`[MockServer] SteamIQ mock API server listening on http://127.0.0.1:${PORT}`);
});
