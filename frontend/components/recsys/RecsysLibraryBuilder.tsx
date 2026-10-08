"use client";

import React, { useState, useEffect, useRef, useTransition } from "react";
import { api, RecsysCatalogItem } from "@/lib/api";

// Verified candidate preset IDs across universe
const PRESET_CONFIGS = [
  {
    id: "action_souls",
    name: "Souls-like & Action RPG",
    icon: "swords",
    appIds: [1245620, 374320, 292030, 1091500], // Elden Ring, Dark Souls III, Witcher 3, Cyberpunk
  },
  {
    id: "cozy_farming",
    name: "Cozy & Simulation",
    icon: "eco",
    appIds: [413150, 105600, 294100, 892970], // Stardew Valley, Terraria, RimWorld, Valheim
  },
  {
    id: "roguelike_deck",
    name: "Roguelike & Indie",
    icon: "casino",
    appIds: [1145360, 646570, 367520, 588650], // Hades, Slay the Spire, Hollow Knight, Dead Cells
  },
];

interface RecsysLibraryBuilderProps {
  selectedGames: RecsysCatalogItem[];
  onChange: (games: RecsysCatalogItem[]) => void;
  universeSize: number | null;
}

export function RecsysLibraryBuilder({
  selectedGames,
  onChange,
  universeSize,
}: RecsysLibraryBuilderProps) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<RecsysCatalogItem[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [resolvedPresets, setResolvedPresets] = useState<
    Array<{ id: string; name: string; icon: string; items: RecsysCatalogItem[] }>
  >([]);
  const [, startTransition] = useTransition();

  const searchInputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // 1. Verify candidate presets against the actual catalog dynamically on mount
  useEffect(() => {
    let isMounted = true;

    async function loadPresets() {
      const validPresets: Array<{
        id: string;
        name: string;
        icon: string;
        items: RecsysCatalogItem[];
      }> = [];

      for (const preset of PRESET_CONFIGS) {
        try {
          const items: RecsysCatalogItem[] = [];
          for (const appId of preset.appIds) {
            // Search or find item by exact query or prefix
            const res = await api.searchRecsysItems(String(appId), 5);
            const match = res.items.find((it) => it.app_id === appId);
            if (match) {
              items.push(match);
            }
          }
          // Only show preset if all of its games were resolved from the catalog
          if (items.length === preset.appIds.length) {
            validPresets.push({
              id: preset.id,
              name: preset.name,
              icon: preset.icon,
              items,
            });
          }
        } catch {
          // If any preset resolution fails, omit it gracefully
        }
      }

      if (isMounted) {
        setResolvedPresets(validPresets);
      }
    }

    loadPresets();
    return () => {
      isMounted = false;
    };
  }, []);

  // 2. Check sessionStorage on mount for items added from Embedding Map
  useEffect(() => {
    if (typeof window === "undefined") return;
    const pendingAppIdStr = sessionStorage.getItem("recsys_add_game_id");
    if (pendingAppIdStr) {
      sessionStorage.removeItem("recsys_add_game_id");
      const appId = parseInt(pendingAppIdStr, 10);
      if (!isNaN(appId)) {
        api.searchRecsysItems(String(appId), 5).then((res) => {
          const found = res.items.find((it) => it.app_id === appId);
          if (found && !selectedGames.some((g) => g.app_id === found.app_id)) {
            if (selectedGames.length < 50) {
              onChange([...selectedGames, found]);
            }
          }
        }).catch(() => {
          // Ignore if lookup fails
        });
      }
    }
  }, [onChange, selectedGames]);

  // 3. Debounced (250 ms) typeahead search
  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    if (!query.trim()) {
      setSuggestions([]);
      setIsOpen(false);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    debounceTimerRef.current = setTimeout(async () => {
      try {
        const res = await api.searchRecsysItems(query.trim(), 8);
        startTransition(() => {
          setSuggestions(res.items);
          setIsOpen(true);
          setIsLoading(false);
        });
      } catch {
        setIsLoading(false);
      }
    }, 250);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [query]);

  // 4. Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        searchInputRef.current &&
        !searchInputRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelectGame = (game: RecsysCatalogItem) => {
    if (selectedGames.length >= 50) return;
    if (!selectedGames.some((g) => g.app_id === game.app_id)) {
      onChange([...selectedGames, game]);
    }
    setQuery("");
    setIsOpen(false);
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  };

  const handleRemoveGame = (appId: number) => {
    onChange(selectedGames.filter((g) => g.app_id !== appId));
  };

  const handleApplyPreset = (items: RecsysCatalogItem[]) => {
    // Merge without duplicates up to 50 items
    const map = new Map<number, RecsysCatalogItem>();
    items.forEach((g) => map.set(g.app_id, g));
    onChange(Array.from(map.values()).slice(0, 50));
  };

  const handleClearAll = () => {
    onChange([]);
  };

  const isFull = selectedGames.length >= 50;

  return (
    <section
      className="card"
      style={{
        backgroundColor: "var(--bg-surface)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-card)",
        padding: "24px",
        marginBottom: "24px",
      }}
    >
      {/* Section Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: "12px",
          marginBottom: "16px",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span
              className="material-symbols-outlined"
              style={{ fontSize: "20px", color: "var(--accent-primary)" }}
            >
              sports_esports
            </span>
            <h2 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)" }}>
              Your Seed Library
            </h2>
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "4px" }}>
            Select games you enjoy. Recommendations are computed instantly using the offline-trained models.
            {universeSize !== null && (
              <span style={{ color: "var(--text-muted)", marginLeft: "6px" }}>
                (Search covers the <strong>{universeSize.toLocaleString()}</strong> modeled catalog items)
              </span>
            )}
          </p>
        </div>

        {/* Counter and Clear Button */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span
            className={`badge-pill ${
              selectedGames.length >= 3 ? "badge-pill--success" : "badge-pill--warning"
            }`}
            style={{ fontFamily: "var(--font-mono, monospace)" }}
          >
            {selectedGames.length} / 50 selected {selectedGames.length < 3 ? "(3+ recommended)" : ""}
          </span>
          {selectedGames.length > 0 && (
            <button
              onClick={handleClearAll}
              className="btn btn--secondary"
              style={{ padding: "4px 10px", fontSize: "12px" }}
              aria-label="Clear all selected games"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Preset Buttons Bar */}
      {resolvedPresets.length > 0 && (
        <div style={{ marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Sample Libraries:
          </span>
          {resolvedPresets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handleApplyPreset(preset.items)}
              className="popular-chip"
              style={{
                cursor: "pointer",
                padding: "6px 12px",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
              }}
              aria-label={`Apply sample library ${preset.name}`}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "var(--accent-primary)" }}>
                {preset.icon}
              </span>
              <span>{preset.name}</span>
              <span className="popular-chip__id">({preset.items.length})</span>
            </button>
          ))}
        </div>
      )}

      {/* Search Input with Debounced Typeahead */}
      <div style={{ position: "relative", marginBottom: "16px" }}>
        <div className="stitch-search-container" style={{ maxWidth: "100%" }}>
          <span
            className="material-symbols-outlined"
            style={{ fontSize: "20px", color: "var(--text-muted)", marginLeft: "4px" }}
          >
            search
          </span>
          <input
            ref={searchInputRef}
            type="text"
            className="stitch-search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => {
              if (suggestions.length > 0) setIsOpen(true);
            }}
            placeholder={
              isFull
                ? "Library is full (50 games max)"
                : "Search catalog games (e.g. Hades, Elden Ring, Stardew Valley)..."
            }
            disabled={isFull}
            aria-label="Search games for seed library"
          />
          {isLoading && (
            <div className="loading-bars" style={{ height: "16px", marginRight: "8px" }}>
              <span />
              <span />
              <span />
            </div>
          )}
        </div>

        {/* Suggestions Dropdown */}
        {isOpen && suggestions.length > 0 && (
          <div
            ref={dropdownRef}
            style={{
              position: "absolute",
              top: "100%",
              left: 0,
              right: 0,
              marginTop: "4px",
              backgroundColor: "var(--bg-surface-low)",
              border: "1px solid var(--border-strong)",
              borderRadius: "var(--radius-card)",
              boxShadow: "var(--shadow-dropdown)",
              zIndex: 60,
              maxHeight: "320px",
              overflowY: "auto",
            }}
          >
            {suggestions.map((item) => {
              const isSelected = selectedGames.some((g) => g.app_id === item.app_id);
              return (
                <button
                  key={item.app_id}
                  onClick={() => handleSelectGame(item)}
                  disabled={isSelected || isFull}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "10px 14px",
                    background: "transparent",
                    border: "none",
                    borderBottom: "1px solid var(--border-subtle)",
                    color: isSelected ? "var(--text-muted)" : "var(--text-primary)",
                    cursor: isSelected || isFull ? "default" : "pointer",
                    textAlign: "left",
                    transition: "background-color var(--transition-fast)",
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected && !isFull) {
                      e.currentTarget.style.backgroundColor = "var(--bg-surface-raised)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "12px", minWidth: 0 }}>
                    <div
                      style={{
                        width: "60px",
                        height: "28px",
                        backgroundColor: "var(--bg-surface-lowest)",
                        borderRadius: "var(--radius-sm)",
                        overflow: "hidden",
                        flexShrink: 0,
                      }}
                    >
                      {item.header_image_url ? (
                        <img
                          src={item.header_image_url}
                          alt={item.title}
                          style={{ width: "100%", height: "100%", objectFit: "cover" }}
                          onError={(e) => {
                            e.currentTarget.style.display = "none";
                          }}
                        />
                      ) : (
                        <div style={{ width: "100%", height: "100%", background: "var(--bg-surface)" }} />
                      )}
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: "13px",
                          fontWeight: 600,
                          color: isSelected ? "var(--text-muted)" : "var(--text-primary)",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {item.title}
                      </div>
                      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        ID: {item.app_id} • Pos: {item.positive_ratio ? `${Math.round(item.positive_ratio * 100)}%` : "N/A"}
                      </div>
                    </div>
                  </div>

                  <div>
                    {isSelected ? (
                      <span className="badge-pill badge-pill--neutral" style={{ fontSize: "10px" }}>
                        Selected
                      </span>
                    ) : (
                      <span
                        className="material-symbols-outlined"
                        style={{ fontSize: "18px", color: "var(--accent-primary)" }}
                      >
                        add_circle
                      </span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Selected Games Chips */}
      {selectedGames.length > 0 ? (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            maxHeight: "180px",
            overflowY: "auto",
            padding: "4px 0",
          }}
        >
          {selectedGames.map((game) => (
            <div
              key={game.app_id}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Backspace" || e.key === "Delete") {
                  handleRemoveGame(game.app_id);
                }
              }}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                backgroundColor: "var(--bg-surface-raised)",
                border: "1px solid var(--border-strong)",
                borderRadius: "var(--radius-pill)",
                padding: "4px 10px 4px 6px",
                fontSize: "12px",
                color: "var(--text-primary)",
                userSelect: "none",
              }}
            >
              <div
                style={{
                  width: "24px",
                  height: "24px",
                  borderRadius: "50%",
                  overflow: "hidden",
                  flexShrink: 0,
                  backgroundColor: "var(--bg-surface-lowest)",
                }}
              >
                {game.header_image_url ? (
                  <img
                    src={game.header_image_url}
                    alt=""
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    onError={(e) => {
                      e.currentTarget.style.display = "none";
                    }}
                  />
                ) : null}
              </div>

              <span
                style={{
                  maxWidth: "180px",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  fontWeight: 600,
                }}
                title={game.title}
              >
                {game.title}
              </span>

              <button
                onClick={() => handleRemoveGame(game.app_id)}
                aria-label={`Remove ${game.title} from library`}
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  padding: "0",
                  color: "var(--text-muted)",
                  transition: "color var(--transition-fast)",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.color = "var(--danger)")}
                onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
              >
                <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                  cancel
                </span>
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div
          style={{
            padding: "16px",
            backgroundColor: "var(--bg-surface-low)",
            borderRadius: "var(--radius-sm)",
            border: "1px dashed var(--border-subtle)",
            textAlign: "center",
            fontSize: "13px",
            color: "var(--text-muted)",
          }}
        >
          No games selected yet. Search above or click a Sample Library preset to begin.
        </div>
      )}
    </section>
  );
}
