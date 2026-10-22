"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

export type UserMode = "player" | "developer";

interface UserModeContextValue {
  mode: UserMode;
  setMode: (mode: UserMode) => void;
  toggleMode: () => void;
  isPlayer: boolean;
  isDeveloper: boolean;
  hydrated: boolean;
}

const UserModeContext = createContext<UserModeContextValue | undefined>(undefined);

const STORAGE_KEY = "steamiq_user_mode";

export function UserModeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<UserMode>("developer");
  const [hydrated, setHydrated] = useState<boolean>(false);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "player" || stored === "developer") {
        setModeState(stored);
      }
    } catch {
      // Ignore localStorage read errors
    } finally {
      setHydrated(true);
    }
  }, []);

  const setMode = (newMode: UserMode) => {
    setModeState(newMode);
    try {
      localStorage.setItem(STORAGE_KEY, newMode);
    } catch {
      // Ignore localStorage write errors
    }
  };

  const toggleMode = () => {
    setMode(mode === "developer" ? "player" : "developer");
  };

  return (
    <UserModeContext.Provider
      value={{
        mode,
        setMode,
        toggleMode,
        isPlayer: mode === "player",
        isDeveloper: mode === "developer",
        hydrated,
      }}
    >
      {children}
    </UserModeContext.Provider>
  );
}

export function useUserMode(): UserModeContextValue {
  const context = useContext(UserModeContext);
  if (!context) {
    // Return graceful fallback when rendered outside provider
    return {
      mode: "developer",
      setMode: () => {},
      toggleMode: () => {},
      isPlayer: false,
      isDeveloper: true,
      hydrated: false,
    };
  }
  return context;
}
