"use client";

import React from "react";
import { DevOnlyGate } from "@/components/DevOnlyGate";

export default function MonitoringPage() {
  return (
    <DevOnlyGate pageTitle="Monitoring & Operations">
      <main className="container" style={{ paddingTop: "88px", paddingBottom: "64px" }}>
        <h1>Monitoring</h1>
        <p>Operations & Telemetry</p>
      </main>
    </DevOnlyGate>
  );
}
