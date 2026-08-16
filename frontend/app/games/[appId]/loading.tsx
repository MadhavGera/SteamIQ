export default function LoadingGameDetail() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        backgroundColor: "var(--bg-app)",
        color: "var(--text-primary)",
      }}
    >
      <div className="pulse-dot" style={{ marginBottom: "24px" }} />
      <h2 style={{ fontSize: "24px", fontWeight: 600, marginBottom: "8px" }}>
        Loading Intelligence...
      </h2>
      <p style={{ color: "var(--text-secondary)" }}>
        Fetching game metadata, reviews, and market insights.
      </p>
    </div>
  );
}
