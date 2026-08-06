import Link from "next/link";

export default function NotFound() {
  return (
    <div className="container">
      <div className="empty-state" style={{ paddingTop: 120 }}>
        <div className="empty-state__icon">🎮</div>
        <h1 className="empty-state__title">Game not found</h1>
        <p className="empty-state__body">
          This game hasn&apos;t been ingested yet. Run{" "}
          <code style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-primary)" }}>
            make ingest APPID=&lt;id&gt;
          </code>{" "}
          to add it to the database.
        </p>
        <a
          href="/"
          style={{
            display: "inline-block",
            marginTop: 24,
            color: "var(--color-accent-primary)",
            textDecoration: "underline",
            fontSize: "0.9rem",
          }}
        >
          ← Back to search
        </a>
      </div>
    </div>
  );
}
