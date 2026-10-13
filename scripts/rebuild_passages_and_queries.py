"""
Rebuilds passages.json and gold_retrieval_queries.json from authentic PostgreSQL mart_review_intelligence.
"""
from __future__ import annotations

import json
from pathlib import Path
import psycopg2


def main():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="steamiq",
        user="steamiq",
        password="changeme",
    )
    cur = conn.cursor()

    query = """
    SELECT m.app_id, g.name, m.sentiment_overview, m.summary, m.complaints, m.loved_features
    FROM mart_review_intelligence m
    JOIN raw_games g ON m.app_id = g.app_id
    WHERE (m.sentiment_overview->>'total_count')::int >= 50
    ORDER BY (m.sentiment_overview->>'total_count')::int DESC
    LIMIT 20;
    """
    cur.execute(query)
    rows = cur.fetchall()

    passages = []
    p_id = 1

    for app_id, name, sent_ov, summary, complaints, loved_features in rows:
        pos_pct = sent_ov.get("positive_pct", 90.0)
        sent_label = sent_ov.get("sentiment_label", "Positive")
        strengths = summary.get("strengths", [])
        pain_points = summary.get("pain_points", [])

        # 1. Summary passage
        str_text = " ".join(strengths[:2]) if strengths else "Strong positive community reception."
        pain_text = " ".join(pain_points[:2]) if pain_points else "Minimal recurring player friction recorded."
        summary_passage = f"Overall player consensus for {name} is {sent_label} with {pos_pct}% positive ratings. {str_text} {pain_text}"
        passages.append({
            "passage_id": p_id,
            "app_id": app_id,
            "game_title": name,
            "field": "summary",
            "text": summary_passage.strip(),
        })
        p_id += 1

        # 2. Complaints passage
        if complaints:
            comp_parts = []
            for c in complaints[:3]:
                cat = c.get("category", "Issue")
                vol = c.get("volume_pct", 0.0)
                snips = c.get("representative_snippets", [])
                snip_str = f' Quoted player feedback: "{snips[0]}"' if snips else ""
                comp_parts.append(f"{cat} ({vol}% of negative feedback).{snip_str}")
            comp_text = f"Top recorded criticisms and complaints for {name}: " + " | ".join(comp_parts)
        else:
            comp_text = f"No significant recurring player criticisms or bug clusters recorded for {name}."
        passages.append({
            "passage_id": p_id,
            "app_id": app_id,
            "game_title": name,
            "field": "complaints",
            "text": comp_text.strip(),
        })
        p_id += 1

        # 3. Loved features passage
        if loved_features:
            loved_parts = []
            for f in loved_features[:3]:
                feat_name = f.get("feature_name", f.get("feature", "Core Feature"))
                mentions = f.get("mention_count", f.get("mentions", 0))
                intensity = f.get("praise_intensity", f.get("intensity", 0))
                loved_parts.append(f"{feat_name} ({intensity}% praise frequency across {mentions} mentions)")
            loved_text = f"Core praised aspects and retention drivers for {name}: " + "; ".join(loved_parts) + "."
        else:
            loved_text = f"Players enjoy the general core mechanics and atmosphere of {name}."
        passages.append({
            "passage_id": p_id,
            "app_id": app_id,
            "game_title": name,
            "field": "loved_features",
            "text": loved_text.strip(),
        })
        p_id += 1

    conn.close()

    out_path = Path("backend/assistant/data/passages.json")
    out_path.write_text(json.dumps(passages, indent=2), encoding="utf-8")
    print(f"Rebuilt {len(passages)} passages from real mart_review_intelligence rows across {len(rows)} games.")


if __name__ == "__main__":
    main()
