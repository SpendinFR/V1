import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timedelta
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

JSONLike = Dict[str, Any]


LOGGER = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[JSONLike]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def read_json(path: str) -> Optional[JSONLike]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bar(x: float, width: int = 20) -> str:
    x = max(0.0, min(1.0, x))
    n = int(x * width)
    return "[" + "#" * n + "-" * (width - n) + f"] {x:.2f}"


def section(title: str):
    print("\n" + title)
    print("-" * len(title))


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except ValueError:
            return None
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value[: len(fmt)], fmt)
            except ValueError:
                continue
    return None


def filter_since(data: Sequence[JSONLike], since: Optional[datetime]) -> List[JSONLike]:
    if not since:
        return list(data)
    out: List[JSONLike] = []
    for item in data:
        ts = (
            parse_timestamp(item.get("timestamp"))
            or parse_timestamp(item.get("time"))
            or parse_timestamp(item.get("created_at"))
        )
        if ts and ts >= since:
            out.append(item)
    return out


def load_logs(since: Optional[datetime] = None) -> Dict[str, List[JSONLike]]:
    reasoning = filter_since(read_jsonl("logs/reasoning.jsonl"), since)
    experiments = filter_since(read_jsonl("logs/experiments.jsonl"), since)
    metacog = filter_since(read_jsonl("logs/metacog.log"), since)
    goals_raw = read_json("logs/goals_dag.json") or {}
    dag_nodes = goals_raw.get("nodes", {})
    goals = list(dag_nodes.values())
    if since:
        recent_goals = [
            g
            for g in goals
            if parse_timestamp(g.get("updated_at"))
            and parse_timestamp(g.get("updated_at")) >= since
        ]
        if recent_goals:
            goals = recent_goals
    return {
        "reasoning": reasoning,
        "experiments": experiments,
        "goals": goals,
        "metacog": metacog,
        "dag": goals_raw,
    }


def compute_reasoning_metrics(reasoning: Sequence[JSONLike]) -> Dict[str, Any]:
    if not reasoning:
        return {}
    confidences = [
        r.get("final_confidence")
        for r in reasoning
        if isinstance(r.get("final_confidence"), (int, float))
    ]
    durations = [
        r.get("reasoning_time")
        for r in reasoning
        if isinstance(r.get("reasoning_time"), (int, float))
    ]
    tokens = [
        r.get("completion_tokens")
        for r in reasoning
        if isinstance(r.get("completion_tokens"), (int, float))
    ]
    return {
        "count": len(reasoning),
        "avg_confidence": mean(confidences) if confidences else None,
        "avg_duration": mean(durations) if durations else None,
        "avg_tokens": mean(tokens) if tokens else None,
    }


def compute_experiment_metrics(experiments: Sequence[JSONLike]) -> Dict[str, Any]:
    if not experiments:
        return {}
    outcomes = [
        e.get("outcome")
        for e in experiments
        if isinstance(e.get("outcome"), dict)
    ]
    successes = [o for o in outcomes if o.get("success") is True]
    failures = [o for o in outcomes if o.get("success") is False]
    metrics = Counter(o.get("metric") for o in outcomes if o.get("metric"))
    return {
        "tested": len(outcomes),
        "success_rate": (len(successes) / len(outcomes)) if outcomes else None,
        "common_metrics": metrics.most_common(3),
        "recent_failure": failures[-1] if failures else None,
    }


def compute_goal_metrics(goals: Sequence[JSONLike]) -> Dict[str, Any]:
    if not goals:
        return {}
    progressing = [g for g in goals if g.get("progress", 0) >= 1]
    stalled = [
        g
        for g in goals
        if g.get("status") == "active" and g.get("progress", 0) < 0.3 and g.get("competence", 0) < 0.3
    ]
    high_value = sorted(
        goals,
        key=lambda g: g.get("value", 0),
        reverse=True,
    )[:3]
    return {
        "active": len([g for g in goals if g.get("status") == "active"]),
        "completed": len(progressing),
        "stalled": stalled[:3],
        "high_value": high_value,
    }


def compute_insights(data: Dict[str, List[JSONLike]]) -> Dict[str, Any]:
    base = {
        "reasoning": compute_reasoning_metrics(data.get("reasoning", [])),
        "experiments": compute_experiment_metrics(data.get("experiments", [])),
        "goals": compute_goal_metrics(data.get("goals", [])),
    }
    llm_summary = try_call_llm_dict(
        "runtime_dash",
        input_payload={"metrics": base},
        logger=LOGGER,
    )
    if llm_summary:
        summary = llm_summary.get("daily_summary")
        actions = llm_summary.get("recommended_actions")
        if isinstance(summary, str):
            base.setdefault("llm_report", {})["daily_summary"] = summary
        if isinstance(actions, list):
            base.setdefault("llm_report", {})["recommended_actions"] = actions
        notes = llm_summary.get("notes")
        if isinstance(notes, str):
            base.setdefault("llm_report", {})["notes"] = notes
    return base


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return f"{value:.2f}"


def render_reasoning_rows(
    reasoning: Sequence[JSONLike],
    limit: int,
    goal_filter: Optional[str] = None,
) -> List[str]:
    rows: List[str] = []
    for r in list(reasoning)[-limit * 3 :]:
        goal = r.get("goal") or {}
        title = goal.get("title", "?")
        if goal_filter and goal_filter.lower() not in title.lower():
            continue
        sol = r.get("solution")
        conf = r.get("final_confidence", 0.5)
        t = r.get("reasoning_time", 0.0)
        rows.append(
            f"- but={title} | conf={conf:.2f} | t={t:.2f}s | solution={str(sol)[:80]}"
        )
    return rows[-limit:]


def render_experiment_rows(
    experiments: Sequence[JSONLike],
    limit: int,
    goal_filter: Optional[str] = None,
) -> List[str]:
    rows: List[str] = []
    for e in list(experiments)[-limit * 3 :]:
        if goal_filter:
            linked = e.get("goal_id") or e.get("goal")
            if linked and goal_filter.lower() not in str(linked).lower():
                continue
        if isinstance(e.get("outcome"), dict):
            o = e["outcome"]
            rows.append(
                f"- Résultat {o.get('metric', '?')} : {'OK' if o.get('success') else 'KO'} "
                f"({o.get('observed', 0):.2f} vs {o.get('goal', 0):.2f})"
            )
        else:
            rows.append(
                f"- Plan: {e.get('metric', '?')} baseline={e.get('baseline', '?')} "
                f"target={e.get('target_change', '?')} plan={e.get('plan', {})}"
            )
    return rows[-limit:]


def render_goal_rows(
    goals: Sequence[JSONLike],
    limit: int,
    goal_filter: Optional[str] = None,
) -> List[JSONLike]:
    active = [
        n
        for n in goals
        if n.get("status") == "active" and n.get("progress", 0) < 1
    ]
    if goal_filter:
        active = [
            g for g in active if goal_filter.lower() in g.get("description", "").lower()
        ]
    active.sort(
        key=lambda x: (x.get("value", 0), 1 - x.get("progress", 0)), reverse=True
    )
    return active[:limit]


def console_once(
    *,
    reasoning_limit: int = 5,
    experiments_limit: int = 10,
    goals_limit: int = 5,
    goal_filter: Optional[str] = None,
    since: Optional[datetime] = None,
):
    data = load_logs(since)
    data_for_display = (
        apply_goal_filter_to_data(data, goal_filter)
        if goal_filter
        else data
    )
    insights = compute_insights(data_for_display)

    section("⏱ Statut")
    print("Maintenant:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if since:
        print("Fenêtre depuis:", since.strftime("%Y-%m-%d %H:%M:%S"))
    if goal_filter:
        print("Filtre objectif:", goal_filter)

    section("📊 Synthèse")
    reasoning_insights = insights.get("reasoning", {})
    if reasoning_insights:
        print(
            "Raisonnements:",
            f"{reasoning_insights['count']} entrées | conf. moyenne={format_metric(reasoning_insights.get('avg_confidence'))} |",
            f"durée moyenne={format_metric(reasoning_insights.get('avg_duration'))}s",
            end="",
        )
        if reasoning_insights.get("avg_tokens") is not None:
            print(f" | tokens moyen={format_metric(reasoning_insights['avg_tokens'])}")
        else:
            print()
    experiments_insights = insights.get("experiments", {})
    if experiments_insights:
        rate = experiments_insights.get("success_rate")
        rate_txt = f"{rate*100:.1f}%" if rate is not None else "?"
        print(
            "Expériences:",
            f"{experiments_insights.get('tested', 0)} tests | taux de succès={rate_txt}",
        )
        if experiments_insights.get("common_metrics"):
            top = ", ".join(
                f"{m} ({c})" for m, c in experiments_insights["common_metrics"]
            )
            print("Top métriques:", top)
        if experiments_insights.get("recent_failure"):
            failure = experiments_insights["recent_failure"]
            print(
                "Dernier échec:",
                f"{failure.get('metric', '?')} ({failure.get('observed', 0):.2f} vs {failure.get('goal', 0):.2f})",
            )
    goals_insights = insights.get("goals", {})
    if goals_insights:
        print(
            "Objectifs:",
            f"{goals_insights.get('active', 0)} actifs | {goals_insights.get('completed', 0)} complétés",
        )
        if goals_insights.get("stalled"):
            stalled_labels = ", ".join(
                g.get("goal_id", "?") for g in goals_insights["stalled"]
            )
            print("⚠️ Objectifs potentiellement bloqués:", stalled_labels)

    section(f"🧠 Raisonnement (derniers {reasoning_limit})")
    reasoning_rows = render_reasoning_rows(
        data_for_display.get("reasoning", []), reasoning_limit, goal_filter
    )
    for row in reasoning_rows:
        print(row)
    if not reasoning_rows:
        print("(aucun élément dans la fenêtre)")

    section(f"🎯 Objectifs (top {goals_limit} actifs)")
    goals_rows = render_goal_rows(
        data_for_display.get("goals", []), goals_limit, goal_filter
    )
    for n in goals_rows:
        print(f"- {n.get('goal_id', '?')}: {n.get('description', '?')}")
        print(
            f"  progress {bar(n.get('progress', 0))} | "
            f"competence {bar(n.get('competence', 0))} | "
            f"value {bar(n.get('value', 0))}"
        )
    if not goals_rows:
        print("(aucun objectif actif trouvé)")

    section(f"🧪 Expériences (derniers {experiments_limit})")
    exp_rows = render_experiment_rows(
        data_for_display.get("experiments", []), experiments_limit, goal_filter
    )
    for row in exp_rows:
        print(row)
    if not exp_rows:
        print("(aucune expérience dans la fenêtre)")

    section("📈 Métacog (derniers 5 événements)")
    metacog = data_for_display.get("metacog", [])
    for m in metacog[-5:]:
        ts = (
            parse_timestamp(m.get("timestamp"))
            or parse_timestamp(m.get("time"))
            or datetime.fromtimestamp(time.time())
        )
        description = m.get("description", "")
        print(
            f"- {ts.strftime('%H:%M:%S')} {m.get('event_type', '?')}: {description[:80]}"
        )
    if not metacog:
        print("(aucun événement métacognitif)")


def console_watch(
    interval: float = 5.0,
    **kwargs: Any,
):
    try:
        while True:
            print("\033c", end="")  # Clear screen
            console_once(**kwargs)
            print(f"\n(Prochaine actualisation dans {interval:.1f}s — Ctrl+C pour arrêter)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nArrêt du mode watch.")


def goal_token_matches(token_candidates: Iterable[Any], goal_filter: Optional[str]) -> bool:
    if not goal_filter:
        return True
    goal_filter = goal_filter.lower()
    for token in token_candidates:
        if token and goal_filter in str(token).lower():
            return True
    return False


def apply_goal_filter_to_data(
    data: Dict[str, List[JSONLike]], goal_filter: Optional[str]
) -> Dict[str, List[JSONLike]]:
    if not goal_filter:
        return data
    filtered: Dict[str, List[JSONLike]] = {}
    goal_filter_lower = goal_filter.lower()

    def matches_reasoning(item: JSONLike) -> bool:
        goal = item.get("goal") or {}
        tokens = [goal.get("title"), goal.get("goal_id"), item.get("goal_id")]
        return goal_token_matches(tokens, goal_filter_lower)

    def matches_experiment(item: JSONLike) -> bool:
        tokens = [
            item.get("goal_id"),
            item.get("goal"),
            item.get("metric"),
        ]
        return goal_token_matches(tokens, goal_filter_lower)

    def matches_goal(item: JSONLike) -> bool:
        tokens = [item.get("goal_id"), item.get("description"), item.get("title")]
        return goal_token_matches(tokens, goal_filter_lower)

    filtered["reasoning"] = [
        r for r in data.get("reasoning", []) if matches_reasoning(r)
    ]
    filtered["experiments"] = [
        e for e in data.get("experiments", []) if matches_experiment(e)
    ]
    filtered["goals"] = [
        g for g in data.get("goals", []) if matches_goal(g)
    ]
    filtered["metacog"] = data.get("metacog", [])
    filtered["dag"] = data.get("dag", {})
    return filtered


def build_html(
    data: Dict[str, List[JSONLike]],
    insights: Dict[str, Any],
    *,
    auto_refresh: int = 0,
    goal_filter: Optional[str] = None,
) -> str:
    reasoning = data.get("reasoning", [])[-50:]
    experiments = data.get("experiments", [])[-50:]
    goals = render_goal_rows(data.get("goals", []), 20, goal_filter)
    metacog = data.get("metacog", [])[-20:]

    def fmt_float(value: Any, suffix: str = "") -> str:
        if value is None:
            return "?"
        try:
            return f"{float(value):.2f}{suffix}"
        except (TypeError, ValueError):
            return "?"

    reasoning_rows = []
    chart_points = []
    for r in reversed(reasoning):
        goal = r.get("goal") or {}
        title = goal.get("title") or goal.get("goal_id") or "?"
        conf = fmt_float(r.get("final_confidence"))
        duration = fmt_float(r.get("reasoning_time"), "s")
        ts = parse_timestamp(r.get("timestamp")) or datetime.now()
        snippet = escape(str(r.get("solution"))[:120])
        title_html = escape(str(title))
        goal_attr = escape(str(goal.get("goal_id") or title))
        reasoning_rows.append(
            f"<tr data-goal=\"{goal_attr}\"><td>{title_html}</td><td>{conf}</td><td>{duration}</td><td><code>{snippet}</code></td></tr>"
        )
        chart_points.append(
            {
                "label": ts.strftime("%H:%M:%S"),
                "duration": float(r.get("reasoning_time") or 0.0),
                "confidence": float(r.get("final_confidence") or 0.0),
            }
        )

    experiment_rows = []
    for e in reversed(experiments):
        if isinstance(e.get("outcome"), dict):
            outcome = e["outcome"]
            success = outcome.get("success")
            metric = escape(str(outcome.get("metric", "?")))
            observed = fmt_float(outcome.get("observed"))
            goal_value = fmt_float(outcome.get("goal"))
            badge = "✅" if success else "❌"
            experiment_rows.append(
                f"<tr><td>{metric}</td><td>{badge}</td><td>{observed}</td><td>{goal_value}</td></tr>"
            )
        else:
            metric = escape(str(e.get("metric", "?")))
            baseline = escape(str(e.get("baseline", "?")))
            target = escape(str(e.get("target_change", "?")))
            plan = escape(str(e.get("plan", {}))[:120])
            experiment_rows.append(
                f"<tr><td>{metric}</td><td colspan=2>Plan</td><td><code>{plan}</code> (base={baseline}, cible={target})</td></tr>"
            )

    goals_rows = []
    for g in goals:
        goals_rows.append(
            "<tr>"
            + f"<td>{escape(str(g.get('goal_id', '?')))}</td>"
            + f"<td>{escape(str(g.get('description', g.get('title', '?'))))}</td>"
            + f"<td>{fmt_float(g.get('progress'))}</td>"
            + f"<td>{fmt_float(g.get('competence'))}</td>"
            + f"<td>{fmt_float(g.get('value'))}</td>"
            + "</tr>"
        )

    metacog_rows = []
    for m in reversed(metacog):
        ts = (
            parse_timestamp(m.get("timestamp"))
            or parse_timestamp(m.get("time"))
            or datetime.now()
        )
        metacog_rows.append(
            f"<li><strong>{ts.strftime('%Y-%m-%d %H:%M:%S')}</strong> — {escape(str(m.get('event_type', '?')))} : {escape(str(m.get('description', '')))}</li>"
        )

    auto_refresh_meta = (
        f'<meta http-equiv="refresh" content="{auto_refresh}">' if auto_refresh else ""
    )
    filter_info = (
        f"<p>Filtre objectif actif: <strong>{escape(goal_filter)}</strong></p>"
        if goal_filter
        else ""
    )

    reasoning_insight = insights.get("reasoning", {})
    experiments_insight = insights.get("experiments", {})
    goals_insight = insights.get("goals", {})

    insight_cards = [
        (
            "Raisonnements",
            f"{reasoning_insight.get('count', 0)} entrées",
            f"Confiance moyenne: {format_metric(reasoning_insight.get('avg_confidence'))}",
            f"Durée moyenne: {format_metric(reasoning_insight.get('avg_duration'))}s",
        ),
        (
            "Expériences",
            f"{experiments_insight.get('tested', 0)} tests",
            f"Taux de succès: {format_metric((experiments_insight.get('success_rate') or 0) * 100)}%",
            "Top métriques: "
            + ", ".join(
                f"{escape(str(m))} ({c})" for m, c in experiments_insight.get("common_metrics", [])
            ),
        ),
        (
            "Objectifs",
            f"{goals_insight.get('active', 0)} actifs",
            f"{goals_insight.get('completed', 0)} complétés",
            "Bloqués: "
            + (
                ", ".join(
                    escape(str(g.get("goal_id", "?")))
                    for g in goals_insight.get("stalled", [])
                )
                or "-"
            ),
        ),
    ]

    insight_html = "".join(
        "<div class='card'>"
        + f"<h3>{title}</h3>"
        + "".join(f"<p>{escape(str(line))}</p>" for line in lines if line)
        + "</div>"
        for title, *lines in insight_cards
    )

    chart_json = json.dumps(chart_points)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>AGI Dashboard</title>
{auto_refresh_meta}
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f7f7fb;color:#1a1a1a}}
h1{{margin-top:0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:24px 0}}
.card{{background:white;border-radius:12px;padding:16px;box-shadow:0 6px 18px rgba(0,0,0,0.06)}}
table{{border-collapse:collapse;width:100%;margin-top:16px;background:white;border-radius:12px;overflow:hidden;box-shadow:0 3px 12px rgba(0,0,0,0.05)}}
th,td{{border-bottom:1px solid #eee;padding:10px 12px;font-size:14px;text-align:left}}
th{{background:#fafafa;font-weight:600}}
tr:hover{{background:#f0f4ff}}
code{{white-space:pre-wrap;font-size:12px}}
.section{{margin-top:40px}}
.metacog{{background:white;border-radius:12px;padding:16px;box-shadow:0 3px 12px rgba(0,0,0,0.05)}}
.metacog ul{{padding-left:20px}}
.metacog li{{margin-bottom:8px}}
.filters{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.filters input{{padding:8px 10px;border-radius:8px;border:1px solid #ccc}}
</style>
</head>
<body>
<h1>AGI Dashboard</h1>
<p>Généré: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
{filter_info}
<div class="cards">{insight_html}</div>

<div class="section">
  <h2>Raisonnements récents</h2>
  <div class="filters">
    <label>Filtrer&nbsp;: <input id="reasoning-filter" placeholder="But, ID..." /></label>
  </div>
  <canvas id="reasoning-chart" height="160"></canvas>
  <table id="reasoning-table"><thead><tr><th>But</th><th>Confiance</th><th>Durée</th><th>Solution</th></tr></thead>
  <tbody>{''.join(reasoning_rows)}</tbody></table>
</div>

<div class="section">
  <h2>Expériences</h2>
  <table><thead><tr><th>Métrique</th><th>Succès</th><th>Observé</th><th>Objectif / Plan</th></tr></thead>
  <tbody>{''.join(experiment_rows)}</tbody></table>
</div>

<div class="section">
  <h2>Objectifs actifs</h2>
  <table><thead><tr><th>ID</th><th>Description</th><th>Progress</th><th>Compétence</th><th>Valeur</th></tr></thead>
  <tbody>{''.join(goals_rows)}</tbody></table>
</div>

<div class="section metacog">
  <h2>Métacognition</h2>
  <ul>{''.join(metacog_rows)}</ul>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const datapoints = {chart_json};
const ctx = document.getElementById('reasoning-chart').getContext('2d');
const chart = new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: datapoints.map(p => p.label),
        datasets: [
            {{label: 'Durée (s)', data: datapoints.map(p => p.duration), borderColor: '#2563eb', fill: false}},
            {{label: 'Confiance', data: datapoints.map(p => p.confidence), borderColor: '#f97316', fill: false}}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{beginAtZero: true}}
        }}
    }}
}});

const filterInput = document.getElementById('reasoning-filter');
if (filterInput) {{
  filterInput.addEventListener('input', () => {{
    const query = filterInput.value.toLowerCase();
    document.querySelectorAll('#reasoning-table tbody tr').forEach(row => {{
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    }});
  }});
}}
</script>
</body></html>"""
    return html


def export_html(
    path: str = "logs/dashboard.html",
    *,
    goal_filter: Optional[str] = None,
    since: Optional[datetime] = None,
    auto_refresh: int = 0,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = load_logs(since)
    data = apply_goal_filter_to_data(data, goal_filter)
    insights = compute_insights(data)
    html = build_html(
        data,
        insights,
        auto_refresh=auto_refresh,
        goal_filter=goal_filter,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML exporté → {path}")


def serve_dashboard(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    goal_filter: Optional[str] = None,
    since: Optional[datetime] = None,
    auto_refresh: int = 5,
):
    context = {
        "goal_filter": goal_filter,
        "since": since,
        "auto_refresh": auto_refresh,
    }

    def handler_factory():
        class DashboardHandler(BaseHTTPRequestHandler):
            def _write(self, body: str, *, status: int = 200, content_type: str = "text/html"):
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 (shadow-builtin)
                return  # Silence default logging

            def do_GET(self):  # type: ignore[override]
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                goal_filter_override = query.get("goal", [context["goal_filter"]])[0]
                since_override = context["since"]
                if "since" in query:
                    try:
                        minutes = float(query["since"][0])
                        since_override = datetime.now() - timedelta(minutes=minutes)
                    except (ValueError, TypeError):
                        pass

                data = load_logs(since_override)
                data = apply_goal_filter_to_data(data, goal_filter_override)
                insights = compute_insights(data)

                if parsed.path in ("/dashboard.json", "/data"):
                    payload = json.dumps({"data": data, "insights": insights})
                    self._write(payload, content_type="application/json")
                    return

                if parsed.path in ("/", "/index.html"):
                    html = build_html(
                        data,
                        insights,
                        auto_refresh=context["auto_refresh"],
                        goal_filter=goal_filter_override,
                    )
                    self._write(html)
                    return

                self._write("Not found", status=404)

        return DashboardHandler

    httpd = ThreadingHTTPServer((host, port), handler_factory())
    print(f"Dashboard disponible sur http://{host}:{port}")
    print("Ctrl+C pour arrêter le serveur.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Arrêt du serveur dashboard.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dashboard AGI évolutif")
    ap.add_argument("--once", action="store_true", help="Affiche un snapshot console et quitte (par défaut si aucune autre action)")
    ap.add_argument("--watch", action="store_true", help="Actualise continuellement la console")
    ap.add_argument("--interval", type=float, default=5.0, help="Intervalle (s) entre rafraîchissements en mode watch")
    ap.add_argument("--reasoning-limit", type=int, default=5, help="Nombre de raisonnements à afficher")
    ap.add_argument("--experiments-limit", type=int, default=10, help="Nombre d'expériences à afficher")
    ap.add_argument("--goals-limit", type=int, default=5, help="Nombre d'objectifs actifs affichés")
    ap.add_argument("--goal", dest="goal_filter", help="Filtre par objectif (texte ou ID)")
    ap.add_argument("--since-minutes", type=float, help="Limite aux événements des X dernières minutes")
    ap.add_argument("--html", action="store_true", help="Exporte un HTML statique amélioré")
    ap.add_argument("--html-path", default="logs/dashboard.html", help="Chemin du fichier HTML exporté")
    ap.add_argument("--auto-refresh", type=int, default=None, help="Intervalle (s) de rafraîchissement automatique (défaut: 0 pour HTML, 5s pour --serve)")
    ap.add_argument("--serve", action="store_true", help="Lance un mini-serveur web interactif")
    ap.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute du serveur web")
    ap.add_argument("--port", type=int, default=8765, help="Port d'écoute du serveur web")
    args = ap.parse_args()

    since_dt: Optional[datetime] = None
    if args.since_minutes:
        since_dt = datetime.now() - timedelta(minutes=float(args.since_minutes))

    console_kwargs = {
        "reasoning_limit": args.reasoning_limit,
        "experiments_limit": args.experiments_limit,
        "goals_limit": args.goals_limit,
        "goal_filter": args.goal_filter,
        "since": since_dt,
    }

    if args.html:
        export_html(
            path=args.html_path,
            goal_filter=args.goal_filter,
            since=since_dt,
            auto_refresh=args.auto_refresh or 0,
        )

    run_default = not (args.watch or args.html or args.serve or args.once)

    if args.watch:
        console_watch(interval=args.interval, **console_kwargs)
    elif args.once or run_default:
        console_once(**console_kwargs)

    if args.serve:
        refresh = args.auto_refresh if args.auto_refresh is not None else 5
        serve_dashboard(
            host=args.host,
            port=args.port,
            goal_filter=args.goal_filter,
            since=since_dt,
            auto_refresh=refresh,
        )
