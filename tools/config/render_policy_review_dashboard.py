#!/usr/bin/env python3
"""Render a standalone HTML dashboard for Citlali config policy review."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import classify_lowlevel_config


DEFAULT_OUTPUT = "/private/tmp/citlali_config_policy_review/index.html"
CLASSIFICATIONS = ("user-facing", "expert", "hidden/internal", "deprecated")


def resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_counts_template() -> dict[str, int]:
    return {classification: 0 for classification in CLASSIFICATIONS}


def build_report_from_args(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path.cwd()
    if args.report:
        return load_json(resolve_path(args.report, base_dir))

    rules = classify_lowlevel_config.load_rules(resolve_path(args.rules, base_dir))
    configs = [
        classify_lowlevel_config.parse_config_arg(value, base_dir)
        for value in args.config
    ]
    if args.cases:
        configs.extend(
            classify_lowlevel_config.load_case_configs(
                resolve_path(args.cases, base_dir)
            )
        )
    return classify_lowlevel_config.build_report(configs, rules, args.require_all)


def load_rules_for_dashboard(path: Path) -> list[dict[str, str]]:
    loaded = classify_lowlevel_config.load_rules(path)
    return loaded["rules"] + [loaded["fallback"]]


def confidence_for_rule(
    rule: dict[str, str],
    rows: list[dict[str, Any]],
    unique_path_count: int,
) -> tuple[int, str]:
    """Return reviewer confidence in the current rule classification."""
    rule_id = rule["id"]
    pattern = rule["pattern"]
    classification = rule["classification"]
    owner = rule["owner"]
    reason = rule["reason"].lower()

    if rule_id == "fallback":
        return (
            5,
            "Fallback behavior is intentionally conservative, but any observed fallback match means the policy needs explicit review.",
        )

    if not rows:
        return (
            6,
            "This rule is not exercised by the current representative baselines, so the classification is policy intent rather than observed evidence.",
        )

    concerns: list[str] = []
    confidence = 10

    if "*" in pattern and unique_path_count > 40:
        confidence = min(confidence, 6)
        concerns.append("broad wildcard rule covers many heterogeneous paths")
    elif "*" in pattern and unique_path_count > 10:
        confidence = min(confidence, 7)
        concerns.append("wildcard rule covers a moderately broad path family")
    elif "*" in pattern:
        confidence = min(confidence, 8)
        concerns.append("wildcard rule should be checked against observed examples")

    if classification == "expert" and unique_path_count > 80:
        confidence = min(confidence, 6)
        concerns.append("large expert bucket may hide paths that deserve mode-specific promotion")
    elif classification == "expert" and unique_path_count > 25:
        confidence = min(confidence, 7)
        concerns.append("expert bucket spans enough paths to merit spot review")

    if classification == "user-facing" and (
        "advanced" in reason
        or "only for" in reason
        or "diagnostic" in reason
        or "sidecar" in reason
        or "tuning" in reason
    ):
        confidence = min(confidence, 8)
        concerns.append("user-facing classification depends on workflow/profile context")

    if classification == "hidden/internal":
        if owner in {"tolteca", "profile"} or pattern.startswith("inputs"):
            confidence = min(confidence, 10)
        else:
            confidence = min(confidence, 8)
            concerns.append("internal ownership should be confirmed with the reducer boundary")

    if classification == "deprecated":
        if "legacy" in reason or "ignored" in reason or "historical" in reason:
            confidence = min(confidence, 9)
        else:
            confidence = min(confidence, 8)
            concerns.append("deprecated classification needs an explicit replacement or removal path")

    if unique_path_count == 1 and "*" not in pattern and classification != "deprecated":
        confidence = max(confidence, 9)

    if concerns:
        return confidence, "; ".join(concerns) + "."
    return confidence, "High confidence: narrow observed rule with a direct policy rationale."


def summarize_rules(report: dict[str, Any], rules: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows_by_rule: dict[str, list[dict[str, Any]]] = {}
    for row in report["rows"]:
        rows_by_rule.setdefault(row["rule_id"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for rule in rules:
        rows = rows_by_rule.get(rule["id"], [])
        unique_paths = sorted({row["normalized_path"] for row in rows})
        intent_counts: dict[str, int] = {}
        top_counts: dict[str, int] = {}
        for row in rows:
            intent = row.get("intent") or row.get("config_label") or "unknown"
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            top_counts[row["top"]] = top_counts.get(row["top"], 0) + 1

        examples = []
        seen_examples: set[tuple[str, str]] = set()
        for row in sorted(rows, key=lambda item: (item["normalized_path"], item["config_label"])):
            key = (row["normalized_path"], row["config_label"])
            if key in seen_examples:
                continue
            seen_examples.add(key)
            examples.append(
                {
                    "path": row["normalized_path"],
                    "config": row["config_label"],
                    "intent": row.get("intent", ""),
                    "value": row["value_preview"],
                }
            )
            if len(examples) >= 30:
                break

        confidence, confidence_reason = confidence_for_rule(
            rule,
            rows,
            len(unique_paths),
        )
        summaries.append(
            {
                "id": rule["id"],
                "pattern": rule["pattern"],
                "classification": rule["classification"],
                "confidence": confidence,
                "confidenceReason": confidence_reason,
                "owner": rule["owner"],
                "reason": rule["reason"],
                "observed": len(rows),
                "uniquePaths": len(unique_paths),
                "intents": dict(sorted(intent_counts.items())),
                "tops": dict(sorted(top_counts.items())),
                "paths": unique_paths[:80],
                "examples": examples,
            }
        )
    return summaries


def summarize_paths(report: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in report["rows"]:
        grouped.setdefault(row["normalized_path"], []).append(row)

    paths: list[dict[str, Any]] = []
    for path, rows in sorted(grouped.items()):
        first = rows[0]
        intents = sorted({row.get("intent") or row.get("config_label", "") for row in rows})
        examples = []
        seen_configs: set[str] = set()
        for row in sorted(rows, key=lambda item: item["config_label"]):
            if row["config_label"] in seen_configs:
                continue
            seen_configs.add(row["config_label"])
            examples.append(
                {
                    "config": row["config_label"],
                    "intent": row.get("intent", ""),
                    "value": row["value_preview"],
                }
            )
        paths.append(
            {
                "path": path,
                "top": first["top"],
                "classification": first["classification"],
                "ruleId": first["rule_id"],
                "rulePattern": first["rule_pattern"],
                "owner": first["owner"],
                "reason": first["reason"],
                "intents": intents,
                "examples": examples,
            }
        )
    return paths


def summarize_modes(report: dict[str, Any]) -> list[dict[str, Any]]:
    modes = []
    for config in report["configs"]:
        label = config["label"]
        rows = [row for row in report["rows"] if row["config_label"] == label]
        counts = class_counts_template()
        top_counts: dict[str, dict[str, int]] = {}
        user_paths: list[str] = []
        deprecated_paths: list[str] = []
        for row in rows:
            classification = row["classification"]
            counts[classification] += 1
            top_counts.setdefault(row["top"], class_counts_template())
            top_counts[row["top"]][classification] += 1
            if classification == "user-facing":
                user_paths.append(row["normalized_path"])
            elif classification == "deprecated":
                deprecated_paths.append(row["normalized_path"])

        modes.append(
            {
                "label": label,
                "intent": config.get("intent", ""),
                "path": config["path"],
                "leafCount": config["leaf_count"],
                "counts": counts,
                "topCounts": dict(sorted(top_counts.items())),
                "userFacingPaths": sorted(set(user_paths)),
                "deprecatedPaths": sorted(set(deprecated_paths)),
            }
        )
    return modes


def dashboard_payload(report: dict[str, Any], rules_path: Path) -> dict[str, Any]:
    rules = load_rules_for_dashboard(rules_path)
    return {
        "generatedBy": "tools/config/render_policy_review_dashboard.py",
        "schema": "citlali-config-policy-review-dashboard-v1",
        "summary": report["summary"],
        "rulesFile": str(rules_path),
        "configs": report["configs"],
        "rules": summarize_rules(report, rules),
        "paths": summarize_paths(report),
        "modes": summarize_modes(report),
    }


def json_for_script(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")


def render_html(payload: dict[str, Any]) -> str:
    data = json_for_script(payload)
    title = "Citlali Config Policy Review"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1b1f24;
      --muted: #5f6b7a;
      --line: #d9dee7;
      --accent: #1967d2;
      --user: #176b4d;
      --expert: #7a4d00;
      --hidden: #4b5563;
      --deprecated: #a33d2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 20px 28px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 20;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .muted {{ color: var(--muted); }}
    .layout {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    aside {{
      position: sticky;
      top: 92px;
      align-self: start;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    main {{
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-width: 0;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .stat .value {{ display: block; font-size: 22px; font-weight: 700; }}
    label {{ display: block; font-weight: 600; margin: 12px 0 5px; }}
    input[type="search"], select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    textarea {{ min-height: 70px; resize: vertical; }}
    input[type="radio"] {{ margin: 0; }}
    .tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }}
    button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 700; }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid currentColor;
      white-space: nowrap;
    }}
    .badge.user-facing {{ color: var(--user); }}
    .badge.expert {{ color: var(--expert); }}
    .badge.hidden-internal {{ color: var(--hidden); }}
    .badge.deprecated {{ color: var(--deprecated); }}
    .confidence {{
      display: inline-flex;
      align-items: center;
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 12px;
      font-weight: 700;
      background: #eef3fb;
      color: #174ea6;
      border: 1px solid #b9c9e6;
      white-space: nowrap;
    }}
    .confidence.low {{
      background: #fff4e5;
      color: #8a4b00;
      border-color: #e1b978;
      cursor: help;
    }}
    .card-list {{ display: grid; gap: 10px; }}
    .rule-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .rule-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }}
    .rule-title {{ font-weight: 700; overflow-wrap: anywhere; }}
    .rule-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0;
      color: var(--muted);
      font-size: 12px;
    }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .examples {{
      display: grid;
      gap: 6px;
      margin-top: 8px;
    }}
    .example {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
      overflow-wrap: anywhere;
    }}
    .review {{
      display: grid;
      grid-template-columns: minmax(160px, 220px) minmax(160px, 220px) minmax(240px, 1fr);
      gap: 10px;
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    .radio-group {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }}
    .radio-choice {{
      display: flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px;
      background: #fbfcfe;
      font-weight: 500;
      margin: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{ color: var(--muted); font-size: 12px; }}
    .mode-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    .path-list {{
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
    }}
    .path-list div {{ margin-bottom: 5px; overflow-wrap: anywhere; }}
    .empty {{ padding: 20px; color: var(--muted); text-align: center; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      .review {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Citlali Config Policy Review</h1>
    <div class="muted">Interactive review of low-level key classification policy. Review changes are stored in this browser until exported.</div>
    <div class="tabs" role="tablist">
      <button class="active" data-view="rules">Rules</button>
      <button data-view="paths">Paths</button>
      <button data-view="modes">Reduction Modes</button>
      <button data-view="queue">Review Queue</button>
    </div>
  </header>
  <div class="layout">
    <aside>
      <h2>Filters</h2>
      <label for="search">Search</label>
      <input id="search" type="search" placeholder="rule id, path, reason">
      <label for="classification">Classification</label>
      <select id="classification">
        <option value="">All</option>
        <option>user-facing</option>
        <option>expert</option>
        <option>hidden/internal</option>
        <option>deprecated</option>
      </select>
      <label for="intent">Reduction mode</label>
      <select id="intent"></select>
      <label for="top">Top-level group</label>
      <select id="top"></select>
      <label for="observed">Observed rules</label>
      <select id="observed">
        <option value="observed">Observed only</option>
        <option value="all">Include unobserved</option>
      </select>
      <label for="sort">Rule order</label>
      <select id="sort">
        <option value="policy">Policy order</option>
        <option value="confidence-asc">Confidence: low to high</option>
        <option value="confidence-desc">Confidence: high to low</option>
      </select>
      <div class="toolbar">
        <button id="clearFilters">Clear Filters</button>
        <button id="exportReview" class="primary">Export Review JSON</button>
      </div>
      <p class="muted">Workflow: filter to a mode or top-level group, review rule cards, set status/proposed class, add notes, then export review JSON.</p>
    </aside>
    <main>
      <section class="panel" id="summary"></section>
      <section class="panel" id="content"></section>
    </main>
  </div>
  <script id="dashboard-data" type="application/json">{data}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    const CLASSES = ['user-facing', 'expert', 'hidden/internal', 'deprecated'];
    const STORE_KEY = 'citlali-config-policy-review-v1';
    let view = 'rules';
    let review = loadReview();

    const controls = {{
      search: document.getElementById('search'),
      classification: document.getElementById('classification'),
      intent: document.getElementById('intent'),
      top: document.getElementById('top'),
      observed: document.getElementById('observed'),
      sort: document.getElementById('sort')
    }};

    function cssClass(name) {{
      return String(name).replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase();
    }}
    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function badge(cls) {{
      return `<span class="badge ${{cssClass(cls)}}">${{escapeHtml(cls)}}</span>`;
    }}
    function confidenceBadge(rule) {{
      const low = rule.confidence <= 8;
      const title = low ? ` title="${{escapeHtml(rule.confidenceReason || '')}}"` : '';
      return `<span class="confidence${{low ? ' low' : ''}}"${{title}}>confidence ${{rule.confidence}}/10</span>`;
    }}
    function loadReview() {{
      try {{
        return JSON.parse(localStorage.getItem(STORE_KEY) || '{{}}');
      }} catch (error) {{
        return {{}};
      }}
    }}
    function saveReview() {{
      localStorage.setItem(STORE_KEY, JSON.stringify(review));
    }}
    function ruleReview(id) {{
      if (!review[id]) {{
        review[id] = {{status: 'unreviewed', proposed: '', note: ''}};
      }}
      return review[id];
    }}
    function ensureAllRuleReviews() {{
      DATA.rules.forEach(rule => ruleReview(rule.id));
    }}

    function initControls() {{
      const intents = Array.from(new Set(DATA.configs.map(c => c.intent || c.label))).sort();
      controls.intent.innerHTML = '<option value="">All</option>' + intents.map(v => `<option>${{escapeHtml(v)}}</option>`).join('');
      const tops = Array.from(new Set(DATA.paths.map(p => p.top))).sort();
      controls.top.innerHTML = '<option value="">All</option>' + tops.map(v => `<option>${{escapeHtml(v)}}</option>`).join('');
      Object.values(controls).forEach(control => control.addEventListener('input', render));
      document.getElementById('clearFilters').addEventListener('click', () => {{
        controls.search.value = '';
        controls.classification.value = '';
        controls.intent.value = '';
        controls.top.value = '';
        controls.observed.value = 'observed';
        controls.sort.value = 'policy';
        render();
      }});
      document.getElementById('exportReview').addEventListener('click', exportReview);
      document.querySelectorAll('[data-view]').forEach(button => {{
        button.addEventListener('click', () => {{
          view = button.dataset.view;
          document.querySelectorAll('[data-view]').forEach(item => item.classList.toggle('active', item.dataset.view === view));
          render();
        }});
      }});
    }}

    function itemMatchesText(item, text) {{
      if (!text) return true;
      return JSON.stringify(item).toLowerCase().includes(text);
    }}
    function activeFilters() {{
      return {{
        text: controls.search.value.trim().toLowerCase(),
        classification: controls.classification.value,
        intent: controls.intent.value,
        top: controls.top.value,
        observed: controls.observed.value,
        sort: controls.sort.value
      }};
    }}
    function filteredRules() {{
      const filters = activeFilters();
      const rules = DATA.rules.filter(rule => {{
        if (filters.observed === 'observed' && rule.observed === 0) return false;
        if (filters.classification && rule.classification !== filters.classification) return false;
        if (filters.intent && !Object.prototype.hasOwnProperty.call(rule.intents, filters.intent)) return false;
        if (filters.top && !Object.prototype.hasOwnProperty.call(rule.tops, filters.top)) return false;
        return itemMatchesText(rule, filters.text);
      }});
      if (filters.sort === 'confidence-asc') {{
        return rules.sort((a, b) => a.confidence - b.confidence || a.id.localeCompare(b.id));
      }}
      if (filters.sort === 'confidence-desc') {{
        return rules.sort((a, b) => b.confidence - a.confidence || a.id.localeCompare(b.id));
      }}
      return rules;
    }}
    function filteredPaths() {{
      const filters = activeFilters();
      return DATA.paths.filter(path => {{
        if (filters.classification && path.classification !== filters.classification) return false;
        if (filters.intent && !path.intents.includes(filters.intent)) return false;
        if (filters.top && path.top !== filters.top) return false;
        return itemMatchesText(path, filters.text);
      }});
    }}
    function renderSummary() {{
      const s = DATA.summary;
      const reviewed = Object.values(review).filter(item => item.status && item.status !== 'unreviewed').length;
      const rules = filteredRules().length;
      const paths = filteredPaths().length;
      document.getElementById('summary').innerHTML = `
        <h2>Summary</h2>
        <div class="muted">Rules file: <code>${{escapeHtml(DATA.rulesFile)}}</code></div>
        <div class="stats">
          <div class="stat"><span class="value">${{s.unique_path_count}}</span><span>Unique paths</span></div>
          <div class="stat"><span class="value">${{s.leaf_count}}</span><span>Leaf occurrences</span></div>
          <div class="stat"><span class="value">${{rules}}</span><span>Visible rules</span></div>
          <div class="stat"><span class="value">${{reviewed}}</span><span>Reviewed rules</span></div>
        </div>
        <div class="stats">
          ${{CLASSES.map(cls => `<div class="stat"><span class="value">${{s.unique_path_count_by_classification[cls] || 0}}</span><span>${{badge(cls)}} unique</span></div>`).join('')}}
        </div>
        <div class="muted" style="margin-top:10px;">Visible paths with current filters: ${{paths}}</div>
      `;
    }}
    function render() {{
      renderSummary();
      if (view === 'rules') renderRules();
      else if (view === 'paths') renderPaths();
      else if (view === 'modes') renderModes();
      else renderQueue();
    }}
    function renderRules() {{
      const rules = filteredRules();
      if (!rules.length) {{
        document.getElementById('content').innerHTML = '<div class="empty">No rules match the current filters.</div>';
        return;
      }}
      document.getElementById('content').innerHTML = `
        <h2>Rules</h2>
        <div class="card-list">
          ${{rules.map(renderRuleCard).join('')}}
        </div>
      `;
      bindReviewInputs();
    }}
    function renderRuleCard(rule) {{
      const state = ruleReview(rule.id);
      const intentText = Object.entries(rule.intents).map(([k, v]) => `${{escapeHtml(k)}}: ${{v}}`).join(', ') || 'not observed';
      const topText = Object.entries(rule.tops).map(([k, v]) => `${{escapeHtml(k)}}: ${{v}}`).join(', ') || 'not observed';
      return `
        <article class="rule-card" data-rule-id="${{escapeHtml(rule.id)}}">
          <div class="rule-head">
            <div>
              <div class="rule-title"><code>${{escapeHtml(rule.id)}}</code></div>
              <div><code>${{escapeHtml(rule.pattern)}}</code></div>
            </div>
            <div>${{badge(rule.classification)}} ${{confidenceBadge(rule)}}</div>
          </div>
          <div class="rule-meta">
            <span>owner: <code>${{escapeHtml(rule.owner || 'unset')}}</code></span>
            <span>observed leaves: ${{rule.observed}}</span>
            <span>unique paths: ${{rule.uniquePaths}}</span>
            <span>confidence: ${{rule.confidence}}/10</span>
            <span>modes: ${{intentText}}</span>
            <span>top groups: ${{topText}}</span>
          </div>
          <p>${{escapeHtml(rule.reason || '')}}</p>
          <details>
            <summary>Observed paths and example values</summary>
            <div class="examples">
              ${{rule.examples.length ? rule.examples.map(ex => `
                <div class="example">
                  <div><code>${{escapeHtml(ex.path)}}</code></div>
                  <div class="muted">${{escapeHtml(ex.config)}}${{ex.intent ? ' / ' + escapeHtml(ex.intent) : ''}} = <code>${{escapeHtml(ex.value)}}</code></div>
                </div>`).join('') : '<div class="muted">No observed paths for this rule.</div>'}}
            </div>
          </details>
          <div class="review">
            <div>
              <label>Status</label>
              <div class="radio-group">
                ${{['unreviewed','accept','change','needs-discussion'].map(v => `
                  <label class="radio-choice">
                    <input type="radio" name="status-${{escapeHtml(rule.id)}}" value="${{v}}" data-review-field="status" data-rule-id="${{escapeHtml(rule.id)}}" ${{state.status === v ? 'checked' : ''}}>
                    <span>${{v}}</span>
                  </label>`).join('')}}
              </div>
            </div>
            <div>
              <label>Proposed class</label>
              <select data-review-field="proposed" data-rule-id="${{escapeHtml(rule.id)}}">
                <option value="">Current: ${{escapeHtml(rule.classification)}}</option>
                ${{CLASSES.map(v => `<option value="${{v}}" ${{state.proposed === v ? 'selected' : ''}}>${{v}}</option>`).join('')}}
              </select>
            </div>
            <div>
              <label>Review note</label>
              <textarea data-review-field="note" data-rule-id="${{escapeHtml(rule.id)}}" placeholder="What should change, or why this is correct?">${{escapeHtml(state.note || '')}}</textarea>
            </div>
          </div>
        </article>
      `;
    }}
    function bindReviewInputs() {{
      document.querySelectorAll('[data-review-field]').forEach(input => {{
        input.addEventListener('input', event => {{
          const id = event.target.dataset.ruleId;
          const field = event.target.dataset.reviewField;
          ruleReview(id)[field] = event.target.value;
          saveReview();
          renderSummary();
        }});
      }});
    }}
    function renderPaths() {{
      const paths = filteredPaths();
      document.getElementById('content').innerHTML = `
        <h2>Paths</h2>
        <table>
          <thead>
            <tr><th style="width:34%;">Path</th><th>Class</th><th>Rule</th><th>Modes</th><th>Examples</th></tr>
          </thead>
          <tbody>
            ${{paths.slice(0, 600).map(path => `
              <tr>
                <td><code>${{escapeHtml(path.path)}}</code><div class="muted">${{escapeHtml(path.top)}}</div></td>
                <td>${{badge(path.classification)}}</td>
                <td><code>${{escapeHtml(path.ruleId)}}</code><div class="muted">${{escapeHtml(path.reason)}}</div></td>
                <td>${{path.intents.map(escapeHtml).join(', ')}}</td>
                <td>${{path.examples.map(ex => `<div><code>${{escapeHtml(ex.config)}}</code>: <code>${{escapeHtml(ex.value)}}</code></div>`).join('')}}</td>
              </tr>`).join('')}}
          </tbody>
        </table>
        ${{paths.length > 600 ? `<p class="muted">Showing first 600 of ${{paths.length}} matching paths. Narrow filters to see more detail.</p>` : ''}}
      `;
    }}
    function renderModes() {{
      document.getElementById('content').innerHTML = `
        <h2>Reduction Mode Assumptions</h2>
        <div class="mode-grid">
          ${{DATA.modes.map(mode => `
            <article class="rule-card">
              <h3>${{escapeHtml(mode.label)}}${{mode.intent ? ' / ' + escapeHtml(mode.intent) : ''}}</h3>
              <div class="muted"><code>${{escapeHtml(mode.path)}}</code></div>
              <div class="stats">
                ${{CLASSES.map(cls => `<div class="stat"><span class="value">${{mode.counts[cls] || 0}}</span><span>${{badge(cls)}}</span></div>`).join('')}}
              </div>
              <details open>
                <summary>User-facing paths observed (${{mode.userFacingPaths.length}})</summary>
                <div class="path-list">${{mode.userFacingPaths.map(path => `<div><code>${{escapeHtml(path)}}</code></div>`).join('')}}</div>
              </details>
              <details>
                <summary>Deprecated paths observed (${{mode.deprecatedPaths.length}})</summary>
                <div class="path-list">${{mode.deprecatedPaths.map(path => `<div><code>${{escapeHtml(path)}}</code></div>`).join('') || '<div class="muted">none</div>'}}</div>
              </details>
            </article>
          `).join('')}}
        </div>
      `;
    }}
    function renderQueue() {{
      const entries = Object.entries(review)
        .filter(([, state]) => state.status && state.status !== 'unreviewed')
        .sort(([a], [b]) => a.localeCompare(b));
      document.getElementById('content').innerHTML = `
        <h2>Review Queue</h2>
        ${{entries.length ? `
          <table>
            <thead><tr><th>Rule</th><th>Status</th><th>Proposed</th><th>Note</th></tr></thead>
            <tbody>
              ${{entries.map(([id, state]) => `<tr><td><code>${{escapeHtml(id)}}</code></td><td>${{escapeHtml(state.status)}}</td><td>${{escapeHtml(state.proposed || '')}}</td><td>${{escapeHtml(state.note || '')}}</td></tr>`).join('')}}
            </tbody>
          </table>` : '<div class="empty">No review decisions recorded yet.</div>'}}
      `;
    }}
    function exportReview() {{
      ensureAllRuleReviews();
      saveReview();
      const payload = {{
        schema: 'citlali-config-policy-review-export-v1',
        exportedAt: new Date().toISOString(),
        rulesFile: DATA.rulesFile,
        decisions: review
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2) + '\\n'], {{type: 'application/json'}});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'citlali_config_policy_review.json';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }}

    ensureAllRuleReviews();
    saveReview();
    initControls();
    render();
  </script>
</body>
</html>
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules",
        default=str(Path(__file__).with_name("config_key_classification.yaml")),
        help="Classification rules YAML file.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Existing JSON report from classify_lowlevel_config.py. If omitted, a report is built from --cases/--config.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Low-level or TolTECA YAML config to classify. May be repeated.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Compact compatibility cases file; unique base_config paths are classified.",
    )
    parser.add_argument("--require-all", action="store_true", help="Fail if any config path is missing.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML file.")
    args = parser.parse_args(argv)
    if not args.report and not args.config and not args.cases:
        parser.error("pass --report, --cases, or at least one --config")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_dir = Path.cwd()
    try:
        rules_path = resolve_path(args.rules, base_dir)
        report = build_report_from_args(args)
        payload = dashboard_payload(report, rules_path)
        output_path = resolve_path(args.output, base_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_html(payload), encoding="utf-8")
    except (OSError, yaml.YAMLError, json.JSONDecodeError, classify_lowlevel_config.ClassificationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {output_path}")
    print(
        "dashboard: "
        f"rules={len(payload['rules'])} paths={len(payload['paths'])} modes={len(payload['modes'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
