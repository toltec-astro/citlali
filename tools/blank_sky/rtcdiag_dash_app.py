#!/usr/bin/env python3
"""Interactive engineering dashboard for rtcdiag survey products."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dash import Dash, Input, Output, dash_table, dcc, html
except ModuleNotFoundError as exc:
    raise SystemExit(
        "dash is not installed in ~/toltec. Run '~/toltec/bin/pip install --upgrade dash'."
    ) from exc

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from .rtcdiag_data import load_reduction_tables
except ImportError:
    from rtcdiag_data import load_reduction_tables


NUMERIC_ROUND = {
    "max_step_det_frac": 4,
    "max_step_alignment_frac": 4,
    "max_cm_lowmid": 3,
    "mean_cm_lowmid": 3,
    "max_row_severity": 3,
    "mean_row_severity": 3,
    "max_network_impulsive_score": 3,
    "max_network_impulsive_det_frac": 4,
    "max_network_impulsive_alignment_frac": 4,
    "max_impulsive_event_score": 3,
    "top_slot_event_score": 3,
    "network_impulsive_score_max": 3,
    "network_impulsive_det_frac": 4,
    "network_impulsive_alignment_frac": 4,
    "step_det_frac": 4,
    "step_alignment_frac": 4,
    "cm_lowmid": 3,
    "row_severity": 3,
    "event_score": 3,
    "peak_abs_z": 3,
    "peak_delta_abs_z": 3,
    "impulsive_mask_cluster_peak_score": 3,
    "impulsive_mask_override_score": 3,
    "impulsive_mask_proposed_flagged_fraction": 4,
}

HELP_BOX_STYLE = {
    "backgroundColor": "#f7f6f2",
    "border": "1px solid #d8d4c8",
    "padding": "12px 14px",
    "marginBottom": "14px",
    "fontSize": "14px",
    "lineHeight": "1.5",
}

PAGE_STYLE = {
    "padding": "20px",
    "maxWidth": "1640px",
    "background": "linear-gradient(180deg, #fbfaf6 0%, #f1eee6 100%)",
    "color": "#1e1c17",
    "fontFamily": "Georgia, 'Iowan Old Style', 'Palatino Linotype', serif",
}

PANEL_STYLE = {
    "backgroundColor": "#fffdf8",
    "border": "1px solid #ddd7c9",
    "borderRadius": "14px",
    "padding": "14px 16px",
    "boxShadow": "0 10px 28px rgba(70, 56, 24, 0.07)",
    "marginBottom": "16px",
}

CARD_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
    "gap": "12px",
    "marginBottom": "16px",
}

CARD_STYLE = {
    "background": "linear-gradient(135deg, #fffef9 0%, #f3efe2 100%)",
    "border": "1px solid #d9d1bc",
    "borderRadius": "14px",
    "padding": "14px",
    "minHeight": "96px",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "space-between",
}

GRAPH_CONFIG = {"displaylogo": False, "responsive": True}

MASK_COLORS = {
    "both masks": "#8f2d56",
    "impulsive": "#d76a03",
    "step": "#355070",
    "candidate only": "#7d6b2f",
    "diagnostic only": "#7a7a7a",
}

EVENT_KIND_COLORS = {
    "raw_like": "#1b6ca8",
    "delta_like": "#cf3f3f",
    "unknown": "#6b6b6b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redu-dir", required=True, help="Reduction directory, e.g. reduced/redu40")
    parser.add_argument("--array", default="a1100", choices=["a1100", "a1400", "a2000"])
    parser.add_argument("--networks", default="0,1,2,3,4,5")
    parser.add_argument("--obsnums", default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def rounded_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    view = df.loc[:, [col for col in columns if col in df.columns]].copy()
    for col, ndigits in NUMERIC_ROUND.items():
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").round(ndigits)
    return view.to_dict("records")


def table(
    columns: list[str],
    data: list[dict[str, object]],
    page_size: int = 12,
    table_id: str | None = None,
):
    kwargs = {
        "columns": [{"name": col, "id": col} for col in columns],
        "data": data,
        "page_size": page_size,
        "sort_action": "native",
        "filter_action": "native",
        "style_table": {"overflowX": "auto"},
        "style_cell": {
            "fontFamily": "Menlo, Monaco, Consolas, monospace",
            "fontSize": "12px",
            "textAlign": "left",
            "padding": "6px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        "style_header": {"fontWeight": "bold", "backgroundColor": "#f5f5f0"},
    }
    if table_id is not None:
        kwargs["id"] = table_id
    return dash_table.DataTable(**kwargs)


def section_help(title: str, body: str):
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "bold", "marginBottom": "4px"}),
            dcc.Markdown(body),
        ],
        style=HELP_BOX_STYLE,
    )


def safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def format_metric(value: object, ndigits: int = 2, suffix: str = "") -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{ndigits}f}{suffix}"


def format_count(value: object) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{int(round(number))}"


def format_fraction_pct(value: object, ndigits: int = 1) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100.0:.{ndigits}f}%"


def flagged_axis_limit(values: list[float]) -> float:
    finite = [float(v) for v in values if pd.notna(v)]
    if not finite:
        return 1.0
    max_value = max(finite)
    if max_value <= 1.0:
        return 1.0
    if max_value <= 2.0:
        return 2.5
    if max_value <= 3.0:
        return 4.0
    if max_value <= 5.0:
        return 6.0
    return min(12.0, max_value * 1.25)


def metric_card(title: str, value: str, note: str, accent: str = "#5b4b2a"):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "letterSpacing": "0.06em", "textTransform": "uppercase", "color": "#695f49"}),
            html.Div(value, style={"fontSize": "34px", "lineHeight": "1.0", "fontWeight": "bold", "margin": "10px 0 6px 0"}),
            html.Div(
                note,
                style={
                    "fontSize": "13px",
                    "color": "#5f5644",
                    "lineHeight": "1.35",
                    "overflowWrap": "anywhere",
                    "wordBreak": "break-word",
                },
            ),
        ],
        style={**CARD_STYLE, "borderTop": f"4px solid {accent}"},
    )


def mask_state_label(row: pd.Series) -> str:
    step = int(row.get("step_mask_applied", 0)) != 0
    imp = int(row.get("impulsive_mask_applied", 0)) != 0
    cand = int(row.get("impulsive_mask_candidate_available", 0)) != 0
    if step and imp:
        return "both masks"
    if imp:
        return "impulsive"
    if step:
        return "step"
    if cand:
        return "candidate only"
    return "diagnostic only"


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
        plot_bgcolor="#fffdf8",
        paper_bgcolor="#fffdf8",
    )
    return fig


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="#fffdf8",
        paper_bgcolor="#fffdf8",
        font={"family": "Georgia, serif", "color": "#1e1c17"},
    )
    return fig


def build_heatmap(scan_df: pd.DataFrame, obsnum: str) -> go.Figure:
    obs_df = scan_df.loc[scan_df["obsnum"] == obsnum].copy()
    if obs_df.empty:
        return empty_figure(f"No timechunk/network summaries for observation {obsnum}")

    obs_df["step_flagged_pct"] = pd.to_numeric(obs_df["step_mask_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    obs_df["imp_flagged_pct"] = pd.to_numeric(obs_df["impulsive_mask_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    obs_df["hover_label"] = obs_df.apply(
        lambda row: (
            f"timechunk={int(row['output_scan_index'])}"
            f"<br>nw={int(row['network'])}"
            f"<br>step flagged={row['step_flagged_pct']:.2f}%"
            f"<br>impulsive flagged={row['imp_flagged_pct']:.2f}%"
            f"<br>severity={row['row_severity']:.3f}"
            f"<br>step_det={row['step_det_frac']:.4f}"
            f"<br>imp_score={row['network_impulsive_score_max']:.3f}"
            f"<br>event={row['max_slot_event_score']:.3f}"
            f"<br>step_mask={int(row['step_mask_applied'])}"
            f"<br>imp_mask={int(row['impulsive_mask_applied'])}"
            f"<br>cand={int(row.get('impulsive_mask_candidate_available', 0))}"
            f"<br>local={int(row.get('impulsive_mask_local_trigger', 0))}"
            f"<br>cross={int(row.get('impulsive_mask_cross_network_trigger', 0))}"
            f"<br>override={int(row.get('impulsive_mask_high_score_override_trigger', 0))}"
            f"<br>cluster_nw={int(row.get('impulsive_mask_cluster_network_count', 0))}"
            f"<br>cluster_peak={row.get('impulsive_mask_cluster_peak_score', float('nan')):.3f}"
            f"<br>override_score={row.get('impulsive_mask_override_score', float('nan')):.3f}"
            f"<br>override_network_peak={int(row.get('impulsive_mask_override_uses_network_peak', 0))}"
        ),
        axis=1,
    )
    step_pivot = obs_df.pivot(index="network", columns="output_scan_index", values="step_flagged_pct").sort_index()
    imp_pivot = obs_df.pivot(index="network", columns="output_scan_index", values="imp_flagged_pct").sort_index()
    hover = obs_df.pivot(index="network", columns="output_scan_index", values="hover_label").sort_index()
    step_zmax = max(float(np.nanmax(step_pivot.values)) if np.isfinite(step_pivot.values).any() else 0.0, 10.0)
    imp_zmax = max(float(np.nanmax(imp_pivot.values)) if np.isfinite(imp_pivot.values).any() else 0.0, 5.0)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Step flagged fraction (%)", "Impulsive flagged fraction (%)"),
    )
    fig.add_trace(
        go.Heatmap(
            x=step_pivot.columns.tolist(),
            y=step_pivot.index.tolist(),
            z=step_pivot.values,
            text=hover.values,
            hovertemplate="%{text}<extra></extra>",
            colorscale="Blues",
            zmin=0.0,
            zmax=step_zmax,
            colorbar={"title": "step %", "len": 0.38, "y": 0.79},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=imp_pivot.columns.tolist(),
            y=imp_pivot.index.tolist(),
            z=imp_pivot.values,
            text=hover.values,
            hovertemplate="%{text}<extra></extra>",
            colorscale="Oranges",
            zmin=0.0,
            zmax=imp_zmax,
            colorbar={"title": "imp %", "len": 0.38, "y": 0.21},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"Observation {obsnum}: RTC flagged fraction by timechunk/network",
        margin={"l": 60, "r": 70, "t": 72, "b": 50},
        height=700,
    )
    fig.update_xaxes(title_text="timechunk", row=2, col=1)
    fig.update_yaxes(title_text="network", row=1, col=1)
    fig.update_yaxes(title_text="network", row=2, col=1)
    return style_figure(fig)


def build_network_trend(scan_df: pd.DataFrame, obsnum: str, network: int) -> go.Figure:
    df = scan_df.loc[(scan_df["obsnum"] == obsnum) & (scan_df["network"] == network)].copy()
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=("Step flagged fraction (%)", "Impulsive flagged fraction (%)", "Supporting diagnostics"),
    )
    if df.empty:
        fig.update_layout(title=f"No timechunk summaries for observation {obsnum} network {network}")
        return style_figure(fig)

    x = df["output_scan_index"]
    step_flagged_pct = pd.to_numeric(df["step_mask_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    imp_flagged_pct = pd.to_numeric(df["impulsive_mask_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    step_pct_max = max(float(step_flagged_pct.max()) if len(step_flagged_pct) else 0.0, 1.0) * 1.10
    imp_pct_max = max(float(imp_flagged_pct.max()) if len(imp_flagged_pct) else 0.0, 0.25) * 1.15
    score_max = max(
        float(pd.to_numeric(df["network_impulsive_score_max"], errors="coerce").max() or 0.0),
        float(pd.to_numeric(df["max_slot_event_score"], errors="coerce").max() or 0.0),
        float(pd.to_numeric(df["row_severity"], errors="coerce").max() or 0.0),
        1.0,
    )

    fig.add_trace(
        go.Bar(
            x=x,
            y=step_flagged_pct,
            name="step flagged %",
            marker={"color": "#355070"},
            opacity=0.80,
            showlegend=False,
            customdata=df[["row_severity"]].values,
            hovertemplate=(
                "timechunk=%{x}<br>step flagged=%{y:.2f}%"
                "<br>severity=%{customdata[0]:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=imp_flagged_pct,
            name="impulsive flagged %",
            marker={"color": "#d76a03"},
            opacity=0.80,
            showlegend=False,
            customdata=df[["row_severity"]].values,
            hovertemplate=(
                "timechunk=%{x}<br>impulsive flagged=%{y:.2f}%"
                "<br>severity=%{customdata[0]:.3f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["step_det_frac"],
            name="step detector fraction",
            mode="lines",
            line={"color": "#4c956c", "width": 3},
            hovertemplate="timechunk=%{x}<br>step detector fraction=%{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["step_alignment_frac"],
            name="step alignment fraction",
            mode="lines",
            line={"color": "#2c7da0", "width": 3},
            hovertemplate="timechunk=%{x}<br>step alignment fraction=%{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["network_impulsive_score_max"],
            name="network impulsive score",
            mode="lines+markers",
            line={"color": "#d76a03", "width": 2},
            hovertemplate="timechunk=%{x}<br>network impulsive score=%{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["max_slot_event_score"],
            name="captured event score",
            mode="lines",
            line={"color": "#355070", "width": 2},
            hovertemplate="timechunk=%{x}<br>captured event score=%{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Observation {obsnum} network {network}: timechunk diagnostics",
        margin={"l": 60, "r": 70, "t": 84, "b": 110},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "x": 0,
            "xanchor": "left",
        },
        height=980,
    )
    fig.update_xaxes(title_text="timechunk", row=3, col=1, showgrid=True, gridcolor="#e6e0d2")
    fig.update_xaxes(showgrid=True, gridcolor="#efeadd", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#efeadd", row=2, col=1)
    fig.update_yaxes(title_text="step %", row=1, col=1, range=[0.0, step_pct_max], showgrid=True, gridcolor="#e6e0d2")
    fig.update_yaxes(title_text="imp %", row=2, col=1, range=[0.0, imp_pct_max], showgrid=True, gridcolor="#e6e0d2")
    fig.update_yaxes(
        title_text="detector fraction (%)",
        row=3,
        col=1,
        secondary_y=False,
        range=[0.0, 1.0],
        tickformat=".0%",
        showgrid=True,
        gridcolor="#e6e0d2",
    )
    fig.update_yaxes(title_text="score", row=3, col=1, secondary_y=True, range=[0.0, score_max * 1.10], showgrid=False)
    return style_figure(fig)


def build_obs_rank_figure(obs_df: pd.DataFrame) -> go.Figure:
    if obs_df.empty:
        return empty_figure("No observation summaries")
    view = obs_df.copy()
    view["sort_flagged_fraction"] = view[["masked_fraction_overall", "impulsive_masked_fraction_overall"]].fillna(0.0).max(axis=1)
    view = view.sort_values("sort_flagged_fraction", ascending=True).tail(18)
    x_max = flagged_axis_limit(
        list(view["masked_fraction_overall"] * 100.0) +
        list(view["impulsive_masked_fraction_overall"] * 100.0)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["masked_fraction_overall"] * 100.0,
            y=view["obsnum"],
            name="step",
            mode="markers",
            marker={"color": "#355070", "size": 13, "symbol": "circle", "line": {"color": "#27425a", "width": 1}},
            customdata=view[
                [
                    "max_row_severity",
                    "masked_network_scans",
                    "masked_fraction_mean",
                    "top_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "obs=%{y}<br>overall step flagged=%{x:.2f}%"
                "<br>fired summaries=%{customdata[1]:.0f}"
                "<br>fired-summary mean=%{customdata[2]:.2%}"
                "<br>worst severity=%{customdata[0]:.3f}"
                "<br>top event score=%{customdata[3]:.1f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["impulsive_masked_fraction_overall"] * 100.0,
            y=view["obsnum"],
            name="impulsive",
            mode="markers",
            marker={"color": "#d76a03", "size": 13, "symbol": "diamond", "line": {"color": "#ab5300", "width": 1}},
            customdata=view[
                [
                    "max_row_severity",
                    "impulsive_masked_network_scans",
                    "impulsive_masked_fraction_mean",
                    "top_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "obs=%{y}<br>overall impulsive flagged=%{x:.2f}%"
                "<br>fired summaries=%{customdata[1]:.0f}"
                "<br>fired-summary mean=%{customdata[2]:.2%}"
                "<br>worst severity=%{customdata[0]:.3f}"
                "<br>top event score=%{customdata[3]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={
            "text": "Observation ranking by overall RTC flagged fraction",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 18},
        },
        xaxis_title="overall flagged fraction (%)",
        yaxis_title="observation",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "x": 0},
        margin={"l": 70, "r": 30, "t": 96, "b": 50},
        xaxis={"range": [0.0, x_max]},
    )
    for xref in (0.5, 1.0, 2.0, 5.0, 10.0):
        if xref > x_max:
            continue
        fig.add_vline(x=xref, line_width=1, line_dash="dot", line_color="#b8b1a0")
    return style_figure(fig)


def build_network_rank_figure(by_network_df: pd.DataFrame) -> go.Figure:
    if by_network_df.empty:
        return empty_figure("No network summary rows")
    view = by_network_df.copy()
    view["sort_flagged_fraction"] = view[["masked_fraction_overall", "impulsive_masked_fraction_overall"]].fillna(0.0).max(axis=1)
    view = view.sort_values("sort_flagged_fraction", ascending=True)
    labels = [f"nw{int(nw)}" for nw in view["network"]]
    x_max = flagged_axis_limit(
        list(view["masked_fraction_overall"] * 100.0) +
        list(view["impulsive_masked_fraction_overall"] * 100.0)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["masked_fraction_overall"] * 100.0,
            y=labels,
            name="step",
            mode="markers",
            marker={"color": "#355070", "size": 13, "symbol": "circle", "line": {"color": "#27425a", "width": 1}},
            customdata=view[
                [
                    "max_row_severity",
                    "total_masked_network_scans",
                    "worst_obsnum",
                    "max_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "network=%{y}<br>overall step flagged=%{x:.2f}%"
                "<br>fired summaries=%{customdata[1]:.0f}"
                "<br>worst observation=%{customdata[2]}"
                "<br>worst severity=%{customdata[0]:.3f}"
                "<br>top event score=%{customdata[3]:.1f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["impulsive_masked_fraction_overall"] * 100.0,
            y=labels,
            name="impulsive",
            mode="markers",
            marker={"color": "#d76a03", "size": 13, "symbol": "diamond", "line": {"color": "#ab5300", "width": 1}},
            customdata=view[
                [
                    "max_row_severity",
                    "total_impulsive_masked_network_scans",
                    "worst_obsnum",
                    "max_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "network=%{y}<br>overall impulsive flagged=%{x:.2f}%"
                "<br>fired summaries=%{customdata[1]:.0f}"
                "<br>worst observation=%{customdata[2]}"
                "<br>worst severity=%{customdata[0]:.3f}"
                "<br>top event score=%{customdata[3]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={
            "text": "Network ranking by overall RTC flagged fraction",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 18},
        },
        xaxis_title="overall flagged fraction (%)",
        yaxis_title="network",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "x": 0},
        margin={"l": 70, "r": 30, "t": 96, "b": 50},
        xaxis={"range": [0.0, x_max]},
    )
    for xref in (0.5, 1.0, 2.0, 5.0, 10.0):
        if xref > x_max:
            continue
        fig.add_vline(x=xref, line_width=1, line_dash="dot", line_color="#b8b1a0")
    return style_figure(fig)


def build_obs_network_rank_figure(obs_network_view: pd.DataFrame, obsnum: str) -> go.Figure:
    if obs_network_view.empty:
        return empty_figure(f"No network summaries for observation {obsnum}")
    view = obs_network_view.copy()
    view["masked_fraction_fired_mean"] = view.apply(
        lambda row: (
            float(row["masked_fraction_sum"]) / max(int(row["masked_scans"]), 1)
            if int(row["masked_scans"]) > 0 else float("nan")
        ),
        axis=1,
    )
    view["impulsive_masked_fraction_fired_mean"] = view.apply(
        lambda row: (
            float(row["impulsive_masked_fraction_sum"]) / max(int(row["impulsive_masked_scans"]), 1)
            if int(row["impulsive_masked_scans"]) > 0 else float("nan")
        ),
        axis=1,
    )
    view["sort_flagged_fraction"] = view[["masked_fraction_overall", "impulsive_masked_fraction_overall"]].fillna(0.0).max(axis=1)
    view = view.sort_values("sort_flagged_fraction", ascending=True)
    labels = [f"nw{int(nw)}" for nw in view["network"]]
    x_max = flagged_axis_limit(
        list(view["masked_fraction_overall"] * 100.0) +
        list(view["impulsive_masked_fraction_overall"] * 100.0)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["masked_fraction_overall"] * 100.0,
            y=labels,
            name="step",
            mode="markers",
            marker={"color": "#355070", "size": 13, "symbol": "circle", "line": {"color": "#27425a", "width": 1}},
            customdata=view[
                [
                    "max_row_severity",
                    "masked_scans",
                    "masked_fraction_fired_mean",
                    "top_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "network=%{y}<br>observation-level step flagged=%{x:.2f}%"
                "<br>fired timechunks=%{customdata[1]:.0f}"
                "<br>fired-timechunk mean=%{customdata[2]:.2%}"
                "<br>worst severity=%{customdata[0]:.3f}"
                "<br>top event score=%{customdata[3]:.1f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["impulsive_masked_fraction_overall"] * 100.0,
            y=labels,
            name="impulsive",
            mode="markers",
            marker={"color": "#d76a03", "size": 13, "symbol": "diamond", "line": {"color": "#ab5300", "width": 1}},
            customdata=view[
                [
                    "max_row_severity",
                    "impulsive_masked_scans",
                    "impulsive_masked_fraction_fired_mean",
                    "top_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "network=%{y}<br>observation-level impulsive flagged=%{x:.2f}%"
                "<br>fired timechunks=%{customdata[1]:.0f}"
                "<br>fired-timechunk mean=%{customdata[2]:.2%}"
                "<br>worst severity=%{customdata[0]:.3f}"
                "<br>top event score=%{customdata[3]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={
            "text": f"Observation {obsnum}: network ranking by flagged fraction",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 18},
        },
        xaxis_title="observation-level flagged fraction (%)",
        yaxis_title="network",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        margin={"l": 70, "r": 30, "t": 94, "b": 50},
        xaxis={"range": [0.0, x_max]},
    )
    for xref in (0.5, 1.0, 2.0, 5.0, 10.0):
        if xref > x_max:
            continue
        fig.add_vline(x=xref, line_width=1, line_dash="dot", line_color="#b8b1a0")
    return style_figure(fig)


def build_top_scan_figure(scan_view: pd.DataFrame, obsnum: str) -> go.Figure:
    if scan_view.empty:
        return empty_figure(f"No timechunk summaries for observation {obsnum}")
    view = scan_view.copy()
    view["step_flagged_pct"] = pd.to_numeric(view["step_mask_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    view["imp_flagged_pct"] = pd.to_numeric(view["impulsive_mask_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    view["sort_flagged_pct"] = view[["step_flagged_pct", "imp_flagged_pct"]].max(axis=1)
    view = view.sort_values("sort_flagged_pct", ascending=False).head(18).copy()
    view["row_label"] = view.apply(
        lambda row: f"timechunk {int(row['output_scan_index'])} nw{int(row['network'])}", axis=1
    )
    view = view.sort_values("sort_flagged_pct", ascending=True)
    x_max = flagged_axis_limit(
        list(view["step_flagged_pct"]) +
        list(view["imp_flagged_pct"])
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["step_flagged_pct"],
            y=view["row_label"],
            name="step",
            mode="markers",
            marker={"color": "#355070", "size": 12, "symbol": "circle", "line": {"color": "#27425a", "width": 1}},
            customdata=view[
                [
                    "row_severity",
                    "step_det_frac",
                    "step_alignment_frac",
                    "max_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "%{y}<br>step flagged=%{x:.2f}%"
                "<br>severity=%{customdata[0]:.3f}"
                "<br>step detector fraction=%{customdata[1]:.2%}"
                "<br>step alignment fraction=%{customdata[2]:.2%}"
                "<br>captured event score=%{customdata[3]:.1f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["imp_flagged_pct"],
            y=view["row_label"],
            name="impulsive",
            mode="markers",
            marker={"color": "#d76a03", "size": 12, "symbol": "diamond", "line": {"color": "#ab5300", "width": 1}},
            customdata=view[
                [
                    "row_severity",
                    "network_impulsive_score_max",
                    "max_slot_event_score",
                    "impulsive_mask_cluster_network_count",
                ]
            ].values,
            hovertemplate=(
                "%{y}<br>impulsive flagged=%{x:.2f}%"
                "<br>severity=%{customdata[0]:.3f}"
                "<br>network impulsive score=%{customdata[1]:.2f}"
                "<br>captured event score=%{customdata[2]:.1f}"
                "<br>cluster networks=%{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={
            "text": f"Observation {obsnum}: most-flagged timechunk/network summaries",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 18},
        },
        xaxis_title="flagged fraction (%)",
        yaxis_title="timechunk / network summary",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        margin={"l": 110, "r": 30, "t": 94, "b": 50},
        xaxis={"range": [0.0, x_max]},
    )
    for xref in (0.5, 1.0, 2.0, 5.0, 10.0):
        if xref > x_max:
            continue
        fig.add_vline(x=xref, line_width=1, line_dash="dot", line_color="#b8b1a0")
    return style_figure(fig)


def build_top_slot_figure(slot_view: pd.DataFrame, obsnum: str, network: int) -> go.Figure:
    if slot_view.empty:
        return empty_figure(f"Observation {obsnum} network {network}: no captured impulsive events")
    view = slot_view.sort_values("event_score", ascending=False).head(20).copy()
    view["slot_label"] = view.apply(
        lambda row: f"timechunk {int(row['output_scan_index'])} event {int(row['slot'])} uid {int(row['apt_uid'])}",
        axis=1,
    )
    view = view.sort_values("event_score", ascending=True)

    fig = go.Figure()
    for kind, color in EVENT_KIND_COLORS.items():
        part = view.loc[view["event_kind_label"] == kind]
        if part.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=part["event_score"],
                y=part["slot_label"],
                orientation="h",
                name=kind,
                marker={"color": color},
                customdata=part[["peak_abs_z", "peak_delta_abs_z"]].values,
                hovertemplate=(
                    "%{y}<br>event score=%{x:.3f}"
                    "<br>max raw abs-z=%{customdata[0]:.3f}"
                    "<br>max delta abs-z=%{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Observation {obsnum} network {network}: top captured impulsive events",
        xaxis_title="event score = max(raw abs-z, delta abs-z)",
        yaxis_title="captured event",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 150, "r": 30, "t": 50, "b": 50},
    )
    return style_figure(fig)


def build_overview_cards(
    obs_df: pd.DataFrame,
    by_network_df: pd.DataFrame,
    scan_df: pd.DataFrame,
    slot_df: pd.DataFrame,
):
    worst_obs = obs_df.sort_values("max_row_severity", ascending=False).iloc[0]
    worst_network = by_network_df.sort_values("max_row_severity", ascending=False).iloc[0]
    top_slot = slot_df.sort_values("event_score", ascending=False).iloc[0] if not slot_df.empty else None
    step_loss = pd.to_numeric(
        scan_df.loc[scan_df["step_mask_applied"] != 0, "step_mask_flagged_fraction"],
        errors="coerce",
    )
    imp_loss = pd.to_numeric(
        scan_df.loc[scan_df["impulsive_mask_applied"] != 0, "impulsive_mask_flagged_fraction"],
        errors="coerce",
    )
    step_loss_all = pd.to_numeric(scan_df["step_mask_flagged_fraction"], errors="coerce").fillna(0.0)
    imp_loss_all = pd.to_numeric(scan_df["impulsive_mask_flagged_fraction"], errors="coerce").fillna(0.0)
    step_loss_overall = float(step_loss_all.mean()) if len(step_loss_all) else float("nan")
    imp_loss_overall = float(imp_loss_all.mean()) if len(imp_loss_all) else float("nan")
    step_fire_frac = float((scan_df["step_mask_applied"] != 0).mean()) if len(scan_df) else float("nan")
    imp_fire_frac = float((scan_df["impulsive_mask_applied"] != 0).mean()) if len(scan_df) else float("nan")
    step_loss_mean = float(step_loss.mean()) if step_loss.notna().any() else float("nan")
    step_loss_max = float(step_loss.max()) if step_loss.notna().any() else float("nan")
    imp_loss_mean = float(imp_loss.mean()) if imp_loss.notna().any() else float("nan")
    imp_loss_max = float(imp_loss.max()) if imp_loss.notna().any() else float("nan")
    return html.Div(
        [
            metric_card("Observations", format_count(len(obs_df)), "distinct observations in this reduction", "#7c644a"),
            metric_card("Timechunk-Network Summaries", format_count(len(scan_df)), "timechunk x network diagnostic summaries", "#8f5536"),
            metric_card(
                "Overall Step Flagged",
                format_metric(step_loss_overall * 100.0, 1, "%"),
                f"mask fired in {format_fraction_pct(step_fire_frac, 0)} of summaries; when fired, mean {format_metric(step_loss_mean * 100.0, 1, '%')}",
                "#355070",
            ),
            metric_card(
                "Overall Impulsive Flagged",
                format_metric(imp_loss_overall * 100.0, 1, "%"),
                f"mask fired in {format_fraction_pct(imp_fire_frac, 0)} of summaries; when fired, mean {format_metric(imp_loss_mean * 100.0, 1, '%')}",
                "#d76a03",
            ),
            metric_card(
                "Worst Observation",
                str(worst_obs["obsnum"]),
                f"max severity {format_metric(worst_obs['max_row_severity'], 3)}",
                "#8b1e3f",
            ),
            metric_card(
                "Worst Network",
                f"nw{int(worst_network['network'])}",
                f"max severity {format_metric(worst_network['max_row_severity'], 3)}",
                "#4c956c",
            ),
            metric_card(
                "Strongest Captured Event Score",
                format_metric(top_slot["event_score"], 1) if top_slot is not None else "n/a",
                (
                    f"robust-sigma; obs {top_slot['obsnum']} nw{int(top_slot['network'])} "
                    f"timechunk {int(top_slot['output_scan_index'])} {top_slot['event_kind_label']}"
                ) if top_slot is not None else "no captured events",
                "#cf3f3f",
            ),
        ],
        style=CARD_GRID_STYLE,
    )


def build_selected_obs_cards(obsnum: str, obs_network_view: pd.DataFrame, scan_view: pd.DataFrame, slot_view: pd.DataFrame):
    if scan_view.empty:
        return html.Div("No timechunk summaries for this observation.", style=PANEL_STYLE)
    worst_row = scan_view.sort_values("row_severity", ascending=False).iloc[0]
    worst_network = obs_network_view.sort_values("max_row_severity", ascending=False).iloc[0]
    top_slot = slot_view.sort_values("event_score", ascending=False).iloc[0] if not slot_view.empty else None
    step_loss = pd.to_numeric(
        scan_view.loc[scan_view["step_mask_applied"] != 0, "step_mask_flagged_fraction"],
        errors="coerce",
    )
    imp_loss = pd.to_numeric(
        scan_view.loc[scan_view["impulsive_mask_applied"] != 0, "impulsive_mask_flagged_fraction"],
        errors="coerce",
    )
    step_loss_all = pd.to_numeric(scan_view["step_mask_flagged_fraction"], errors="coerce").fillna(0.0)
    imp_loss_all = pd.to_numeric(scan_view["impulsive_mask_flagged_fraction"], errors="coerce").fillna(0.0)
    step_loss_overall = float(step_loss_all.mean()) if len(step_loss_all) else float("nan")
    imp_loss_overall = float(imp_loss_all.mean()) if len(imp_loss_all) else float("nan")
    step_fire_frac = float((scan_view["step_mask_applied"] != 0).mean()) if len(scan_view) else float("nan")
    imp_fire_frac = float((scan_view["impulsive_mask_applied"] != 0).mean()) if len(scan_view) else float("nan")
    step_loss_mean = float(step_loss.mean()) if step_loss.notna().any() else float("nan")
    step_loss_max = float(step_loss.max()) if step_loss.notna().any() else float("nan")
    imp_loss_mean = float(imp_loss.mean()) if imp_loss.notna().any() else float("nan")
    imp_loss_max = float(imp_loss.max()) if imp_loss.notna().any() else float("nan")
    return html.Div(
        [
            metric_card(
                f"Observation {obsnum} Worst Summary",
                format_metric(worst_row["row_severity"], 3),
                f"timechunk {int(worst_row['output_scan_index'])} nw{int(worst_row['network'])}",
                "#8b1e3f",
            ),
            metric_card(
                "Worst Network",
                f"nw{int(worst_network['network'])}",
                f"severity {format_metric(worst_network['max_row_severity'], 3)}",
                "#4c956c",
            ),
            metric_card(
                "Overall Step Flagged",
                format_metric(step_loss_overall * 100.0, 1, "%"),
                f"mask fired in {format_fraction_pct(step_fire_frac, 0)} of summaries; when fired, mean {format_metric(step_loss_mean * 100.0, 1, '%')}",
                "#355070",
            ),
            metric_card(
                "Overall Impulsive Flagged",
                format_metric(imp_loss_overall * 100.0, 1, "%"),
                f"mask fired in {format_fraction_pct(imp_fire_frac, 0)} of summaries; when fired, mean {format_metric(imp_loss_mean * 100.0, 1, '%')}",
                "#d76a03",
            ),
            metric_card(
                "Strongest Captured Event Score",
                format_metric(top_slot["event_score"], 1) if top_slot is not None else "n/a",
                (
                    f"robust-sigma; "
                    f"nw{int(top_slot['network'])} timechunk {int(top_slot['output_scan_index'])} "
                    f"{top_slot['event_kind_label']}"
                ) if top_slot is not None else "no captured event in selected network",
                "#cf3f3f",
            ),
        ],
        style=CARD_GRID_STYLE,
    )


def build_detail_panel(
    obs_network_view: pd.DataFrame,
    scan_view: pd.DataFrame,
    slot_view: pd.DataFrame,
) -> html.Details:
    scan_top_view = scan_view.sort_values("row_severity", ascending=False).head(12)
    return html.Details(
        [
            html.Summary("Exact Values"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Observation-network summaries", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            table(
                                [
                                    "network",
                                    "max_row_severity",
                                    "max_step_det_frac",
                                    "max_network_impulsive_score",
                                    "max_cm_lowmid",
                                    "masked_scans",
                                    "impulsive_masked_scans",
                                    "top_slot_event_score",
                                ],
                                rounded_records(
                                    obs_network_view,
                                    [
                                        "network",
                                        "max_row_severity",
                                        "max_step_det_frac",
                                        "max_network_impulsive_score",
                                        "max_cm_lowmid",
                                        "masked_scans",
                                        "impulsive_masked_scans",
                                        "top_slot_event_score",
                                    ],
                                ),
                                page_size=8,
                            ),
                        ],
                        style=PANEL_STYLE,
                    ),
                    html.Div(
                        [
                            html.Div("Top timechunk/network summaries", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            table(
                                [
                                    "output_scan_index",
                                    "network",
                                    "row_severity",
                                    "step_det_frac",
                                    "step_alignment_frac",
                                    "network_impulsive_score_max",
                                    "max_slot_event_score",
                                    "step_mask_applied",
                                    "impulsive_mask_applied",
                                    "impulsive_mask_candidate_available",
                                    "impulsive_mask_cross_network_trigger",
                                    "impulsive_mask_high_score_override_trigger",
                                ],
                                rounded_records(
                                    scan_top_view,
                                    [
                                        "output_scan_index",
                                        "network",
                                        "row_severity",
                                        "step_det_frac",
                                        "step_alignment_frac",
                                        "network_impulsive_score_max",
                                        "max_slot_event_score",
                                        "step_mask_applied",
                                        "impulsive_mask_applied",
                                        "impulsive_mask_candidate_available",
                                        "impulsive_mask_cross_network_trigger",
                                        "impulsive_mask_high_score_override_trigger",
                                    ],
                                ),
                                page_size=8,
                            ),
                        ],
                        style=PANEL_STYLE,
                    ),
                    html.Div(
                        [
                            html.Div("Top captured events for the selected network", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            table(
                                [
                                    "output_scan_index",
                                    "slot",
                                    "apt_uid",
                                    "event_kind_label",
                                    "event_score",
                                    "peak_abs_z",
                                    "peak_delta_abs_z",
                                ],
                                rounded_records(
                                    slot_view.head(16),
                                    [
                                        "output_scan_index",
                                        "slot",
                                        "apt_uid",
                                        "event_kind_label",
                                        "event_score",
                                        "peak_abs_z",
                                        "peak_delta_abs_z",
                                    ],
                                ),
                                page_size=8,
                            ),
                        ],
                        style=PANEL_STYLE,
                    ),
                ]
            ),
        ],
        open=False,
        style={**HELP_BOX_STYLE, "marginTop": "6px"},
    )


def build_app(args: argparse.Namespace) -> Dash:
    data = load_reduction_tables(
        Path(args.redu_dir),
        array=args.array,
        networks_spec=args.networks,
        obsnums_spec=args.obsnums,
    )

    obs_df = pd.DataFrame(data["obs_rows"]).sort_values(
        ["max_row_severity", "obsnum"], ascending=[False, True]
    )
    obs_network_df = pd.DataFrame(data["obs_network_rows"]).sort_values(
        ["obsnum", "network"], ascending=[True, True]
    )
    scan_df = pd.DataFrame(data["scan_network_rows"]).sort_values(
        ["obsnum", "output_scan_index", "network"], ascending=[True, True, True]
    )
    slot_df = pd.DataFrame(data["slot_rows"]).sort_values(
        ["obsnum", "event_score"], ascending=[True, False]
    )
    by_network_df = pd.DataFrame(data["by_network_rows"]).sort_values(
        ["max_row_severity", "network"], ascending=[False, True]
    )

    obs_options = [{"label": obs, "value": obs} for obs in obs_df["obsnum"].astype(str).tolist()]
    default_obs = obs_options[0]["value"]
    default_networks = (
        obs_network_df.loc[obs_network_df["obsnum"] == default_obs, "network"].astype(int).tolist()
    )
    default_network = default_networks[0] if default_networks else int(data["selected_networks"][0])

    app = Dash(__name__)
    app.title = "rtcdiag engineering dashboard"
    app.layout = html.Div(
        [
            html.H2("RTC Engineering Dashboard", style={"marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(f"Reduction: {Path(args.redu_dir).expanduser().resolve()}"),
                    html.Div(
                        f"Array: {data['array']} | rtcdiag={data['n_rtcdiag']} | rtc fallback={data['n_rtc_fallback']}"
                    ),
                    html.Div(
                        f"Networks: {','.join(str(nw) for nw in data['selected_networks'])} | "
                        f"Impulsive threshold: {data['impulsive_threshold']}"
                    ),
                ],
                style={"marginBottom": "16px", "fontFamily": "Menlo, Monaco, Consolas, monospace", "fontSize": "13px"},
            ),
            html.Details(
                [
                    html.Summary("How To Read This Dashboard"),
                    dcc.Markdown(
                        """
This view is meant for fast engineering triage, not for exhaustive tabular inspection.

- Start with the **overview cards** and **ranking panels** to find the worst observations and networks.
- Use the **heatmap** to see whether a problem is isolated to a few timechunks or spread across an observation.
- Use the **trend plot** to see which metric is actually driving a suspicious network.
- Use the **timechunk ranking** and **captured event ranking** panels to jump from broad survey context to concrete events.
- Open **Exact Values** only when you need precise numbers for a small set of summaries.

Definitions:

- **event**: a localized contamination episode found in one detector timestream; captured events are stored examples of the strongest ones.

    - **impulsive**: a brief spike-like event concentrated in time. In practice this includes both narrow cosmic-ray-like hits and some bursty RFI-like events when they are compact enough to trigger the impulsive finders.
    - **step**: a level-shift event with a persistent baseline change after the transition, unlike a brief impulse.

- **severity**: a dimensionless ranking score used to decide where to look first. Around `1` means at least one reference contamination threshold was reached, values below `1` are usually quieter, and values above `1` are progressively more suspicious. It is not the keep-or-reject metric by itself.
- **robust sigma units**: the excursion size after dividing by a robust scatter estimate rather than an ordinary RMS standard deviation.
- **overall step flagged** / **overall impulsive flagged**: the mean flagged fraction across all timechunk-network summaries, with non-fired summaries contributing zero. The subtitle also shows the conditional mean and maximum over just the summaries where that mask fired.
- **strongest captured event score**: the largest stored RTC event score, in robust-sigma units. For a raw-like event it is `|x-center|/sigma_robust`; for a delta-like event it is `|delta_x-median(delta_x)|/sigma_delta,robust`. Use it to compare how extreme captured transients are, not as a calibrated physical unit.

`row_severity` is a ranking score, not a physical unit. It is the largest of several normalized contamination indicators, so it should be used to rank and compare timechunk-network summaries, not as a calibrated threshold by itself.
                        """
                    ),
                ],
                style={**HELP_BOX_STYLE, "marginBottom": "18px"},
                open=False,
            ),
            html.Div(build_overview_cards(obs_df, by_network_df, scan_df, slot_df), style=PANEL_STYLE),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Observation Triage", style={"marginTop": "0"}),
                            html.Div(
                                "Grouped bars show overall flagged fraction for step and impulsive masking. Use these first; severity is still available in hover, but the flagged fractions are the more practical keep-or-reject signal.",
                                style={"marginBottom": "8px", "fontSize": "14px", "color": "#4e483c"},
                            ),
                            dcc.Graph(figure=build_obs_rank_figure(obs_df), config=GRAPH_CONFIG),
                        ],
                        style=PANEL_STYLE,
                    ),
                    html.Div(
                        [
                            html.H3("Network Triage", style={"marginTop": "0"}),
                            html.Div(
                                "Grouped bars show how much each network is being flagged overall across the selected reduction. Dashed lines at 1%, 5%, and 10% give quick reference points.",
                                style={"marginBottom": "8px", "fontSize": "14px", "color": "#4e483c"},
                            ),
                            dcc.Graph(figure=build_network_rank_figure(by_network_df), config=GRAPH_CONFIG),
                        ],
                        style=PANEL_STYLE,
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Observation"),
                            dcc.Dropdown(
                                id="obs-dropdown",
                                options=obs_options,
                                value=default_obs,
                                clearable=False,
                            ),
                        ],
                        style={"width": "240px"},
                    ),
                    html.Div(
                        [
                            html.Label("Network"),
                            dcc.Dropdown(id="network-dropdown", value=default_network, clearable=False),
                        ],
                        style={"width": "180px"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
            ),
            html.Div(id="selected-obs-cards", style=PANEL_STYLE),
            section_help(
                "Selected Observation Views",
                """
The next panels are for one selected observation and one selected network.

- **Heatmaps** answer: where are step and impulsive masks actually removing data, and by how much?
- **Trend plot** answers: is this network step-like, impulsive, or both?
- **Network ranking** answers: which networks dominate this observation?
- **Timechunk ranking** answers: which exact timechunk/network summaries deserve inspection first?
- **Captured event ranking** answers: what are the strongest captured compact events in the selected network?
                """,
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="heatmap", config=GRAPH_CONFIG), style=PANEL_STYLE),
                    html.Div(dcc.Graph(id="network-trend", config=GRAPH_CONFIG), style=PANEL_STYLE),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="obs-network-rank", config=GRAPH_CONFIG), style=PANEL_STYLE),
                    html.Div(dcc.Graph(id="top-scan-rank", config=GRAPH_CONFIG), style=PANEL_STYLE),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [html.Div(dcc.Graph(id="top-slot-rank", config=GRAPH_CONFIG), style=PANEL_STYLE)],
                style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"},
            ),
            html.Div(id="detail-panel"),
        ],
        style=PAGE_STYLE,
    )

    @app.callback(Output("network-dropdown", "options"), Output("network-dropdown", "value"), Input("obs-dropdown", "value"))
    def update_network_options(obsnum: str):
        networks = (
            obs_network_df.loc[obs_network_df["obsnum"] == obsnum, "network"].astype(int).tolist()
        )
        options = [{"label": str(nw), "value": int(nw)} for nw in networks]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("selected-obs-cards", "children"),
        Output("heatmap", "figure"),
        Output("network-trend", "figure"),
        Output("obs-network-rank", "figure"),
        Output("top-scan-rank", "figure"),
        Output("top-slot-rank", "figure"),
        Output("detail-panel", "children"),
        Input("obs-dropdown", "value"),
        Input("network-dropdown", "value"),
    )
    def update_detail(obsnum: str, network: int):
        obs_network_view = obs_network_df.loc[obs_network_df["obsnum"] == obsnum].copy()
        scan_view = scan_df.loc[scan_df["obsnum"] == obsnum].copy()
        slot_view = slot_df.loc[(slot_df["obsnum"] == obsnum) & (slot_df["network"] == network)].copy()

        return (
            build_selected_obs_cards(obsnum, obs_network_view, scan_view, slot_view),
            build_heatmap(scan_df, obsnum),
            build_network_trend(scan_df, obsnum, int(network)),
            build_obs_network_rank_figure(obs_network_view, obsnum),
            build_top_scan_figure(scan_view, obsnum),
            build_top_slot_figure(slot_view, obsnum, int(network)),
            build_detail_panel(obs_network_view, scan_view, slot_view),
        )

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
