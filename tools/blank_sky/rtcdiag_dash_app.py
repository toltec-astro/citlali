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


def metric_card(title: str, value: str, note: str, accent: str = "#5b4b2a"):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "letterSpacing": "0.06em", "textTransform": "uppercase", "color": "#695f49"}),
            html.Div(value, style={"fontSize": "34px", "lineHeight": "1.0", "fontWeight": "bold", "margin": "10px 0 6px 0"}),
            html.Div(note, style={"fontSize": "13px", "color": "#5f5644", "lineHeight": "1.35"}),
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
        return empty_figure(f"No scan/network rows for obsnum {obsnum}")

    obs_df["label"] = obs_df.apply(
        lambda row: (
            f"scan={int(row['output_scan_index'])}"
            f"<br>nw={int(row['network'])}"
            f"<br>severity={row['row_severity']:.3f}"
            f"<br>step_det={row['step_det_frac']:.4f}"
            f"<br>imp_score={row['network_impulsive_score_max']:.3f}"
            f"<br>slot={row['max_slot_event_score']:.3f}"
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
    pivot = obs_df.pivot(index="network", columns="output_scan_index", values="row_severity").sort_index()
    hover = obs_df.pivot(index="network", columns="output_scan_index", values="label").sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.values,
            text=hover.values,
            hovertemplate="%{text}<extra></extra>",
            colorscale="YlOrRd",
            colorbar={"title": "severity"},
        )
    )
    fig.update_layout(
        title=f"Obsnum {obsnum}: scan/network severity",
        xaxis_title="output scan",
        yaxis_title="network",
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
    )
    return style_figure(fig)


def build_network_trend(scan_df: pd.DataFrame, obsnum: str, network: int) -> go.Figure:
    df = scan_df.loc[(scan_df["obsnum"] == obsnum) & (scan_df["network"] == network)].copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty:
        fig.update_layout(title=f"No scan rows for obsnum {obsnum} network {network}")
        return style_figure(fig)

    x = df["output_scan_index"]
    fig.add_trace(
        go.Scatter(x=x, y=df["row_severity"], name="row_severity", mode="lines+markers", line={"color": "#8b1e3f"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["network_impulsive_score_max"], name="imp_score_max", mode="lines", line={"color": "#d76a03"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["max_slot_event_score"], name="slot_score_max", mode="lines", line={"color": "#355070"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["step_det_frac"], name="step_det_frac", mode="lines", line={"color": "#4c956c"}),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["step_alignment_frac"], name="step_alignment_frac", mode="lines", line={"color": "#2c7da0"}),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["impulsive_mask_applied"],
            name="imp_mask",
            mode="markers",
            marker={"symbol": "x", "size": 8, "color": "#d76a03"},
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["step_mask_applied"],
            name="step_mask",
            mode="markers",
            marker={"symbol": "cross", "size": 8, "color": "#355070"},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Obsnum {obsnum} network {network}: scan trends",
        margin={"l": 60, "r": 60, "t": 50, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(title_text="output scan")
    fig.update_yaxes(title_text="severity / score", secondary_y=False)
    fig.update_yaxes(title_text="fraction / mask flag", secondary_y=True)
    return style_figure(fig)


def build_obs_rank_figure(obs_df: pd.DataFrame) -> go.Figure:
    if obs_df.empty:
        return empty_figure("No obsnum rows")
    view = obs_df.sort_values("max_row_severity", ascending=True).tail(18).copy()
    fig = go.Figure(
        go.Bar(
            x=view["max_row_severity"],
            y=view["obsnum"],
            orientation="h",
            marker={
                "color": view["impulsive_masked_network_scans"],
                "colorscale": "YlOrBr",
                "line": {"color": "#7b6740", "width": 1},
                "colorbar": {"title": "imp masked"},
            },
            customdata=view[
                [
                    "max_step_det_frac",
                    "max_cm_lowmid",
                    "top_slot_event_score",
                    "masked_network_scans",
                    "impulsive_masked_network_scans",
                ]
            ].values,
            hovertemplate=(
                "obs=%{y}<br>severity=%{x:.3f}"
                "<br>max_step_det=%{customdata[0]:.4f}"
                "<br>max_cm_lowmid=%{customdata[1]:.3f}"
                "<br>top_slot=%{customdata[2]:.3f}"
                "<br>step_masked_rows=%{customdata[3]:.0f}"
                "<br>imp_masked_rows=%{customdata[4]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Obsnum ranking by worst scan/network severity",
        xaxis_title="max row severity",
        yaxis_title="obsnum",
        margin={"l": 70, "r": 30, "t": 50, "b": 50},
    )
    return style_figure(fig)


def build_network_rank_figure(by_network_df: pd.DataFrame) -> go.Figure:
    if by_network_df.empty:
        return empty_figure("No network summary rows")
    view = by_network_df.sort_values("max_row_severity", ascending=True).copy()
    fig = go.Figure(
        go.Bar(
            x=view["max_row_severity"],
            y=[f"nw{int(nw)}" for nw in view["network"]],
            orientation="h",
            marker={
                "color": view["total_impulsive_masked_network_scans"],
                "colorscale": "Sunsetdark",
                "line": {"color": "#5b4324", "width": 1},
                "colorbar": {"title": "imp masked"},
            },
            customdata=view[
                [
                    "max_max_step_det_frac",
                    "max_max_cm_lowmid",
                    "max_slot_event_score",
                    "worst_obsnum",
                ]
            ].values,
            hovertemplate=(
                "network=%{y}<br>severity=%{x:.3f}"
                "<br>max_step_det=%{customdata[0]:.4f}"
                "<br>max_cm_lowmid=%{customdata[1]:.3f}"
                "<br>top_slot=%{customdata[2]:.3f}"
                "<br>worst_obs=%{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Network ranking across the selected reduction",
        xaxis_title="max row severity",
        yaxis_title="network",
        margin={"l": 70, "r": 30, "t": 50, "b": 50},
    )
    return style_figure(fig)


def build_obs_network_rank_figure(obs_network_view: pd.DataFrame, obsnum: str) -> go.Figure:
    if obs_network_view.empty:
        return empty_figure(f"No network rows for obsnum {obsnum}")
    view = obs_network_view.sort_values("max_row_severity", ascending=True).copy()
    fig = go.Figure(
        go.Bar(
            x=view["max_row_severity"],
            y=[f"nw{int(nw)}" for nw in view["network"]],
            orientation="h",
            marker={
                "color": view["impulsive_masked_scans"],
                "colorscale": "Tealgrn",
                "line": {"color": "#416165", "width": 1},
                "colorbar": {"title": "imp masked scans"},
            },
            customdata=view[
                [
                    "masked_scans",
                    "impulsive_masked_scans",
                    "max_network_impulsive_score",
                    "top_slot_event_score",
                ]
            ].values,
            hovertemplate=(
                "network=%{y}<br>severity=%{x:.3f}"
                "<br>step_masked_scans=%{customdata[0]:.0f}"
                "<br>imp_masked_scans=%{customdata[1]:.0f}"
                "<br>max_network_imp_score=%{customdata[2]:.3f}"
                "<br>top_slot=%{customdata[3]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Obsnum {obsnum}: network ranking",
        xaxis_title="max row severity",
        yaxis_title="network",
        margin={"l": 70, "r": 30, "t": 50, "b": 50},
    )
    return style_figure(fig)


def build_top_scan_figure(scan_view: pd.DataFrame, obsnum: str) -> go.Figure:
    if scan_view.empty:
        return empty_figure(f"No scan rows for obsnum {obsnum}")
    view = scan_view.sort_values("row_severity", ascending=False).head(18).copy()
    view["row_label"] = view.apply(
        lambda row: f"scan {int(row['output_scan_index'])} nw{int(row['network'])}", axis=1
    )
    view["mask_state"] = view.apply(mask_state_label, axis=1)
    view = view.sort_values("row_severity", ascending=True)

    fig = go.Figure()
    for state, color in MASK_COLORS.items():
        part = view.loc[view["mask_state"] == state]
        if part.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=part["row_severity"],
                y=part["row_label"],
                orientation="h",
                name=state,
                marker={"color": color},
                customdata=part[
                    [
                        "step_det_frac",
                        "step_alignment_frac",
                        "network_impulsive_score_max",
                        "max_slot_event_score",
                        "impulsive_mask_cluster_network_count",
                    ]
                ].values,
                hovertemplate=(
                    "%{y}<br>severity=%{x:.3f}"
                    "<br>step_det=%{customdata[0]:.4f}"
                    "<br>step_align=%{customdata[1]:.4f}"
                    "<br>imp_score=%{customdata[2]:.3f}"
                    "<br>slot_score=%{customdata[3]:.3f}"
                    "<br>cluster_nw=%{customdata[4]:.0f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Obsnum {obsnum}: highest-severity scan/network rows",
        xaxis_title="row severity",
        yaxis_title="scan / network row",
        barmode="overlay",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 110, "r": 30, "t": 50, "b": 50},
    )
    return style_figure(fig)


def build_top_slot_figure(slot_view: pd.DataFrame, obsnum: str, network: int) -> go.Figure:
    if slot_view.empty:
        return empty_figure(f"Obsnum {obsnum} network {network}: no stored impulsive slots")
    view = slot_view.sort_values("event_score", ascending=False).head(20).copy()
    view["slot_label"] = view.apply(
        lambda row: f"scan {int(row['output_scan_index'])} slot {int(row['slot'])} uid {int(row['apt_uid'])}",
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
                    "%{y}<br>event_score=%{x:.3f}"
                    "<br>peak_abs_z=%{customdata[0]:.3f}"
                    "<br>peak_delta_abs_z=%{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Obsnum {obsnum} network {network}: top captured impulsive slots",
        xaxis_title="event score",
        yaxis_title="captured slot",
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
    return html.Div(
        [
            metric_card("Obsnums", format_count(len(obs_df)), "distinct observations in this reduction", "#7c644a"),
            metric_card("Rows", format_count(len(scan_df)), "scan x network summary rows", "#8f5536"),
            metric_card(
                "Step-Masked Rows",
                format_count(scan_df["step_mask_applied"].sum()),
                "rows where the RTC step path acted",
                "#355070",
            ),
            metric_card(
                "Impulsive-Masked Rows",
                format_count(scan_df["impulsive_mask_applied"].sum()),
                "rows where the RTC impulsive path acted",
                "#d76a03",
            ),
            metric_card(
                "Worst Obsnum",
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
                "Strongest Slot",
                format_metric(top_slot["event_score"], 1) if top_slot is not None else "n/a",
                (
                    f"obs {top_slot['obsnum']} nw{int(top_slot['network'])} "
                    f"{top_slot['event_kind_label']}"
                ) if top_slot is not None else "no captured slots",
                "#cf3f3f",
            ),
        ],
        style=CARD_GRID_STYLE,
    )


def build_selected_obs_cards(obsnum: str, obs_network_view: pd.DataFrame, scan_view: pd.DataFrame, slot_view: pd.DataFrame):
    if scan_view.empty:
        return html.Div("No scan rows for this obsnum.", style=PANEL_STYLE)
    worst_row = scan_view.sort_values("row_severity", ascending=False).iloc[0]
    worst_network = obs_network_view.sort_values("max_row_severity", ascending=False).iloc[0]
    top_slot = slot_view.sort_values("event_score", ascending=False).iloc[0] if not slot_view.empty else None
    return html.Div(
        [
            metric_card(
                f"Obs {obsnum} Worst Row",
                format_metric(worst_row["row_severity"], 3),
                f"scan {int(worst_row['output_scan_index'])} nw{int(worst_row['network'])}",
                "#8b1e3f",
            ),
            metric_card(
                "Worst Network",
                f"nw{int(worst_network['network'])}",
                f"severity {format_metric(worst_network['max_row_severity'], 3)}",
                "#4c956c",
            ),
            metric_card(
                "Step-Masked Rows",
                format_count(scan_view["step_mask_applied"].sum()),
                "within the selected obsnum",
                "#355070",
            ),
            metric_card(
                "Impulsive-Masked Rows",
                format_count(scan_view["impulsive_mask_applied"].sum()),
                "within the selected obsnum",
                "#d76a03",
            ),
            metric_card(
                "Strongest Slot",
                format_metric(top_slot["event_score"], 1) if top_slot is not None else "n/a",
                (
                    f"nw{int(top_slot['network'])} scan {int(top_slot['output_scan_index'])} "
                    f"{top_slot['event_kind_label']}"
                ) if top_slot is not None else "no stored slot in selected network",
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
            html.Summary("Exact Rows"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Obsnum-network rows", style={"fontWeight": "bold", "marginBottom": "8px"}),
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
                            html.Div("Top scan/network rows", style={"fontWeight": "bold", "marginBottom": "8px"}),
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
                            html.Div("Top stored slots for the selected network", style={"fontWeight": "bold", "marginBottom": "8px"}),
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

- Start with the **overview cards** and **ranking panels** to find the worst obsnums and networks.
- Use the **heatmap** to see whether a problem is isolated to a few output scans or spread across an obsnum.
- Use the **trend plot** to see which metric is actually driving a suspicious network.
- Use the **scan ranking** and **slot ranking** panels to jump from broad survey context to concrete events.
- Open **Exact Rows** only when you need precise values for a small number of rows.

`row_severity` is a ranking score, not a physical unit. It is the largest of several normalized contamination indicators, so it should be used to rank and compare rows, not as a calibrated threshold by itself.
                        """
                    ),
                ],
                style={**HELP_BOX_STYLE, "marginBottom": "18px"},
                open=True,
            ),
            html.Div(build_overview_cards(obs_df, by_network_df, scan_df, slot_df), style=PANEL_STYLE),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Obsnum Triage", style={"marginTop": "0"}),
                            html.Div(
                                "Worst obsnums by maximum scan/network severity. Bar color tracks how often the impulsive mask actually acted.",
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
                                "Worst networks across the selected reduction. Bar color tracks total impulsive-masked rows.",
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
                            html.Label("Obsnum"),
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
                "Selected Obsnum Views",
                """
The next panels are for one selected obsnum and one selected network.

- **Heatmap** answers: where are the bad scan/network rows?
- **Trend plot** answers: is this network step-like, impulsive, or both?
- **Network ranking** answers: which networks dominate this obsnum?
- **Scan ranking** answers: which exact rows deserve inspection first?
- **Slot ranking** answers: what are the strongest captured compact events in the selected network?
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
