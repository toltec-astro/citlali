#!/usr/bin/env python3
"""Interactive engineering dashboard for rtcdiag, ptcdiag, and mapdiag products."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dash import Dash, Input, Output, dcc, html
except ModuleNotFoundError as exc:
    raise SystemExit(
        "dash is not installed in ~/toltec. Run '~/toltec/bin/pip install --upgrade dash'."
    ) from exc

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THIS_DIR = Path(__file__).resolve().parent
BLANK_SKY_DIR = THIS_DIR.parent / "blank_sky"
for path in (THIS_DIR, BLANK_SKY_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import mapdiag_data
import ptcdiag_data
import rtcdiag_dash_app as rtc_dash
import rtcdiag_data


GRAPH_CONFIG = {"displaylogo": False, "responsive": True}
PTC_HEATMAP_SCALE = "YlOrRd"
MAP_STAGE_COLORS = {
    "raw_obs": "#577590",
    "filtered_obs": "#43aa8b",
    "raw_coadd": "#f8961e",
    "filtered_coadd": "#bc4749",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redu-dir", required=True, help="Reduction directory, e.g. reduced/redu40")
    parser.add_argument("--array", default="a1100", choices=["a1100", "a1400", "a2000"])
    parser.add_argument("--networks", default="all")
    parser.add_argument("--obsnums", default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
        plot_bgcolor="#fffdf8",
        paper_bgcolor="#fffdf8",
    )
    return fig


def load_bundle(args: argparse.Namespace) -> dict[str, object]:
    bundle: dict[str, object] = {}

    try:
        rtc_data = rtcdiag_data.load_reduction_tables(
            Path(args.redu_dir),
            array=args.array,
            networks_spec=args.networks,
            obsnums_spec=args.obsnums,
        )
        bundle["rtc"] = {
            "error": None,
            "data": rtc_data,
            "obs_df": pd.DataFrame(rtc_data["obs_rows"]).sort_values(
                ["max_row_severity", "obsnum"], ascending=[False, True]
            ),
            "obs_network_df": pd.DataFrame(rtc_data["obs_network_rows"]).sort_values(
                ["obsnum", "network"], ascending=[True, True]
            ),
            "scan_df": pd.DataFrame(rtc_data["scan_network_rows"]).sort_values(
                ["obsnum", "output_scan_index", "network"], ascending=[True, True, True]
            ),
            "slot_df": pd.DataFrame(rtc_data["slot_rows"]).sort_values(
                ["obsnum", "event_score"], ascending=[True, False]
            ),
            "by_network_df": pd.DataFrame(rtc_data["by_network_rows"]).sort_values(
                ["max_row_severity", "network"], ascending=[False, True]
            ),
        }
    except Exception as exc:
        bundle["rtc"] = {"error": str(exc)}

    try:
        ptc_data = ptcdiag_data.load_reduction_tables(
            Path(args.redu_dir),
            array=args.array,
            networks_spec=args.networks,
            obsnums_spec=args.obsnums,
        )
        bundle["ptc"] = {
            "error": None,
            "data": ptc_data,
            "obs_df": pd.DataFrame(ptc_data["obs_rows"]).sort_values(
                ["max_ptc_severity", "obsnum"], ascending=[False, True]
            ),
            "scan_df": pd.DataFrame(ptc_data["scan_network_rows"]).sort_values(
                ["obsnum", "output_scan_index", "network"], ascending=[True, True, True]
            ),
            "by_network_df": pd.DataFrame(ptc_data["by_network_rows"]).sort_values(
                ["max_ptc_severity", "network"], ascending=[False, True]
            ),
        }
    except Exception as exc:
        bundle["ptc"] = {"error": str(exc)}

    try:
        map_data = mapdiag_data.load_reduction_tables(
            Path(args.redu_dir),
            array=args.array,
            obsnums_spec=args.obsnums,
        )
        bundle["map"] = {
            "error": None,
            "data": map_data,
            "map_df": pd.DataFrame(map_data["map_rows"]).sort_values(
                ["is_coadd", "stage", "map_selector", "obs_context"], ascending=[False, True, True, True]
            ),
            "contrib_df": pd.DataFrame(map_data["contribution_rows"]).sort_values(
                ["stage", "map_selector", "core_weight_frac"], ascending=[True, True, False]
            ),
        }
    except Exception as exc:
        bundle["map"] = {"error": str(exc)}

    return bundle


def info_box(message: str) -> html.Div:
    return html.Div(message, style=rtc_dash.HELP_BOX_STYLE)


def style_figure(fig: go.Figure) -> go.Figure:
    return rtc_dash.style_figure(fig)


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(number):
        return float("nan")
    return number


def _format_number(value: object, ndigits: int = 2) -> str:
    number = _safe_float(value)
    if not pd.notna(number):
        return "n/a"
    return f"{number:.{ndigits}f}"


def _max_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.max()) if values.notna().any() else float("nan")


def _map_peak_series(frame: pd.DataFrame) -> pd.Series:
    if "core_peak_abs_sig2noise" in frame.columns:
        core = pd.to_numeric(frame["core_peak_abs_sig2noise"], errors="coerce")
        raw = pd.to_numeric(frame["peak_abs_sig2noise"], errors="coerce")
        return core.where(core.notna(), raw)
    return pd.to_numeric(frame["peak_abs_sig2noise"], errors="coerce")


def _format_percent(value: object, ndigits: int = 2) -> str:
    number = _safe_float(value)
    if not pd.notna(number):
        return "n/a"
    return f"{number * 100.0:.{ndigits}f}%"


def _small_percent_axis_limit(values: list[float]) -> float:
    finite = [float(v) for v in values if pd.notna(v)]
    if not finite:
        return 0.2
    max_value = max(finite)
    if max_value <= 0.05:
        return 0.05
    if max_value <= 0.10:
        return 0.10
    if max_value <= 0.20:
        return 0.20
    if max_value <= 0.50:
        return 0.50
    if max_value <= 1.00:
        return 1.00
    return max_value * 1.20


def build_ptc_obs_rank_figure(obs_df: pd.DataFrame) -> go.Figure:
    if obs_df.empty:
        return empty_figure("No PTC observation summaries")
    view = obs_df.sort_values("max_newly_flagged_fraction", ascending=True).tail(18).copy()
    x_max = _small_percent_axis_limit(list(view["max_newly_flagged_fraction"] * 100.0))
    fig = go.Figure(
        go.Scatter(
            x=view["max_newly_flagged_fraction"] * 100.0,
            y=view["obsnum"],
            mode="markers",
            marker={"color": "#d76a03", "size": 14, "symbol": "diamond", "line": {"color": "#ab5300", "width": 1}},
            customdata=view[
                [
                    "n_rows_with_new_flags",
                    "max_top_event_score",
                    "max_unflagged_residual_z",
                    "max_ptc_severity",
                ]
            ].values,
            hovertemplate=(
                "observation=%{y}<br>largest added-flag fraction=%{x:.3f}%"
                "<br>summaries with added flags=%{customdata[0]:.0f}"
                "<br>highest event score=%{customdata[1]:.2f}"
                "<br>max residual z=%{customdata[2]:.2f}"
                "<br>severity=%{customdata[3]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="PTC observation ranking by largest added-flag fraction",
        xaxis_title="largest added-flag fraction (%)",
        yaxis_title="observation",
        margin={"l": 70, "r": 30, "t": 70, "b": 50},
        xaxis={"range": [0.0, x_max]},
    )
    for xref in (0.05, 0.10, 0.20, 0.50, 1.0):
        if xref > x_max:
            continue
        fig.add_vline(x=xref, line_width=1, line_dash="dot", line_color="#b8b1a0")
    return style_figure(fig)


def build_ptc_network_rank_figure(by_network_df: pd.DataFrame) -> go.Figure:
    if by_network_df.empty:
        return empty_figure("No PTC network summaries")
    view = by_network_df.sort_values("max_newly_flagged_fraction", ascending=True).copy()
    x_max = _small_percent_axis_limit(list(view["max_newly_flagged_fraction"] * 100.0))
    fig = go.Figure(
        go.Scatter(
            x=view["max_newly_flagged_fraction"] * 100.0,
            y=[f"nw{int(v)}" for v in view["network"]],
            mode="markers",
            marker={"color": "#d76a03", "size": 14, "symbol": "diamond", "line": {"color": "#ab5300", "width": 1}},
            customdata=view[
                [
                    "n_rows_with_new_flags",
                    "max_top_event_score",
                    "max_unflagged_residual_z",
                    "worst_obsnum",
                ]
            ].values,
            hovertemplate=(
                "network=%{y}<br>largest added-flag fraction=%{x:.3f}%"
                "<br>summaries with added flags=%{customdata[0]:.0f}"
                "<br>highest event score=%{customdata[1]:.2f}"
                "<br>max residual z=%{customdata[2]:.2f}"
                "<br>worst observation=%{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="PTC network ranking by largest added-flag fraction",
        xaxis_title="largest added-flag fraction (%)",
        yaxis_title="network",
        margin={"l": 70, "r": 30, "t": 70, "b": 50},
        xaxis={"range": [0.0, x_max]},
    )
    for xref in (0.05, 0.10, 0.20, 0.50, 1.0):
        if xref > x_max:
            continue
        fig.add_vline(x=xref, line_width=1, line_dash="dot", line_color="#b8b1a0")
    return style_figure(fig)


def build_ptc_heatmap(scan_df: pd.DataFrame, obsnum: str) -> go.Figure:
    obs_df = scan_df.loc[scan_df["obsnum"] == obsnum].copy()
    if obs_df.empty:
        return empty_figure(f"No PTC timechunk/network summaries for observation {obsnum}")
    obs_df["new_pct"] = pd.to_numeric(obs_df["newly_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    obs_df["label"] = obs_df.apply(
        lambda row: (
            f"timechunk={int(row['output_scan_index'])}"
            f"<br>nw={int(row['network'])}"
            f"<br>added flags={_format_percent(row['newly_flagged_fraction'], 3)}"
            f"<br>proposed flags={_format_percent(row['proposed_flagged_fraction'], 3)}"
            f"<br>event score={_format_number(row['top_event_score'], 2)}"
            f"<br>max residual z={_format_number(row['max_unflagged_residual_z'], 2)}"
            f"<br>busy-row suppression factor={_format_number(row['busy_row_factor'], 3)}"
            f"<br>severity={_format_number(row['ptc_severity'], 3)}"
        ),
        axis=1,
    )
    pivot = obs_df.pivot(index="network", columns="output_scan_index", values="new_pct").sort_index()
    hover = obs_df.pivot(index="network", columns="output_scan_index", values="label").sort_index()
    zmax = max(float(pivot.max().max()) if pivot.notna().any().any() else 0.0, 0.20)
    fig = go.Figure(
        go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.values,
            text=hover.values,
            hovertemplate="%{text}<extra></extra>",
            colorscale=PTC_HEATMAP_SCALE,
            zmin=0.0,
            zmax=zmax,
            colorbar={"title": "added %"},
        )
    )
    fig.update_layout(
        title=f"Observation {obsnum}: PTC added-flag fraction by timechunk/network",
        xaxis_title="timechunk",
        yaxis_title="network",
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
    )
    return style_figure(fig)


def build_ptc_network_trend(scan_df: pd.DataFrame, obsnum: str, network: int) -> go.Figure:
    df = scan_df.loc[(scan_df["obsnum"] == obsnum) & (scan_df["network"] == network)].copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=("Added-flag fraction (%)", "Supporting diagnostics"),
    )
    if df.empty:
        fig.update_layout(title=f"No PTC timechunk summaries for observation {obsnum} network {network}")
        return style_figure(fig)
    x = df["output_scan_index"]
    new_pct = pd.to_numeric(df["newly_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    prop_pct = pd.to_numeric(df["proposed_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    busy_deficit_pct = (1.0 - pd.to_numeric(df["busy_row_factor"], errors="coerce").fillna(1.0)).clip(lower=0.0) * 100.0
    resid_z = pd.to_numeric(df["max_unflagged_residual_z"], errors="coerce")
    top_event = pd.to_numeric(df["top_event_score"], errors="coerce")
    pct_max = _small_percent_axis_limit(list(new_pct) + list(prop_pct))
    score_max = max(
        float(top_event.max()) if top_event.notna().any() else 0.0,
        float(resid_z.max()) if resid_z.notna().any() else 0.0,
        1.0,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=new_pct,
            name="added flags",
            marker={"color": "#d76a03"},
            opacity=0.80,
            hovertemplate="timechunk=%{x}<br>added flags=%{y:.3f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prop_pct,
            name="proposed flags",
            mode="lines+markers",
            line={"color": "#8b1e3f", "width": 2},
            hovertemplate="timechunk=%{x}<br>proposed flags=%{y:.3f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=busy_deficit_pct,
            name="busy-row deficit",
            mode="lines",
            line={"color": "#7b2cbf", "width": 2},
            hovertemplate="timechunk=%{x}<br>busy-row deficit=%{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=top_event,
            name="event score",
            mode="lines+markers",
            line={"color": "#d76a03", "width": 2},
            hovertemplate="timechunk=%{x}<br>event score=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=resid_z,
            name="max residual z",
            mode="lines",
            line={"color": "#355070", "width": 2},
            hovertemplate="timechunk=%{x}<br>max residual z=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Observation {obsnum} network {network}: PTC diagnostics",
        margin={"l": 60, "r": 70, "t": 84, "b": 90},
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "x": 0, "xanchor": "left"},
        barmode="overlay",
        height=820,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#efeadd", row=1, col=1)
    fig.update_xaxes(title_text="timechunk", showgrid=True, gridcolor="#e6e0d2", row=2, col=1)
    fig.update_yaxes(title_text="added / proposed (%)", range=[0.0, pct_max], showgrid=True, gridcolor="#e6e0d2", row=1, col=1)
    fig.update_yaxes(title_text="busy-row deficit (%)", range=[0.0, 100.0], showgrid=True, gridcolor="#e6e0d2", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="score / residual z", range=[0.0, score_max * 1.10], row=2, col=1, secondary_y=True, showgrid=False)
    return style_figure(fig)


def build_ptc_scan_rank_figure(scan_df: pd.DataFrame, obsnum: str) -> go.Figure:
    view = scan_df.loc[scan_df["obsnum"] == obsnum].copy()
    if view.empty:
        return empty_figure(f"No PTC timechunk summaries for observation {obsnum}")
    view["new_pct"] = pd.to_numeric(view["newly_flagged_fraction"], errors="coerce").fillna(0.0) * 100.0
    view = view.sort_values("new_pct", ascending=False).head(18)
    view["row_label"] = view.apply(lambda row: f"timechunk {int(row['output_scan_index'])} nw{int(row['network'])}", axis=1)
    view = view.sort_values("new_pct", ascending=True)
    x_max = _small_percent_axis_limit(list(view["new_pct"]))
    fig = go.Figure(
        go.Bar(
            x=view["new_pct"],
            y=view["row_label"],
            orientation="h",
            marker={"color": "#d76a03", "line": {"color": "#ab5300", "width": 1}},
            customdata=view[
                [
                    "top_event_score",
                    "max_unflagged_residual_z",
                    "busy_row_factor",
                    "ptc_severity",
                ]
            ].values,
            hovertemplate=(
                "%{y}<br>added flags=%{x:.3f}%"
                "<br>event score=%{customdata[0]:.2f}"
                "<br>max residual z=%{customdata[1]:.2f}"
                "<br>busy-row factor=%{customdata[2]:.3f}"
                "<br>severity=%{customdata[3]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Observation {obsnum}: largest added-flag PTC summaries",
        xaxis_title="added-flag fraction (%)",
        yaxis_title="timechunk / network summary",
        margin={"l": 120, "r": 30, "t": 70, "b": 50},
        xaxis={"range": [0.0, x_max]},
    )
    for xref in (0.05, 0.10, 0.20, 0.50, 1.0):
        if xref > x_max:
            continue
        fig.add_vline(x=xref, line_width=1, line_dash="dot", line_color="#b8b1a0")
    return style_figure(fig)


def build_ptc_detector_figure(det_df: pd.DataFrame, network: int, output_scan_index: int) -> go.Figure:
    if det_df.empty:
        return empty_figure(f"No detector weights for timechunk {output_scan_index}")
    fig = go.Figure()
    for nw, part in det_df.groupby("network"):
        fig.add_trace(
            go.Box(
                y=part["weight"],
                x=[f"nw{int(nw)}"] * len(part),
                name=f"nw{int(nw)}",
                boxpoints=False,
                marker_color="#bc4749" if int(nw) == int(network) else "#7c8da0",
            )
        )
    fig.update_layout(
        title=f"Timechunk {output_scan_index}: detector weight distribution by network",
        xaxis_title="network",
        yaxis_title="detector weight",
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
        showlegend=False,
    )
    return style_figure(fig)


def build_ptc_overview_cards(obs_df: pd.DataFrame, by_network_df: pd.DataFrame, scan_df: pd.DataFrame) -> html.Div:
    new_vals = pd.to_numeric(scan_df["newly_flagged_fraction"], errors="coerce")
    event_vals = pd.to_numeric(scan_df["top_event_score"], errors="coerce")
    resid_vals = pd.to_numeric(scan_df["max_unflagged_residual_z"], errors="coerce")
    max_new_row = scan_df.loc[new_vals.idxmax()] if new_vals.notna().any() else None
    max_event_row = scan_df.loc[event_vals.idxmax()] if event_vals.notna().any() else None
    n_with_new = int((new_vals.fillna(0.0) > 0.0).sum())
    return html.Div(
        [
            rtc_dash.metric_card("Observations", rtc_dash.format_count(len(obs_df)), "distinct PTC sidecars", "#7c644a"),
            rtc_dash.metric_card("Timechunk-Network Summaries", rtc_dash.format_count(len(scan_df)), "PTC timechunk x network summaries", "#8f5536"),
            rtc_dash.metric_card(
                "Summaries With Added Flags",
                rtc_dash.format_count(n_with_new),
                f"{_format_number(100.0 * n_with_new / max(len(scan_df), 1), 1)}% of summaries",
                "#8b1e3f",
            ),
            rtc_dash.metric_card(
                "Largest Added-Flag Fraction",
                _format_percent(_max_or_nan(scan_df["newly_flagged_fraction"]), 3),
                (
                    f"obs {max_new_row['obsnum']} nw{int(max_new_row['network'])} timechunk {int(max_new_row['output_scan_index'])}"
                    if max_new_row is not None else "no added flags"
                ),
                "#4c956c",
            ),
            rtc_dash.metric_card(
                "Highest Event Score",
                _format_number(_max_or_nan(scan_df["top_event_score"]), 2),
                (
                    f"obs {max_event_row['obsnum']} nw{int(max_event_row['network'])} timechunk {int(max_event_row['output_scan_index'])}"
                    if max_event_row is not None else "no accepted events"
                ),
                "#355070",
            ),
            rtc_dash.metric_card(
                "Max Residual Z",
                _format_number(_max_or_nan(scan_df["max_unflagged_residual_z"]), 2),
                "largest unflagged residual after second pass",
                "#d76a03",
            ),
        ],
        style=rtc_dash.CARD_GRID_STYLE,
    )


def build_ptc_detail_panel(scan_view: pd.DataFrame, det_df: pd.DataFrame) -> html.Details:
    return html.Details(
        [
            html.Summary("Exact Values"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Top PTC timechunk/network summaries", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            rtc_dash.table(
                                [
                                    "output_scan_index",
                                    "network",
                                    "ptc_severity",
                                    "newly_flagged_fraction",
                                    "proposed_flagged_fraction",
                                    "corr_penalty_factor",
                                    "busy_row_factor",
                                    "adaptive_chosen_k",
                                    "top_event_score",
                                ],
                                rtc_dash.rounded_records(
                                    scan_view.sort_values("ptc_severity", ascending=False).head(16),
                                    [
                                        "output_scan_index",
                                        "network",
                                        "ptc_severity",
                                        "newly_flagged_fraction",
                                        "proposed_flagged_fraction",
                                        "corr_penalty_factor",
                                        "busy_row_factor",
                                        "adaptive_chosen_k",
                                        "top_event_score",
                                    ],
                                ),
                                page_size=8,
                            ),
                        ],
                        style=rtc_dash.PANEL_STYLE,
                    ),
                    html.Div(
                        [
                            html.Div("Detector weights in selected timechunk", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            rtc_dash.table(
                                ["uid", "network", "weight", "flagged_fraction", "rms"],
                                rtc_dash.rounded_records(det_df.head(24), ["uid", "network", "weight", "flagged_fraction", "rms"]),
                                page_size=8,
                            ),
                        ],
                        style=rtc_dash.PANEL_STYLE,
                    ),
                ]
            ),
        ],
        open=False,
        style={**rtc_dash.HELP_BOX_STYLE, "marginTop": "6px"},
    )


def build_map_contrib_figure(contrib_df: pd.DataFrame, map_selector: str, stage: str) -> go.Figure:
    view = contrib_df.loc[
        (contrib_df["map_selector"] == map_selector) & (contrib_df["stage"] == stage)
    ].copy()
    view = view.sort_values("core_weight_frac", ascending=True)
    if view.empty:
        return empty_figure(f"No coadd contributions for {map_selector} [{stage}]")
    core_pct = view["core_weight_frac"] * 100.0
    whole_pct = view["weight_frac"] * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=core_pct,
            y=view["contrib_obsnum"],
            orientation="h",
            name="core contribution",
            marker={"color": "#bc4749"},
            customdata=view[["weight_frac", "core_weight_sum", "dateobs"]].values,
            hovertemplate=(
                "observation=%{y}<br>core contribution=%{x:.1f}%"
                "<br>whole-map contribution=%{customdata[0]:.1%}"
                "<br>core weight sum=%{customdata[1]:.4g}"
                "<br>dateobs=%{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=whole_pct,
            y=view["contrib_obsnum"],
            mode="markers",
            name="whole-map contribution",
            marker={"size": 10, "color": "#355070", "symbol": "circle-open", "line": {"width": 2}},
            hovertemplate="observation=%{y}<br>whole-map contribution=%{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{map_selector} [{stage}]: coadd observation contribution",
        xaxis_title="coadd contribution (%)",
        yaxis_title="observation",
        margin={"l": 100, "r": 30, "t": 70, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(ticksuffix="%")
    return style_figure(fig)


def build_map_obs_rank_figure(map_df: pd.DataFrame, contrib_df: pd.DataFrame, map_selector: str, stage: str) -> go.Figure:
    view = map_df.loc[(map_df["map_selector"] == map_selector) & (map_df["is_coadd"] == 0)].copy()
    if view.empty:
        return empty_figure(f"No observation-level map diagnostics for {map_selector}")
    contrib_view = contrib_df.loc[
        (contrib_df["map_selector"] == map_selector) & (contrib_df["stage"] == stage),
        ["contrib_obsnum", "core_weight_frac", "weight_frac", "dateobs"],
    ].copy()
    view = view.merge(contrib_view, how="left", left_on="obs_context", right_on="contrib_obsnum")
    view["core_weight_frac"] = view["core_weight_frac"].fillna(0.0)
    view["weight_frac"] = view["weight_frac"].fillna(0.0)
    view["display_peak_abs_sig2noise"] = _map_peak_series(view)
    view = view.sort_values("display_peak_abs_sig2noise", ascending=True)
    color_series = view["core_weight_frac"] * 100.0
    fig = go.Figure(
        go.Bar(
            x=view["display_peak_abs_sig2noise"],
            y=view["obs_context"],
            orientation="h",
            marker={
                "color": color_series,
                "colorscale": "Teal",
                "line": {"color": "#355070", "width": 1},
                "colorbar": {"title": "core contrib %"},
                "cmin": 0,
                "cmax": max(float(color_series.max()) if not color_series.empty else 0.0, 35.0),
            },
            customdata=view[["core_weight_frac", "weight_frac", "n_core_pixels", "core_coverage_median"]].values,
            hovertemplate=(
                "observation=%{y}<br>core peak |S/N|=%{x:.2f}"
                "<br>core contribution=%{customdata[0]:.1%}"
                "<br>whole-map contribution=%{customdata[1]:.1%}"
                "<br>n_core_pixels=%{customdata[2]:.0f}"
                "<br>core coverage median=%{customdata[3]:.3g}<extra></extra>"
            ),
        )
    )
    coadd_rows = map_df.loc[
        (map_df["map_selector"] == map_selector) & (map_df["is_coadd"] == 1) & (map_df["stage"] == stage)
    ]
    if not coadd_rows.empty:
        coadd_peak = float(_map_peak_series(coadd_rows).iloc[0])
        fig.add_vline(
            x=coadd_peak,
            line={"color": "#bc4749", "width": 2, "dash": "dash"},
            annotation_text=f"coadd {coadd_peak:.2f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title=f"{map_selector}: observation map ranking by core peak |S/N|",
        xaxis_title="core peak |signal-to-noise|",
        yaxis_title="observation",
        margin={"l": 90, "r": 40, "t": 70, "b": 60},
    )
    return style_figure(fig)


def build_map_contrib_scatter_figure(map_df: pd.DataFrame, contrib_df: pd.DataFrame, map_selector: str, stage: str) -> go.Figure:
    view = map_df.loc[(map_df["map_selector"] == map_selector) & (map_df["is_coadd"] == 0)].copy()
    contrib_view = contrib_df.loc[
        (contrib_df["map_selector"] == map_selector) & (contrib_df["stage"] == stage),
        ["contrib_obsnum", "core_weight_frac", "weight_frac", "dateobs"],
    ].copy()
    if view.empty or contrib_view.empty:
        return empty_figure(f"No joined coadd/observation rows for {map_selector} [{stage}]")
    view = view.merge(contrib_view, how="inner", left_on="obs_context", right_on="contrib_obsnum")
    if view.empty:
        return empty_figure(f"No joined coadd/observation rows for {map_selector} [{stage}]")
    view["display_peak_abs_sig2noise"] = _map_peak_series(view)
    fig = go.Figure(
        go.Scatter(
            x=view["core_weight_frac"] * 100.0,
            y=view["display_peak_abs_sig2noise"],
            mode="markers+text",
            text=view["obs_context"],
            textposition="top center",
            marker={
                "size": 14,
                "color": view["core_coverage_median"],
                "colorscale": "Teal",
                "line": {"color": "#355070", "width": 1},
                "colorbar": {"title": "core cov"},
            },
            customdata=view[["weight_frac", "n_core_pixels", "dateobs"]].values,
            hovertemplate=(
                "observation=%{text}<br>core contribution=%{x:.1f}%"
                "<br>core peak |S/N|=%{y:.2f}"
                "<br>whole-map contribution=%{customdata[0]:.1%}"
                "<br>n_core_pixels=%{customdata[1]:.0f}"
                "<br>dateobs=%{customdata[2]}<extra></extra>"
            ),
        )
    )
    coadd_rows = map_df.loc[
        (map_df["map_selector"] == map_selector) & (map_df["is_coadd"] == 1) & (map_df["stage"] == stage)
    ]
    if not coadd_rows.empty:
        fig.add_hline(
            y=float(_map_peak_series(coadd_rows).iloc[0]),
            line={"color": "#bc4749", "width": 2, "dash": "dash"},
            annotation_text=f"coadd core peak {_format_number(_map_peak_series(coadd_rows).iloc[0], 2)}",
            annotation_position="top right",
        )
    fig.update_layout(
        title=f"{map_selector} [{stage}]: contribution versus observation core peak",
        xaxis_title="core coadd contribution (%)",
        yaxis_title="observation core peak |signal-to-noise|",
        margin={"l": 70, "r": 70, "t": 70, "b": 60},
    )
    fig.update_xaxes(ticksuffix="%")
    return style_figure(fig)


def build_map_overview_cards(map_df: pd.DataFrame, contrib_df: pd.DataFrame) -> html.Div:
    coadd_df = map_df.loc[map_df["is_coadd"] == 1].copy()
    obs_df = map_df.loc[map_df["is_coadd"] == 0].copy()
    if not obs_df.empty:
        obs_df = obs_df.assign(display_peak_abs_sig2noise=_map_peak_series(obs_df))
        strongest_obs = obs_df.sort_values("display_peak_abs_sig2noise", ascending=False).iloc[0]
    else:
        all_df = map_df.assign(display_peak_abs_sig2noise=_map_peak_series(map_df))
        strongest_obs = all_df.sort_values("display_peak_abs_sig2noise", ascending=False).iloc[0]
    if not coadd_df.empty:
        coadd_df = coadd_df.assign(display_peak_abs_sig2noise=_map_peak_series(coadd_df))
        strongest_coadd = coadd_df.sort_values("display_peak_abs_sig2noise", ascending=False).iloc[0]
    else:
        strongest_coadd = None
    top_contrib = contrib_df.sort_values("core_weight_frac", ascending=False).iloc[0] if not contrib_df.empty else None
    return html.Div(
        [
            rtc_dash.metric_card("Map Summaries", rtc_dash.format_count(len(map_df)), "selected array map summaries", "#7c644a"),
            rtc_dash.metric_card("Observation Maps", rtc_dash.format_count(len(obs_df)), "observation-level map rows", "#8f5536"),
            rtc_dash.metric_card("Coadd Maps", rtc_dash.format_count(len(coadd_df)), "coadd map rows", "#8b1e3f"),
            rtc_dash.metric_card(
                "Highest Core Peak",
                str(strongest_obs["obs_context"]),
                f"peak |S/N| {_format_number(strongest_obs['display_peak_abs_sig2noise'], 2)} inside the weight cut",
                "#4c956c",
            ),
            rtc_dash.metric_card(
                "Top Coadd Contributor",
                str(top_contrib["contrib_obsnum"]) if top_contrib is not None else "n/a",
                f"core contribution {_format_percent(top_contrib['core_weight_frac'], 1)}" if top_contrib is not None else "no coadd table",
                "#355070",
            ),
            rtc_dash.metric_card(
                "Coadd Core Peak |S/N|",
                _format_number(strongest_coadd["display_peak_abs_sig2noise"], 2) if strongest_coadd is not None else "n/a",
                strongest_coadd["stage"] if strongest_coadd is not None else "no coadd row",
                "#c97b00",
            ),
        ],
        style=rtc_dash.CARD_GRID_STYLE,
    )


def build_map_detail_panel(map_df: pd.DataFrame, contrib_df: pd.DataFrame, map_selector: str, stage: str) -> html.Details:
    stage_rows = map_df.loc[map_df["map_selector"] == map_selector].copy()
    contrib_rows = contrib_df.loc[
        (contrib_df["map_selector"] == map_selector) & (contrib_df["stage"] == stage)
    ].copy()
    return html.Details(
        [
            html.Summary("Exact Values"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Stage rows", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            rtc_dash.table(
                                [
                                    "obs_context",
                                    "stage",
                                    "peak_abs_sig2noise",
                                    "median_rms",
                                    "n_core_pixels",
                                    "weight_threshold",
                                ],
                                rtc_dash.rounded_records(
                                    stage_rows,
                                    [
                                        "obs_context",
                                        "stage",
                                        "peak_abs_sig2noise",
                                        "median_rms",
                                        "n_core_pixels",
                                        "weight_threshold",
                                    ],
                                ),
                                page_size=8,
                            ),
                        ],
                        style=rtc_dash.PANEL_STYLE,
                    ),
                    html.Div(
                        [
                            html.Div("Coadd contribution rows", style={"fontWeight": "bold", "marginBottom": "8px"}),
                            rtc_dash.table(
                                ["contrib_obsnum", "core_weight_frac", "weight_frac", "core_weight_sum", "dateobs"],
                                rtc_dash.rounded_records(
                                    contrib_rows.head(24),
                                    ["contrib_obsnum", "core_weight_frac", "weight_frac", "core_weight_sum", "dateobs"],
                                ),
                                page_size=8,
                            ),
                        ],
                        style=rtc_dash.PANEL_STYLE,
                    ),
                ]
            ),
        ],
        open=False,
        style={**rtc_dash.HELP_BOX_STYLE, "marginTop": "6px"},
    )


def build_rtc_tab(section: dict[str, object]) -> html.Div:
    if section["error"] is not None:
        return info_box(f"RTC unavailable: {section['error']}")
    obs_df = section["obs_df"]
    obs_network_df = section["obs_network_df"]
    scan_df = section["scan_df"]
    slot_df = section["slot_df"]
    by_network_df = section["by_network_df"]
    data = section["data"]
    obs_options = [{"label": obs, "value": obs} for obs in obs_df["obsnum"].astype(str).tolist()]
    default_obs = obs_options[0]["value"]
    default_networks = obs_network_df.loc[obs_network_df["obsnum"] == default_obs, "network"].astype(int).tolist()
    default_network = default_networks[0] if default_networks else int(data["selected_networks"][0])
    return html.Div(
        [
            rtc_dash.section_help(
                "RTC",
                "Plot-first RTC triage. Start with severity ranks, then use the selected observation plots to decide whether a problem is step-like, impulsive, or both.",
            ),
            html.Div(rtc_dash.build_overview_cards(obs_df, by_network_df, scan_df, slot_df), style=rtc_dash.PANEL_STYLE),
            html.Div(
                [
                    html.Div(dcc.Graph(figure=rtc_dash.build_obs_rank_figure(obs_df), config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                    html.Div(dcc.Graph(figure=rtc_dash.build_network_rank_figure(by_network_df), config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div([html.Label("Observation"), dcc.Dropdown(id="rtc-obs-dropdown", options=obs_options, value=default_obs, clearable=False)], style={"width": "240px"}),
                    html.Div([html.Label("Network"), dcc.Dropdown(id="rtc-network-dropdown", value=default_network, clearable=False)], style={"width": "180px"}),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
            ),
            html.Div(id="rtc-selected-cards", style=rtc_dash.PANEL_STYLE),
            html.Div(
                [
                    html.Div(dcc.Graph(id="rtc-heatmap", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                    html.Div(dcc.Graph(id="rtc-network-trend", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="rtc-obs-network-rank", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                    html.Div(dcc.Graph(id="rtc-top-scan-rank", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div([html.Div(dcc.Graph(id="rtc-top-slot-rank", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE)]),
            html.Div(id="rtc-detail-panel"),
        ]
    )


def build_ptc_tab(section: dict[str, object]) -> html.Div:
    if section["error"] is not None:
        return info_box(f"PTC unavailable: {section['error']}")
    obs_df = section["obs_df"]
    scan_df = section["scan_df"]
    by_network_df = section["by_network_df"]
    obs_options = [{"label": obs, "value": obs} for obs in obs_df["obsnum"].astype(str).tolist()]
    default_obs = obs_options[0]["value"]
    default_networks = scan_df.loc[scan_df["obsnum"] == default_obs, "network"].astype(int).drop_duplicates().tolist()
    default_network = default_networks[0]
    default_scans = scan_df.loc[scan_df["obsnum"] == default_obs, "output_scan_index"].astype(int).drop_duplicates().tolist()
    default_scan = default_scans[0]
    return html.Div(
        [
            rtc_dash.section_help(
                "PTC",
                "Use the heatmap and rankings to see where PTC is actually adding flags. The most useful quick indicators here are added-flag fraction, top accepted event score, and maximum unflagged residual z. Treat the legacy severity score as secondary context only.",
            ),
            html.Div(build_ptc_overview_cards(obs_df, by_network_df, scan_df), style=rtc_dash.PANEL_STYLE),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(figure=build_ptc_obs_rank_figure(obs_df), config=GRAPH_CONFIG),
                        style={**rtc_dash.PANEL_STYLE, "overflow": "hidden"},
                    ),
                    html.Div(
                        dcc.Graph(figure=build_ptc_network_rank_figure(by_network_df), config=GRAPH_CONFIG),
                        style={**rtc_dash.PANEL_STYLE, "overflow": "hidden"},
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div([html.Label("Observation"), dcc.Dropdown(id="ptc-obs-dropdown", options=obs_options, value=default_obs, clearable=False)], style={"width": "240px"}),
                    html.Div([html.Label("Network"), dcc.Dropdown(id="ptc-network-dropdown", value=default_network, clearable=False)], style={"width": "180px"}),
                    html.Div([html.Label("Timechunk"), dcc.Dropdown(id="ptc-scan-dropdown", value=default_scan, clearable=False)], style={"width": "180px"}),
                ],
                style={
                    **rtc_dash.PANEL_STYLE,
                    "display": "flex",
                    "gap": "16px",
                    "marginTop": "8px",
                    "marginBottom": "16px",
                    "flexWrap": "wrap",
                    "position": "relative",
                    "zIndex": 1,
                },
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="ptc-heatmap", config=GRAPH_CONFIG), style={**rtc_dash.PANEL_STYLE, "overflow": "hidden"}),
                    html.Div(dcc.Graph(id="ptc-network-trend", config=GRAPH_CONFIG), style={**rtc_dash.PANEL_STYLE, "overflow": "hidden"}),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="ptc-scan-rank", config=GRAPH_CONFIG), style={**rtc_dash.PANEL_STYLE, "overflow": "hidden"}),
                    html.Div(dcc.Graph(id="ptc-detector-figure", config=GRAPH_CONFIG), style={**rtc_dash.PANEL_STYLE, "overflow": "hidden"}),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(id="ptc-detail-panel"),
        ]
    )


def build_map_tab(section: dict[str, object]) -> html.Div:
    if section["error"] is not None:
        return info_box(f"Map diagnostics unavailable: {section['error']}")
    map_df = section["map_df"]
    contrib_df = section["contrib_df"]
    map_selectors = sorted(map_df["map_selector"].drop_duplicates().tolist())
    default_selector = map_selectors[0]
    coadd_stage_options = sorted(
        contrib_df.loc[contrib_df["map_selector"] == default_selector, "stage"].drop_duplicates().tolist()
    )
    default_stage = coadd_stage_options[0] if coadd_stage_options else "raw_coadd"
    return html.Div(
        [
            rtc_dash.section_help(
                "Maps",
                "This view is for fast coadd sanity checks. Start with the coadd contribution chart to see which observations dominate the coadd core, then compare those contributions with the individual observation map peaks. All map peak values shown here are core-restricted: they are evaluated only over pixels that satisfy the map weight cut. Coadd contribution is based on inverse-variance map weight over the coadd core, not on which observation has the highest peak. If there is only one coadd stage, the dashboard avoids stage-comparison plots and focuses on observation-versus-coadd relationships instead.",
            ),
            html.Div(build_map_overview_cards(map_df, contrib_df), style=rtc_dash.PANEL_STYLE),
            html.Div(
                [
                    html.Div([html.Label("Map"), dcc.Dropdown(id="map-selector-dropdown", options=[{"label": value, "value": value} for value in map_selectors], value=default_selector, clearable=False)], style={"minWidth": "260px", "flex": "1 1 260px"}),
                    html.Div([html.Label("Coadd Stage"), dcc.Dropdown(id="map-stage-dropdown", value=default_stage, clearable=False)], style={"width": "220px"}),
                ],
                style={**rtc_dash.PANEL_STYLE, "display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="map-contrib-figure", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                    html.Div(dcc.Graph(id="map-obs-rank-figure", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "16px"},
            ),
            html.Div(
                [html.Div(dcc.Graph(id="map-scatter-figure", config=GRAPH_CONFIG), style=rtc_dash.PANEL_STYLE)],
                style={"display": "grid", "gridTemplateColumns": "minmax(520px, 1fr)", "gap": "16px"},
            ),
            html.Div(id="map-detail-panel"),
        ]
    )


def build_app(args: argparse.Namespace) -> Dash:
    bundle = load_bundle(args)
    app = Dash(__name__)
    app.title = "citlali diagnostics dashboard"
    app.layout = html.Div(
        [
            html.H2("Citlali Diagnostics Dashboard", style={"marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(f"Reduction: {Path(args.redu_dir).expanduser().resolve()}"),
                    html.Div(f"Array: {args.array}"),
                ],
                style={"marginBottom": "16px", "fontFamily": "Menlo, Monaco, Consolas, monospace", "fontSize": "13px"},
            ),
            html.Details(
                [
                    html.Summary("How To Read This Dashboard"),
                    dcc.Markdown(
                        """
This dashboard is built for quick human triage.

- Start with the plots, not the tables.
- Use color and shape to decide whether an observation, timechunk, network, or map stage looks normal.
- Open the exact values only after a plot tells you where to dig.
- The PTC and map tabs are intentionally early-stage and should be treated as engineering views, not frozen reporting products.

Definitions:

- **event**: a localized contamination episode found in one detector timestream. In RTC and PTC views, stored events are compact examples of the strongest such transients.

    - **impulsive**: a brief spike-like event concentrated in time. In practice this includes both narrow cosmic-ray-like hits and some bursty RFI-like events when they are compact enough to trigger the impulsive finders.
    - **step**: a level-shift event with a persistent baseline change after the transition, unlike a brief impulse.

- **severity**: a dimensionless ranking score used to decide where to look first. Around `1` means at least one reference contamination threshold was reached. Values below `1` are usually quieter; values above `1` deserve progressively closer inspection. It is not the keep-or-reject metric by itself.
- **robust sigma units**: the excursion size after dividing by a robust scatter estimate rather than an ordinary RMS standard deviation.
- **data loss**: for RTC especially, the more practical question is how much data the masks are actually removing. A few percent flagged can be acceptable; tens of percent may make an observation questionable. The RTC cards report the mean flagged fraction over the timechunk-network summaries where that mask actually fired, not the total fraction of the whole observation that was lost.
- **strongest captured event score**: in RTC, this is the largest stored event score, in robust-sigma units. For a raw-like event it is `|x-center|/sigma_robust`; for a delta-like event it is `|delta_x-median(delta_x)|/sigma_delta,robust`. It is useful for ranking how extreme transients are, not as a calibrated physical unit.
                        """
                    ),
                ],
                style={**rtc_dash.HELP_BOX_STYLE, "marginBottom": "18px"},
                open=False,
            ),
            dcc.Tabs(
                [
                    dcc.Tab(label="RTC", children=[build_rtc_tab(bundle["rtc"])], style={"padding": "12px"}, selected_style={"padding": "12px", "fontWeight": "bold"}),
                    dcc.Tab(label="PTC", children=[build_ptc_tab(bundle["ptc"])], style={"padding": "12px"}, selected_style={"padding": "12px", "fontWeight": "bold"}),
                    dcc.Tab(label="Maps", children=[build_map_tab(bundle["map"])], style={"padding": "12px"}, selected_style={"padding": "12px", "fontWeight": "bold"}),
                ]
            ),
        ],
        style=rtc_dash.PAGE_STYLE,
    )

    rtc_section = bundle["rtc"]
    if rtc_section["error"] is None:
        rtc_obs_network_df = rtc_section["obs_network_df"]
        rtc_scan_df = rtc_section["scan_df"]
        rtc_slot_df = rtc_section["slot_df"]

        @app.callback(Output("rtc-network-dropdown", "options"), Output("rtc-network-dropdown", "value"), Input("rtc-obs-dropdown", "value"))
        def update_rtc_network_options(obsnum: str):
            networks = rtc_obs_network_df.loc[rtc_obs_network_df["obsnum"] == obsnum, "network"].astype(int).tolist()
            options = [{"label": str(nw), "value": int(nw)} for nw in networks]
            value = options[0]["value"] if options else None
            return options, value

        @app.callback(
            Output("rtc-selected-cards", "children"),
            Output("rtc-heatmap", "figure"),
            Output("rtc-network-trend", "figure"),
            Output("rtc-obs-network-rank", "figure"),
            Output("rtc-top-scan-rank", "figure"),
            Output("rtc-top-slot-rank", "figure"),
            Output("rtc-detail-panel", "children"),
            Input("rtc-obs-dropdown", "value"),
            Input("rtc-network-dropdown", "value"),
        )
        def update_rtc_detail(obsnum: str, network: int):
            obs_network_view = rtc_obs_network_df.loc[rtc_obs_network_df["obsnum"] == obsnum].copy()
            scan_view = rtc_scan_df.loc[rtc_scan_df["obsnum"] == obsnum].copy()
            slot_view = rtc_slot_df.loc[(rtc_slot_df["obsnum"] == obsnum) & (rtc_slot_df["network"] == network)].copy()
            return (
                rtc_dash.build_selected_obs_cards(obsnum, obs_network_view, scan_view, slot_view),
                rtc_dash.build_heatmap(rtc_scan_df, obsnum),
                rtc_dash.build_network_trend(rtc_scan_df, obsnum, int(network)),
                rtc_dash.build_obs_network_rank_figure(obs_network_view, obsnum),
                rtc_dash.build_top_scan_figure(scan_view, obsnum),
                rtc_dash.build_top_slot_figure(slot_view, obsnum, int(network)),
                rtc_dash.build_detail_panel(obs_network_view, scan_view, slot_view),
            )

    ptc_section = bundle["ptc"]
    if ptc_section["error"] is None:
        ptc_scan_df = ptc_section["scan_df"]
        product_paths = ptc_section["data"]["product_paths"]

        @app.callback(Output("ptc-network-dropdown", "options"), Output("ptc-network-dropdown", "value"), Input("ptc-obs-dropdown", "value"))
        def update_ptc_network_options(obsnum: str):
            networks = ptc_scan_df.loc[ptc_scan_df["obsnum"] == obsnum, "network"].astype(int).drop_duplicates().tolist()
            options = [{"label": str(nw), "value": int(nw)} for nw in networks]
            value = options[0]["value"] if options else None
            return options, value

        @app.callback(Output("ptc-scan-dropdown", "options"), Output("ptc-scan-dropdown", "value"), Input("ptc-obs-dropdown", "value"))
        def update_ptc_scan_options(obsnum: str):
            scans = ptc_scan_df.loc[ptc_scan_df["obsnum"] == obsnum, "output_scan_index"].astype(int).drop_duplicates().tolist()
            options = [{"label": str(scan), "value": int(scan)} for scan in scans]
            value = options[0]["value"] if options else None
            return options, value

        @app.callback(
            Output("ptc-heatmap", "figure"),
            Output("ptc-network-trend", "figure"),
            Output("ptc-scan-rank", "figure"),
            Output("ptc-detector-figure", "figure"),
            Output("ptc-detail-panel", "children"),
            Input("ptc-obs-dropdown", "value"),
            Input("ptc-network-dropdown", "value"),
            Input("ptc-scan-dropdown", "value"),
        )
        def update_ptc_detail(obsnum: str, network: int, output_scan_index: int):
            scan_view = ptc_scan_df.loc[ptc_scan_df["obsnum"] == obsnum].copy()
            det_rows = ptcdiag_data.load_detector_rows(product_paths[obsnum], array=args.array, output_scan_index=int(output_scan_index))
            det_df = pd.DataFrame(det_rows).sort_values(["network", "uid"], ascending=[True, True]) if det_rows else pd.DataFrame(columns=["uid", "network", "weight", "flagged_fraction", "rms"])
            return (
                build_ptc_heatmap(ptc_scan_df, obsnum),
                build_ptc_network_trend(ptc_scan_df, obsnum, int(network)),
                build_ptc_scan_rank_figure(ptc_scan_df, obsnum),
                build_ptc_detector_figure(det_df, int(network), int(output_scan_index)),
                build_ptc_detail_panel(scan_view, det_df),
            )

    map_section = bundle["map"]
    if map_section["error"] is None:
        map_stage_df = map_section["map_df"]
        map_contrib_df = map_section["contrib_df"]

        @app.callback(Output("map-stage-dropdown", "options"), Output("map-stage-dropdown", "value"), Input("map-selector-dropdown", "value"))
        def update_map_stage_options(map_selector: str):
            stages = map_contrib_df.loc[map_contrib_df["map_selector"] == map_selector, "stage"].drop_duplicates().tolist()
            if not stages:
                coadd_stage_rows = map_stage_df.loc[(map_stage_df["map_selector"] == map_selector) & (map_stage_df["is_coadd"] == 1), "stage"].drop_duplicates().tolist()
                stages = coadd_stage_rows if coadd_stage_rows else ["raw_coadd"]
            options = [{"label": stage, "value": stage} for stage in stages]
            value = options[0]["value"] if options else None
            return options, value

        @app.callback(
            Output("map-contrib-figure", "figure"),
            Output("map-obs-rank-figure", "figure"),
            Output("map-scatter-figure", "figure"),
            Output("map-detail-panel", "children"),
            Input("map-selector-dropdown", "value"),
            Input("map-stage-dropdown", "value"),
        )
        def update_map_detail(map_selector: str, stage: str):
            return (
                build_map_contrib_figure(map_contrib_df, map_selector, stage),
                build_map_obs_rank_figure(map_stage_df, map_contrib_df, map_selector, stage),
                build_map_contrib_scatter_figure(map_stage_df, map_contrib_df, map_selector, stage),
                build_map_detail_panel(map_stage_df, map_contrib_df, map_selector, stage),
            )

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
