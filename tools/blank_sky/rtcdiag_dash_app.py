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


def table(columns: list[str], data: list[dict[str, object]], page_size: int = 12, table_id: str | None = None):
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


def build_heatmap(scan_df: pd.DataFrame, obsnum: str) -> go.Figure:
    obs_df = scan_df.loc[scan_df["obsnum"] == obsnum].copy()
    if obs_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No scan/network rows for obsnum {obsnum}")
        return fig

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
    return fig


def build_network_trend(scan_df: pd.DataFrame, obsnum: str, network: int) -> go.Figure:
    df = scan_df.loc[(scan_df["obsnum"] == obsnum) & (scan_df["network"] == network)].copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty:
        fig.update_layout(title=f"No scan rows for obsnum {obsnum} network {network}")
        return fig

    x = df["output_scan_index"]
    fig.add_trace(
        go.Scatter(x=x, y=df["row_severity"], name="row_severity", mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["network_impulsive_score_max"], name="imp_score_max", mode="lines"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["max_slot_event_score"], name="slot_score_max", mode="lines"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["step_det_frac"], name="step_det_frac", mode="lines"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["step_alignment_frac"], name="step_alignment_frac", mode="lines"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["impulsive_mask_applied"],
            name="imp_mask",
            mode="markers",
            marker={"symbol": "x", "size": 8},
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["step_mask_applied"],
            name="step_mask",
            mode="markers",
            marker={"symbol": "cross", "size": 8},
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
    return fig


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
            html.H2("RTC Engineering Dashboard"),
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
                style={"marginBottom": "16px", "fontFamily": "Menlo, Monaco, Consolas, monospace"},
            ),
            html.Details(
                [
                    html.Summary("How To Read This Dashboard"),
                    dcc.Markdown(
                        """
This dashboard is built from `rtcdiag` survey rows.

- One **obsnum** is one observation.
- Within an obsnum, each **output scan** is one output time chunk.
- Within an output scan, each **network row** summarizes all detector samples for one network.
- So a heatmap cell or scan/network table row means: **one obsnum, one output scan, one network**.

`row_severity` is a compact ranking score, not a physical unit. It is the largest of several normalized contamination indicators:

- step-like detector fraction times step alignment
- low/mid common-mode ratio
- impulsive event score
- impulsive slot capture score

Use it for triage and ranking, not as a calibrated threshold by itself.

`step_mask_applied` and `impulsive_mask_applied` are binary indicators showing whether those RTC masking paths fired on that scan/network row.
                        """
                    ),
                ],
                style={**HELP_BOX_STYLE, "marginBottom": "18px"},
                open=True,
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
            html.H3("Obs Summary"),
            section_help(
                "What this table shows",
                """
Each row is one obsnum summarized across all selected networks and all output scans.

Use this table to answer: which obsnums are the hardest overall? The most useful columns are:

- `max_row_severity`: worst single scan/network row in the obsnum
- `max_step_det_frac`: densest step-like row seen anywhere in the obsnum
- `max_cm_lowmid`: strongest low-frequency common-mode excess
- `max_impulsive_event_score` / `top_slot_event_score`: strongest impulsive behavior seen in detector-level or captured-slot diagnostics
- `masked_network_scans` / `impulsive_masked_network_scans`: how many scan/network rows were acted on by the step or impulsive mask paths
                """,
            ),
            table(
                [
                    "obsnum",
                    "product_kind",
                    "max_row_severity",
                    "max_step_det_frac",
                    "max_cm_lowmid",
                    "max_impulsive_event_score",
                    "top_slot_event_score",
                    "masked_network_scans",
                    "impulsive_masked_network_scans",
                ],
                rounded_records(
                    obs_df,
                    [
                        "obsnum",
                        "product_kind",
                        "max_row_severity",
                        "max_step_det_frac",
                        "max_cm_lowmid",
                        "max_impulsive_event_score",
                        "top_slot_event_score",
                        "masked_network_scans",
                        "impulsive_masked_network_scans",
                    ],
                ),
                page_size=15,
            ),
            html.H3("Network Summary"),
            section_help(
                "What this table shows",
                """
Each row is one network aggregated across all selected obsnums.

This is for finding chronic network-level behavior. `worst_obsnum` tells you which obsnum produced the worst severity for that network, while `total_masked_network_scans` and `total_impulsive_masked_network_scans` tell you how often the network was acted on across the whole reduction set.
                """,
            ),
            table(
                [
                    "network",
                    "n_obsnums",
                    "max_row_severity",
                    "max_max_step_det_frac",
                    "max_max_cm_lowmid",
                    "max_impulsive_frac_ge_threshold",
                    "total_masked_network_scans",
                    "total_impulsive_masked_network_scans",
                    "worst_obsnum",
                ],
                rounded_records(
                    by_network_df,
                    [
                        "network",
                        "n_obsnums",
                        "max_row_severity",
                        "max_max_step_det_frac",
                        "max_max_cm_lowmid",
                        "max_impulsive_frac_ge_threshold",
                        "total_masked_network_scans",
                        "total_impulsive_masked_network_scans",
                        "worst_obsnum",
                    ],
                ),
                page_size=10,
            ),
            section_help(
                "Heatmap: how to read it",
                """
Each heatmap cell is one **scan/network row** for the selected obsnum.

- x-axis: output scan index
- y-axis: network id
- color: `row_severity`

Use this to see whether a problem is isolated to a few scan chunks, concentrated in a few networks, or broadly spread across the obsnum. Hover text shows the key local metrics that produced the severity ranking, along with whether the step or impulsive masks fired.
                """,
            ),
            section_help(
                "Trend plot: how to read it",
                """
The trend plot is for one selected obsnum and one selected network.

- left axis: severity-like quantities and impulsive scores
- right axis: fractions and binary mask flags

`row_severity`, `imp_score_max`, and `slot_score_max` help you see whether the row is driven by impulsive structure. `step_det_frac` and `step_alignment_frac` show whether the same network/scan row also looks step-like. Marker-only traces show where the step and impulsive masking paths actually fired.
                """,
            ),
            html.Div(
                [
                    dcc.Graph(id="heatmap"),
                    dcc.Graph(id="network-trend"),
                ]
            ),
            html.H3("Obs-Network Summary"),
            section_help(
                "What this table shows",
                """
Each row is one network within the selected obsnum, aggregated over all output scans in that obsnum.

This is the bridge between the all-obs summary tables and the scan-level detail views. It tells you which networks are dominating the selected obsnum and whether that dominance comes from step-like, impulsive, or common-mode metrics.
                """,
            ),
            html.Div(id="obs-network-table"),
            html.H3("Top Scan/Network Rows"),
            section_help(
                "What this table shows",
                """
These are the highest-severity scan/network rows within the selected obsnum.

This is usually the first table to inspect after the heatmap. It surfaces the exact scan chunks and networks that drive the obsnum ranking. `step_mask_applied` and `impulsive_mask_applied` let you compare diagnostic severity against the actual runtime actions taken by RTC, while the `impulsive_mask_*trigger` and cluster columns show why the impulsive path decided to act or not act.
                """,
            ),
            html.Div(id="scan-row-table"),
            html.H3("Top Impulsive Slots"),
            section_help(
                "What this table shows",
                """
This table shows the top captured impulsive events for the selected obsnum and selected network.

Each row is one stored slot from the compact RTC impulsive capture product. `event_kind_label` distinguishes raw-like and delta-like captures. `event_score`, `peak_abs_z`, and `peak_delta_abs_z` help separate broad raw excursions from sharper delta-like hits.
                """,
            ),
            html.Div(id="slot-row-table"),
        ],
        style={"padding": "20px", "maxWidth": "1600px"},
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
        Output("heatmap", "figure"),
        Output("network-trend", "figure"),
        Output("obs-network-table", "children"),
        Output("scan-row-table", "children"),
        Output("slot-row-table", "children"),
        Input("obs-dropdown", "value"),
        Input("network-dropdown", "value"),
    )
    def update_detail(obsnum: str, network: int):
        obs_network_view = obs_network_df.loc[obs_network_df["obsnum"] == obsnum].copy()
        scan_view = scan_df.loc[scan_df["obsnum"] == obsnum].copy()
        slot_view = slot_df.loc[(slot_df["obsnum"] == obsnum) & (slot_df["network"] == network)].copy()
        scan_top_view = scan_view.sort_values("row_severity", ascending=False).head(30)

        return (
            build_heatmap(scan_df, obsnum),
            build_network_trend(scan_df, obsnum, int(network)),
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
                page_size=10,
            ),
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
                    "impulsive_mask_local_trigger",
                    "impulsive_mask_cross_network_trigger",
                    "impulsive_mask_high_score_override_trigger",
                    "impulsive_mask_rejected_max_fraction",
                    "impulsive_mask_cluster_network_count",
                    "impulsive_mask_cluster_peak_score",
                    "impulsive_mask_override_score",
                    "impulsive_mask_override_uses_network_peak",
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
                        "impulsive_mask_local_trigger",
                        "impulsive_mask_cross_network_trigger",
                        "impulsive_mask_high_score_override_trigger",
                        "impulsive_mask_rejected_max_fraction",
                        "impulsive_mask_cluster_network_count",
                        "impulsive_mask_cluster_peak_score",
                        "impulsive_mask_override_score",
                        "impulsive_mask_override_uses_network_peak",
                    ],
                ),
                page_size=12,
            ),
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
                    slot_view.head(40),
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
                page_size=12,
            ),
        )

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
