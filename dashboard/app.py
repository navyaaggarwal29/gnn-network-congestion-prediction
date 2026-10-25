"""Interactive Streamlit NOC dashboard for saved replays and live simulation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from congestion_sim.dashboard_data import (  # noqa: E402
    DashboardDataError, STRATEGY_LABELS, STRATEGY_ORDER, comparison_data, decision_counts,
    event_rows, filter_run, load_dashboard_data, measurement_rows,
    metric_improvements, parse_json_value,
)
from congestion_sim.onset import SCENARIOS  # noqa: E402
from congestion_sim.routing import OnsetPredictor  # noqa: E402
from congestion_sim.topology import build_default_topology  # noqa: E402
from phase4_experiment import run_one  # noqa: E402

st.set_page_config(page_title="NetScope | Congestion Lab", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  :root { --ink:#eaf2ff; --muted:#92a4bd; --cyan:#39d7f2; --blue:#5d8cff; --panel:#111c2d; }
  .stApp { background: radial-gradient(ellipse at 70% -20%, #17304a 0, #0b1322 48%, #08101c 100%); color:var(--ink); }
  [data-testid="stSidebar"] { background:#0b1422; border-right:1px solid #21324a; }
  [data-testid="stHeader"] { background:rgba(8,16,28,.75); }
  h1,h2,h3 { letter-spacing:-.025em; }
  .eyebrow { color:#39d7f2; font-size:.76rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }
  .hero { padding:.35rem 0 1rem 0; }
  .hero p,.caption { color:#9aacc2; }
  [data-testid="stMetric"] { background:linear-gradient(145deg,#142238,#101a2a); border:1px solid #243751; padding:16px 18px; border-radius:14px; }
  [data-testid="stMetricLabel"] { color:#9eb0c6 !important; }
  [data-testid="stMetricValue"] { color:#f2f7ff; }
  div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#22344c; }
  .stButton>button { border-radius:9px; border:1px solid #2a455f; }
  .stButton>button[kind="primary"] { background:#103d54; border-color:#27b8d4; }
  hr { border-color:#23344b; }
</style>
""", unsafe_allow_html=True)

SCENARIO_LABELS = {
    "gradual_ramp": "Gradual ramp", "sustained_bottleneck": "Sustained bottleneck",
    "periodic": "Periodic traffic", "correlated": "Correlated demand", "sudden_burst": "Sudden burst",
}
METRICS = {
    "Mean delay (ms)": "mean_delay_ms",
    "p95 delay (ms)": "p95_delay_ms",
    "Congestion duration (steps)": "congestion_duration_steps",
    "Hop-level delivery proxy": "packet_delivery_ratio_hop_proxy",
    "Hop-level throughput sum (Mbps)": "delivered_throughput_mbps_hop_sum",
    "Applied reroutes": "reroutes", "Rejected reroutes": "rejected_reroutes",
}


@st.cache_data(show_spinner="Loading saved simulator and model results…")
def cached_dashboard_data(root: str):
    return load_dashboard_data(root)


@st.cache_resource(show_spinner="Loading trained onset model…")
def cached_predictor(model_name: str, root: str):
    return OnsetPredictor(model_name, Path(root) / "results/phase3_6",
                          build_default_topology(), device="cpu")


def route_to_edges(route):
    return {frozenset((a, b)) for a, b in zip(route or [], (route or [])[1:])}


def to_float(value, missing=np.nan):
    try:
        return float(value) if pd.notna(value) else missing
    except (TypeError, ValueError):
        return missing


def render_table(frame: pd.DataFrame, *, max_rows: int = 100):
    """Render compact dashboard tables using Markdown instead of Arrow IPC."""
    if frame.empty:
        st.caption("No rows available for this selection.")
        return
    headers = [str(column).replace("|", "\\|") for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.head(max_rows).itertuples(index=False, name=None):
        cells = []
        for value in row:
            if value is None or pd.isna(value):
                text = "—"
            elif isinstance(value, (float, np.floating)):
                text = f"{value:.3f}"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))
    if len(frame) > max_rows:
        st.caption(f"Showing {max_rows} of {len(frame)} rows.")


def current_route_map(step_frame: pd.DataFrame) -> dict:
    if step_frame.empty:
        return {}
    for value in step_frame.flow_routes.dropna().tolist():
        parsed = parse_json_value(value, expected_type=dict)
        if parsed:
            return parsed
    return {}


def make_topology_figure(topology, step_frame: pd.DataFrame, decisions: pd.DataFrame,
                         selected_link: tuple[str, str], selected_route: list[str] | None,
                         predicted_threshold: float | None, step: int) -> go.Figure:
    graph = topology.graph
    positions = nx.spring_layout(graph, seed=23, weight="propagation_delay_ms", k=1.25, iterations=80)
    directions = {(r.source_router, r.destination_router): r
                  for r in step_frame.itertuples(index=False)}
    route_edges = route_to_edges(selected_route)
    applied_edges = set()
    latest_applied = decisions[pd.to_numeric(decisions.applied_step, errors="coerce").le(step)] if not decisions.empty else decisions
    if not latest_applied.empty:
        latest = latest_applied.sort_values("applied_step").iloc[-1]
        applied_edges = route_to_edges(parse_json_value(latest.alternative_route, expected_type=list))
    figure = go.Figure()
    status_color = {"low": "#39d98a", "moderate": "#ffc857", "high": "#ff5364"}
    for u, v in topology.links:
        uv, vu = directions.get((u, v)), directions.get((v, u))
        observed = [to_float(x.utilization) for x in (uv, vu) if x is not None]
        observed_max = max(observed) if observed else np.nan
        probs = [to_float(x.predicted_congestion_probability) for x in (uv, vu) if x is not None]
        probs = [p for p in probs if np.isfinite(p)]
        risk = max(probs) if probs else np.nan
        observed_hot = bool(np.isfinite(observed_max) and observed_max >= .85)
        predicted_hot = bool(predicted_threshold is not None and np.isfinite(risk) and risk >= predicted_threshold)
        level = "high" if observed_hot else "moderate" if np.isfinite(observed_max) and observed_max >= .60 else "low"
        edge = frozenset((u, v))
        color = status_color[level]
        width, dash = 2.4, "solid"
        if predicted_hot and not observed_hot:
            color, width, dash = "#bd8cff", 3.1, "dash"
        if edge in route_edges:
            color, width = "#40cfff", 4.8
        if edge in applied_edges:
            color, width = "#ff9f43", 5.5
        if edge == frozenset(selected_link):
            width = max(width, 6.2)
        pos_u, pos_v = positions[u], positions[v]
        bandwidth = float(graph[u][v]["bandwidth_mbps"])
        details = []
        for a, b, item in ((u, v, uv), (v, u, vu)):
            if item is None:
                details.append(f"{a} → {b}: unavailable")
            else:
                details.append(
                    f"{a} → {b}: utilization {item.utilization:.1%}, queue {item.queue_occupancy:.1%}, "
                    f"delay {item.delay_ms:.2f} ms, loss {item.packet_loss_pct:.2f}%" +
                    (f", onset risk {to_float(item.predicted_congestion_probability):.3f}" if pd.notna(item.predicted_congestion_probability) else "")
                )
        figure.add_trace(go.Scatter(
            x=[pos_u[0], pos_v[0], None], y=[pos_u[1], pos_v[1], None], mode="lines",
            line={"color": color, "width": width, "dash": dash},
            customdata=[["<br>".join(details)]] * 3,
            hovertemplate=f"<b>{u} ↔ {v}</b> · {bandwidth:.0f} Mbps<br>%{{customdata[0]}}<extra></extra>",
            showlegend=False, name=f"{u}-{v}"))
        mid = ((pos_u[0] + pos_v[0]) / 2, (pos_u[1] + pos_v[1]) / 2)
        figure.add_annotation(x=mid[0], y=mid[1], text=f"{bandwidth:.0f}", showarrow=False,
                              font={"size": 8, "color": "#8da4bf"}, bgcolor="rgba(8,16,28,.72)")
    figure.add_trace(go.Scatter(
        x=[positions[n][0] for n in topology.routers], y=[positions[n][1] for n in topology.routers],
        mode="markers+text", text=topology.routers, textposition="middle center",
        marker={"size": 34, "color": "#112a40", "line": {"color": "#39d7f2", "width": 2}},
        textfont={"color": "#f3f8ff", "size": 10}, hovertemplate="<b>%{text}</b><extra>Router</extra>",
        showlegend=False))
    figure.update_layout(height=560, margin={"l": 4, "r": 4, "t": 5, "b": 5},
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"},
                         hoverlabel={"bgcolor": "#122239", "font_color": "#edf5ff"})
    return figure


def build_timeline(step_frame: pd.DataFrame, decisions: pd.DataFrame,
                   threshold: float | None) -> tuple[go.Figure, pd.DataFrame]:
    timeline = []
    measurements = step_frame.sort_values(["link_id", "step"])
    for link, group in measurements.groupby("link_id", sort=False):
        group = group.sort_values("step")
        hot = pd.to_numeric(group.utilization, errors="coerce").ge(.85)
        starts = hot & ~hot.shift(1, fill_value=False)
        for row in group[starts].itertuples():
            timeline.append({"step": int(row.step), "event": "Observed congestion", "link": link,
                             "detail": f"Observed utilization crossed 85% ({row.utilization:.1%})."})
        if threshold is not None:
            scores = pd.to_numeric(group.predicted_congestion_probability, errors="coerce")
            for row in group[scores.ge(threshold)].itertuples():
                timeline.append({"step": int(row.step), "event": "Predicted warning", "link": link,
                                 "detail": f"Model risk {row.predicted_congestion_probability:.3f} ≥ saved threshold {threshold:.3f}."})
    demand = measurements.groupby("step").offered_demand_mbps.sum(min_count=1).sort_index()
    if len(demand) > 1:
        delta = demand.diff()
        cutoff = max(8.0, float(demand.max(skipna=True) or 0) * .18)
        for step_id in delta.index[delta.abs() >= cutoff]:
            timeline.append({"step": int(step_id), "event": "Traffic change", "link": "—",
                             "detail": f"Measured link-hop offered load changed by {delta.loc[step_id]:+.1f} Mbps."})
    for row in decisions.itertuples(index=False):
        decision_step = int(to_float(row.step, 0))
        alt = parse_json_value(row.alternative_route, expected_type=list)
        link = str(row.target_link) if pd.notna(row.target_link) else "—"
        if alt:
            timeline.append({"step": decision_step, "event": "Pending route", "link": link,
                             "detail": f"Flow {row.flow_id}: route scheduled {row.old_route if hasattr(row, 'old_route') else row.current_route} → {alt}; reason: {row.routing_decision}."})
            applied = to_float(row.applied_step)
            if np.isfinite(applied):
                timeline.append({"step": int(applied), "event": "Applied route", "link": link,
                                 "detail": f"Flow {row.flow_id}: route applied at step {int(applied)}; alternative {alt}."})
        else:
            timeline.append({"step": decision_step, "event": "Rejected / retained", "link": link,
                             "detail": str(row.routing_decision)})
    frame = pd.DataFrame(timeline)
    figure = go.Figure()
    order = ["Traffic change", "Predicted warning", "Observed congestion", "Pending route", "Applied route", "Rejected / retained"]
    colors = {"Traffic change": "#66aaff", "Predicted warning": "#bd8cff", "Observed congestion": "#ff5364",
              "Pending route": "#ffc857", "Applied route": "#39d98a", "Rejected / retained": "#ff9f43"}
    for event_type in order:
        subset = frame[frame.event == event_type] if not frame.empty else frame
        if subset.empty:
            continue
        figure.add_trace(go.Scatter(
            x=subset.step, y=[event_type] * len(subset), mode="markers",
            marker={"color": colors[event_type], "size": 10, "line": {"color": "#09111d", "width": 1}},
            customdata=subset[["link", "detail"]].to_numpy(),
            hovertemplate="Step %{x} · %{y}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            name=event_type))
    figure.update_layout(height=280, margin={"l": 8, "r": 8, "t": 10, "b": 10},
                         xaxis_title="Simulation step", yaxis={"categoryorder": "array", "categoryarray": order},
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         legend={"orientation": "h", "y": -0.27}, hoverlabel={"bgcolor": "#122239"})
    return figure, frame.sort_values(["step", "event"]).reset_index(drop=True) if not frame.empty else frame


def render_live_view(data, scenario: str, strategy: str, seed: int, source: str,
                     live_run=None, playback_speed: float = 1.0):
    topology = build_default_topology()
    if source == "Replay saved results":
        run = filter_run(data["log"], scenario, strategy, seed)
        run_label = "SAVED PHASE 4 REPLAY"
    else:
        if live_run is None:
            st.info("Choose a traffic scenario and routing strategy, then use **Generate new simulation** in the sidebar.")
            return
        run = live_run
        run_label = "NEW SIMULATED RUN · EXECUTED NOW"
    try:
        measures = measurement_rows(run)
        decisions = event_rows(run)
    except DashboardDataError as exc:
        st.error(str(exc)); return
    steps = sorted(measures.step.unique().tolist())
    signature = f"{source}|{scenario}|{strategy}|{seed}|{len(steps)}"
    if st.session_state.get("playback_signature") != signature:
        st.session_state.playback_signature = signature
        st.session_state.playback_step = int(steps[0])
        st.session_state.playback_playing = False
    interval = float(playback_speed)

    def start_playback():
        st.session_state.playback_playing = True

    def pause_playback():
        st.session_state.playback_playing = False

    def reset_playback():
        st.session_state.playback_playing = False
        st.session_state.playback_step = int(steps[0])

    def advance_playback():
        current_step = int(st.session_state.get("playback_step", steps[0]))
        next_steps = [value for value in steps if value > current_step]
        if next_steps:
            st.session_state.playback_step = next_steps[0]

    @st.fragment(run_every=interval if st.session_state.get("playback_playing", False) else None)
    def dynamic_replay():
        if st.session_state.get("playback_playing", False):
            current = int(st.session_state.get("playback_step", steps[0]))
            next_steps = [x for x in steps if x > current]
            if next_steps:
                st.session_state.playback_step = next_steps[0]
            else:
                st.session_state.playback_playing = False
        step = st.slider("Simulation step", min_value=int(steps[0]), max_value=int(steps[-1]),
                         step=1,
                         key="playback_step", help="Scrub only measured steps in this selected run.")
        controls = st.columns([1, 1, 1, 1, 4])
        with controls[0]:
            st.button("▶ Play", disabled=st.session_state.get("playback_playing", False),
                      key="play_button", on_click=start_playback)
        with controls[1]:
            st.button("⏸ Pause", disabled=not st.session_state.get("playback_playing", False),
                      key="pause_button", on_click=pause_playback)
        with controls[2]:
            st.button("↺ Reset", key="reset_button", on_click=reset_playback)
        with controls[3]:
            st.button("Step +1", key="step_button", on_click=advance_playback)
        current = measures[measures.step == int(step)].copy()
        if current.empty:
            st.warning("No measurements are available at this recorded step."); return
        applied = pd.to_numeric(decisions.applied_step, errors="coerce") if not decisions.empty else pd.Series(dtype=float)
        counts = decision_counts(decisions, int(step))
        pending, applied_count = counts["pending"], counts["applied"]
        rejected, decision_count = counts["rejected"], counts["issued"]
        directional_hot = int(pd.to_numeric(current.utilization, errors="coerce").ge(.85).sum())
        scenario_label = SCENARIO_LABELS.get(scenario, scenario)
        st.markdown(f'<div class="eyebrow">{run_label} · SIMULATED DATA · {scenario_label} · {STRATEGY_LABELS[strategy]}</div>', unsafe_allow_html=True)
        if source != "Replay saved results":
            st.caption("This run was executed with the simulator and selected controller; it is not a relabeled replay.")
        st.markdown("### Network state")
        kpis = st.columns(6)
        kpis[0].metric("Routers", topology.graph.number_of_nodes())
        kpis[1].metric("Physical links", topology.graph.number_of_edges())
        kpis[2].metric("Simulation step", int(step))
        kpis[3].metric("Hot directions", directional_hot)
        kpis[4].metric("Pending changes", pending)
        kpis[5].metric("Applied reroutes", applied_count)
        current["physical_link"] = current.apply(lambda r: " ↔ ".join(sorted([r.source_router, r.destination_router])), axis=1)
        physical_links = sorted({tuple(sorted(edge)) for edge in topology.links})
        selected_label = st.selectbox("Inspect a physical link", physical_links,
                                      format_func=lambda p: f"{p[0]} ↔ {p[1]}", key="inspect_link")
        route_map = current_route_map(current)
        route_names = list(route_map)
        selected_route = None
        route_name = None
        if route_names:
            route_name = st.selectbox("Highlight a current traffic route", route_names, key="inspect_route")
            selected_route = route_map.get(route_name)
        threshold = None
        model = None
        if strategy in {"gru", "gnn"}:
            model = "gru_onset" if strategy == "gru" else "gnn_onset"
            threshold = float(data["selection"]["threshold_by_model"][model])
        fig = make_topology_figure(topology, current, decisions, selected_label,
                                   selected_route, threshold, int(step))
        left, right = st.columns([1.55, 1])
        with left:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Solid green/amber/red = observed utilization. Dashed violet = model risk above its saved threshold before observed congestion. Cyan = selected route; orange = latest applied reroute.")
        with right:
            st.markdown("#### Selected link · both directions")
            selected_pair = set(selected_label)
            link_data = current[current.apply(lambda r: {r.source_router, r.destination_router} == selected_pair, axis=1)].copy()
            if link_data.empty:
                st.info("No selected-link measurement is present at this step.")
            else:
                display = link_data[["link_id", "bandwidth_mbps", "utilization", "queue_occupancy", "delay_ms", "packet_loss_pct"]].copy()
                display = display.rename(columns={"link_id": "Direction", "bandwidth_mbps": "Bandwidth Mbps",
                    "utilization": "Utilization", "queue_occupancy": "Queue occupancy", "delay_ms": "Delay ms", "packet_loss_pct": "Packet loss %"})
                display["Utilization"] *= 100
                display["Queue occupancy"] *= 100
                display["Utilization"] = display["Utilization"].map(lambda x: f"{x:.0f}%")
                display["Queue occupancy"] = display["Queue occupancy"].map(lambda x: f"{x:.0f}%")
                render_table(display)
            st.markdown("#### Selected route")
            if selected_route:
                st.code(" → ".join(selected_route))
            else:
                st.caption("No route map was recorded for this step.")
            st.markdown("#### Controller state")
            count_cols = st.columns(2)
            count_cols[0].metric("Pending", pending)
            count_cols[1].metric("Applied", applied_count)
            count_cols = st.columns(2)
            count_cols[0].metric("Rejected / retained", rejected)
            count_cols[1].metric("Decisions issued", decision_count)
            if not decisions.empty:
                applied_now = decisions[pd.to_numeric(decisions.applied_step, errors="coerce").eq(int(step))]
                if not applied_now.empty:
                    st.success(f"{len(applied_now)} route change(s) applied at step {int(step)}.")
        st.markdown("### Prediction and traffic signals")
        signal_cols = st.columns([1.1, .9, 1.4])
        with signal_cols[0]:
            hop_load = pd.to_numeric(current.offered_demand_mbps, errors="coerce").sum(min_count=1)
            st.metric("Observed link-hop offered load", f"{hop_load:.1f} Mbps" if pd.notna(hop_load) else "—")
        with signal_cols[1]:
            current_drops = pd.to_numeric(current.packets_dropped, errors="coerce").sum(min_count=1)
            st.metric("Packets dropped across hops", f"{current_drops:.0f}" if pd.notna(current_drops) else "—")
        with signal_cols[2]:
            if threshold is not None:
                st.metric("Saved onset threshold", f"{threshold:.4f}", help="Validation-selected Phase 3.6 threshold.")
            else:
                st.metric("Predicted onset risk", "Not generated", help="Static and reactive strategies do not call the onset model.")
        if threshold is not None:
            scored = current[pd.to_numeric(current.predicted_congestion_probability, errors="coerce").notna()].copy()
            scored["predicted_congestion_probability"] = pd.to_numeric(scored.predicted_congestion_probability, errors="coerce")
            at_risk = scored[scored.predicted_congestion_probability >= threshold].sort_values("predicted_congestion_probability", ascending=False)
            if at_risk.empty:
                st.info("No links crossed the saved model threshold at this recorded step. Missing scores are not treated as zero.")
            else:
                render_table(at_risk[["link_id", "utilization", "queue_occupancy", "delay_ms", "packet_loss_pct", "predicted_congestion_probability"]])
        elif strategy == "static":
            st.caption("Static Dijkstra does not generate onset predictions.")
        else:
            st.caption("Reactive routing responds to observed utilization or queue occupancy; it does not call the onset model.")
        st.markdown("### Event timeline")
        timeline_figure, timeline = build_timeline(measures, decisions, threshold)
        if timeline.empty:
            st.info("No traffic-change or routing events were recorded for this run.")
        else:
            st.plotly_chart(timeline_figure, use_container_width=True, config={"displayModeBar": False})
            current_events = timeline[timeline.step <= int(step)]
            if not current_events.empty:
                labels = [f"step {r.step} · {r.event} · {r.link}" for r in current_events.itertuples()]
                selected_event = st.selectbox("Inspect event", range(len(labels)), format_func=lambda i: labels[i], key="inspect_event")
                st.info(current_events.iloc[int(selected_event)].detail)
            else:
                st.caption("No timeline events have occurred by this step.")

    dynamic_replay()


def render_comparison(data, scenario: str):
    st.markdown("## Strategy comparison")
    st.caption("Saved Phase 4 paired runs · all values are simulator outputs. Delivery and throughput metrics are hop-level proxies/sums, not end-to-end metrics.")
    metric_label = st.selectbox("Metric", list(METRICS), index=0, key="comparison_metric")
    metric = METRICS[metric_label]
    comparison = comparison_data(data["metrics"], scenario, metric)
    bar_options = {}
    if comparison.value_std.notna().all():
        bar_options["error_y"] = {"type": "data", "array": comparison.value_std}
    chart = go.Figure(go.Bar(x=comparison.strategy_label, y=comparison.value_mean,
                             marker_color=["#718096", "#39d98a", "#54a7ff", "#bd8cff"],
                             customdata=np.stack([comparison.seed_count], axis=1),
                             hovertemplate="%{x}<br>mean %{y:.4g}<br>paired seeds %{customdata[0]}<extra></extra>",
                             **bar_options))
    if metric == "packet_delivery_ratio_hop_proxy":
        chart.update_yaxes(tickformat=".0%")
    chart.update_layout(height=390, yaxis_title=metric_label, xaxis_title="Routing strategy",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin={"t": 20, "b": 15}, hoverlabel={"bgcolor": "#122239"})
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    table = data["metrics"][data["metrics"].scenario == scenario].groupby("strategy", as_index=False).agg(
        mean_delay_ms=("mean_delay_ms", "mean"), p95_delay_ms=("p95_delay_ms", "mean"),
        congestion_duration_steps=("congestion_duration_steps", "mean"),
        packet_delivery_ratio_hop_proxy=("packet_delivery_ratio_hop_proxy", "mean"),
        delivered_throughput_mbps_hop_sum=("delivered_throughput_mbps_hop_sum", "mean"),
        reroutes=("reroutes", "mean"), rejected_reroutes=("rejected_reroutes", "mean"))
    table["strategy"] = table.strategy.map(STRATEGY_LABELS)
    render_table(table)
    st.markdown("#### Seed-paired change versus Static Dijkstra")
    rows = []
    for strat in ("reactive", "gru", "gnn"):
        values = metric_improvements(data["metrics"], scenario, strat)
        rows.append({"Strategy": STRATEGY_LABELS[strat],
                     "Mean delay improvement %": values["mean_delay_improvement_pct"],
                     "Delivery proxy change (pp)": values["delivery_proxy_delta_pp"],
                     "Congestion-duration improvement %": values["congestion_duration_improvement_pct"]})
    render_table(pd.DataFrame(rows))


def render_model_performance(data):
    st.markdown("## Phase 3.6 model performance")
    st.caption("Held-out test episodes and windows from the saved model evaluation. These are prediction metrics, separate from Phase 4 routing outcomes.")
    model_table = data["model_metrics"][data["model_metrics"].model.isin(["gru_onset", "gnn_onset"])].copy()
    if model_table.empty:
        st.warning("The Phase 3.6 model test metrics are unavailable."); return
    order = ["gru_onset", "gnn_onset"]
    values = model_table.set_index("model").reindex(order)
    labels = ["GRU onset", "GNN onset"]
    metrics = ["precision", "recall", "f1", "window_pr_auc"]
    chart = go.Figure()
    colors = ["#39d7f2", "#bd8cff"]
    for color, metric in zip(colors, metrics):
        chart.add_trace(go.Bar(name=metric.replace("_", " ").upper(), x=labels,
                               y=values[metric].to_numpy(), marker_color=color))
    chart.update_layout(barmode="group", height=380, yaxis={"range": [0, 1], "title": "Score"},
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    shown = values.reset_index()[["model", "threshold", "precision", "recall", "f1", "window_pr_auc",
                                   "false_alert_episodes_per_run", "mean_lead_steps", "eligible_onset_episodes"]]
    shown["model"] = shown.model.map({"gru_onset": "GRU onset", "gnn_onset": "GNN onset"})
    shown = shown.rename(columns={"threshold": "Saved threshold", "precision": "Episode precision",
        "recall": "Episode recall", "f1": "Episode F1", "window_pr_auc": "Window PR-AUC",
        "false_alert_episodes_per_run": "False alert episodes / run", "mean_lead_steps": "Mean lead steps",
        "eligible_onset_episodes": "Eligible test episodes"})
    render_table(shown)
    gru_row, gnn_row = values.loc["gru_onset"], values.loc["gnn_onset"]
    winner = "GRU" if gru_row.precision > gnn_row.precision and gru_row.f1 > gnn_row.f1 else "GNN"
    st.info(f"In the saved held-out evaluation, {winner} scored higher on both episode precision and F1. "
            f"GRU and GNN recall were {gru_row.recall:.1%} and {gnn_row.recall:.1%}; "
            f"their false-alert rates were {gru_row.false_alert_episodes_per_run:.2f} and "
            f"{gnn_row.false_alert_episodes_per_run:.2f} episodes/run. These data show limited, "
            "scenario-dependent generalization and do not establish reliable proactive routing.")
    scenario = data["scenario_model_metrics"]
    scenario = scenario[scenario.model.isin(["gru_onset", "gnn_onset"])][["model", "scenario", "precision", "recall", "f1", "eligible_onset_episodes", "true_positive_episodes", "false_alert_episodes"]]
    scenario["model"] = scenario.model.map({"gru_onset": "GRU onset", "gnn_onset": "GNN onset"})
    st.markdown("#### Held-out scenario breakdown")
    render_table(scenario)


def main():
    st.markdown('<div class="hero"><div class="eyebrow">NETWORK OPERATIONS · SIMULATION LAB</div><h1>NetScope</h1><p>Congestion prediction and routing control · simulated network only</p></div>', unsafe_allow_html=True)
    try:
        with st.spinner("Loading saved Phase 4 and Phase 3.6 artifacts…"):
            data = cached_dashboard_data(str(ROOT))
    except DashboardDataError as exc:
        st.error(str(exc))
        st.markdown("Generate the Phase 3.6 and Phase 4 artifacts first. See the README reproduction commands.")
        st.stop()
    sidebar = st.sidebar
    sidebar.markdown("### ◈ NetScope")
    section = sidebar.radio("Workspace", ["Live monitor", "Routing comparison", "Model performance", "Demo guide"], label_visibility="collapsed")
    sidebar.markdown("---")
    scenario = "periodic"
    strategy = "static"
    seed = 3000
    source = "Replay saved results"
    live_run = None
    speed = 1.0
    if section in {"Live monitor", "Demo guide"}:
        source = sidebar.radio("Data source", ["Replay saved results", "Run a new simulation"],
                               help="Replay is a saved Phase 4 trace. New simulation executes the selected simulator and controller now.")
        demo_mode = sidebar.toggle("Guided demonstration", value=(section == "Demo guide"),
                                   help="Uses a specifically selected periodic run to make alerts and route changes inspectable; it is not representative of all experiments.")
        if demo_mode:
            scenario = "periodic"
            sidebar.info("Demo scenario is fixed to periodic traffic. Saved seed 3000 was selected from actual runs; the system does not force an alert or route change.")
            strategy = sidebar.selectbox("Demo strategy", ["static", "gnn", "reactive"], format_func=lambda x: STRATEGY_LABELS[x], index=0)
        else:
            scenario = sidebar.selectbox("Traffic scenario", list(SCENARIOS), format_func=lambda x: SCENARIO_LABELS[x])
            strategy = sidebar.selectbox("Routing strategy", STRATEGY_ORDER, format_func=lambda x: STRATEGY_LABELS[x])
        if source == "Replay saved results":
            available = data["metrics"][(data["metrics"].scenario == scenario) & (data["metrics"].strategy == strategy)].seed.astype(int).tolist()
            if not available:
                sidebar.error("No saved run exists for this scenario and strategy.")
            else:
                default_seed = 3000 if demo_mode and 3000 in available else available[0]
                seed = int(sidebar.selectbox("Replay seed", available, index=available.index(default_seed)))
        else:
            seed = int(sidebar.number_input("Random seed", min_value=0, max_value=2_000_000_000, value=3000, step=1))
            if sidebar.button("▶ Generate new simulation", type="primary", use_container_width=True):
                try:
                    predictor = cached_predictor(f"{strategy}_onset", str(ROOT)) if strategy in {"gru", "gnn"} else None
                    with st.spinner(f"Running 80 simulated steps · {SCENARIO_LABELS[scenario]} · {STRATEGY_LABELS[strategy]}…"):
                        run_log, run_events, live_metrics = run_one(scenario, seed, strategy,
                            ROOT / "results/phase3_6", steps=80, predictor_override=predictor)
                    st.session_state["phase5_live_run"] = run_log
                    st.session_state["phase5_live_signature"] = (scenario, strategy, seed)
                    st.session_state["phase5_live_metrics"] = live_metrics
                    st.session_state["playback_step"] = 0
                    st.session_state["playback_playing"] = False
                    st.success("Simulation completed. Measurements and controller outputs come from this new run.")
                except Exception as exc:
                    st.error(f"Simulation could not be generated: {exc}")
        speed = float(sidebar.select_slider("Playback interval", options=[0.25, 0.5, 1.0, 1.5, 2.0], value=1.0,
                                             format_func=lambda x: f"{x:g} sec / step"))
    if section == "Live monitor":
        if source == "Run a new simulation":
            signature = (scenario, strategy, seed)
            if st.session_state.get("phase5_live_signature") != signature:
                live_run = None
            else:
                live_run = st.session_state.get("phase5_live_run")
            render_live_view(data, scenario, strategy, seed, source, live_run, speed)
            if live_run is not None:
                st.download_button("Download this run log", live_run.to_csv(index=False).encode(),
                                   file_name=f"phase5_{scenario}_{strategy}_{seed}.csv", mime="text/csv")
                st.caption("Run metrics are computed from this live execution; strategy comparison below remains the saved paired Phase 4 experiment.")
                st.json(st.session_state.get("phase5_live_metrics", {}), expanded=False)
        else:
            render_live_view(data, scenario, strategy, seed, source, None, speed)
    elif section == "Routing comparison":
        scenario = sidebar.selectbox("Scenario for comparison", list(SCENARIOS), format_func=lambda x: SCENARIO_LABELS[x])
        render_comparison(data, scenario)
    elif section == "Model performance":
        render_model_performance(data)
    elif section == "Demo guide":
        st.markdown("## Guided project walkthrough")
        st.write("Use the controls below to replay the saved periodic scenario at seed 3000. Compare Static Dijkstra, GNN proactive, and Reactive routing. This is a deliberately selected demonstration case, not a typical-performance claim.")
        st.markdown("1. Start with **Static Dijkstra** and inspect link utilization, queues, and timeline.\n2. Select **GNN proactive** and play the same scenario and seed. Observe only model probabilities and reroutes present in that actual run.\n3. Select **Reactive routing** and compare observed-trigger behavior.\n4. Open **Routing comparison** to see all five scenarios and paired seeds.")
        st.caption("No animation simulates packet movement. Playback advances through recorded simulator intervals.")
        if source == "Run a new simulation":
            signature = (scenario, strategy, seed)
            live_run = st.session_state.get("phase5_live_run") if st.session_state.get("phase5_live_signature") == signature else None
        else:
            live_run = None
        render_live_view(data, scenario, strategy, seed, source, live_run, speed)
        render_comparison(data, "periodic")
    else:
        scenario = sidebar.selectbox("Scenario for comparison", list(SCENARIOS), format_func=lambda x: SCENARIO_LABELS[x])
        render_model_performance(data)


if __name__ == "__main__":
    main()
