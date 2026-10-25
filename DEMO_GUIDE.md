# NetScope project demonstration guide

## Before the demonstration

From the repository root, use Python 3.11 or newer and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

The dashboard requires generated artifacts in `results/phase3_6/` and
`results/phase4/`. If they are missing, reproduce them with the Phase 3.6 and
Phase 4 commands in `README.md`. The Phase 4 log is large; the dashboard caches
it after the first load.

## Guided walkthrough

1. Open **Demo guide** in the sidebar. Leave **Guided demonstration** on and
   choose **Replay saved results**.
2. Start with **Static Dijkstra**. The walkthrough uses saved `periodic`
   traffic and seed `3000`. Scrub the step slider or play through the run.
   Inspect one physical link and its two directional measurements, then review
   the traffic and congestion timeline.
3. Select **GNN proactive** at the same scenario and seed. Observe actual
   predicted-risk edges, probabilities, the saved decision threshold, pending
   actions, and applied-route highlights. The currently saved run contains
   seven prediction-triggered route decisions, all applied.
4. Select **Reactive routing** with the same scenario and seed. Compare
   observed triggers and controller outcomes with the GNN replay.
5. Open **Routing comparison** and inspect other scenarios, including sustained
   bottleneck and sudden burst. Try delay, congestion duration, delivery proxy,
   throughput sum, and routing counts.
6. Open **Model performance** to review Phase 3.6 held-out metrics and scenario
   breakdowns.

The periodic seed is a selected teaching example because it contains model
alerts and applied actions. It is not representative of all seeds or
scenarios. The dashboard does not force an alert or route change. From **Live
monitor**, **Run a new simulation** executes the existing simulator and
controller with the selected seed.

## What to point out

- The network has 10 routers and 15 physical full-duplex links; each physical
  link has two direction-specific measurements.
- Green, amber, and red show observed utilization. Dashed violet marks a model
  score above its saved threshold before observed congestion. Orange marks a
  route change recorded as applied.
- Routing decisions have a two-step delay and can be rejected if no feasible
  route meets the capacity reserve. A rejection retains the current route and
  records a reason.
- Playback advances through recorded simulator intervals. It does not
  fabricate packet movement or interpolate predictions.
- Model prediction metrics and Phase 4 routing outcomes are separate
  evaluations.

## What the results support

Across five paired seeds per scenario, the saved Phase 4 aggregates showed
small improvements for the GNN over static Dijkstra on some hop-level proxies,
but GNN routing did not reroute in the sustained-bottleneck or sudden-burst
scenario averages. Reactive routing performed best overall. In Phase 3.6, the
GRU outperformed the GNN on held-out event precision and F1. These results do
not establish general GNN superiority or reliable proactive routing.

Delivery ratio and packet loss are hop-level proxies; throughput is a sum of
link-hop Mbps. The simulator does not track packet identity end-to-end, so do
not present these as end-to-end delivery or throughput. Queue delay is an
interval-level estimate. All values are simulated. Five seeds per scenario
provide limited statistical power, and static-routing training creates
possible distribution shift after routes change.

## Short presentation script

“This dashboard replays a simulated 10-router network. Each physical link has
independent directional queues and measurements. I’ll start with static
Dijkstra, then use the same periodic scenario and seed to inspect saved GNN
predictions and applied route decisions, and compare them with reactive
routing. Predictions use causal history, the saved Phase 3.6 checkpoint, and
its validation-selected threshold. This periodic example is selected to make
the event path visible; the scenario comparison shows it is not representative.
The GNN had only modest aggregate gains over static routing in the Phase 4
simulation, while reactive routing performed better. Delivery and throughput
figures are hop-level proxies, not end-to-end measurements.”
