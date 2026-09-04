#!/usr/bin/env python3
"""Interactive evaluation report for one parsed SFU run (docs/09 Part 1).

Consumes the directory written by ``parse_epochs.py`` and renders a single
self-contained ``report.html`` with a shared time axis:

  1. Bandwidth estimates       per receiver: raw BWE + effective b_hat (rho*BWE),
                               congestion markers, probe/bwe-change event overlays
  2. Allocation vs capacity    per receiver: decided kbps (step) vs its b_hat
                               ceiling; optional SFU expected-usage overlay when
                               bwe_samples.csv exists (toggle via legend)
  3. Bridge budget             total decided kbps vs the B limit and the
                               stock-greedy demand
  4. Allocated levels          one row per subscriber, a step line per incoming
                               track (colored by publisher), y = quality levels
                               0..5 mapped to the sf ladder
  5. Reception quality         Q_corr per receiver over time (+ Jain(Q) toggle)
  6. Level changes             per-epoch bars split into penalized (within the
                               stability window) and free changes
  7. Solver diagnostics        spent vs budget cells (collapsed detail)

Colors follow the participant identity across every panel (fixed categorical
slots, validated palette). Bitrates shown are the allocator's *decided* values;
see docs/09 §1.1 for the decided-vs-actual accounting note.

Usage:
    .venv/bin/python analysis/plot_epochs.py results/<log-stem>/
        [--export png|svg] [--open] [--out report.html]

``--export`` additionally writes static images (full report + one file per
panel) next to the HTML — requires the ``kaleido`` package.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import webbrowser

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LEVEL_LABELS = ["paused", "S0T0", "S1T0", "S1T1", "S2T1", "S2T2"]

# Validated categorical palette (dataviz reference instance, light mode).
# Slots 1..3 (participants) validate under all-pairs; violet = bridge total,
# red = penalized changes. Reference lines use neutral ink.
IDENTITY_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
C_BRIDGE = "#4a3aa7"
C_PENALIZED = "#e34948"
C_FREE = "#b9b7b0"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#eceae6"
SURFACE = "#fcfcfb"


def load_run(results_dir: str) -> dict:
    def read(name: str) -> pd.DataFrame:
        path = os.path.join(results_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    with open(os.path.join(results_dir, "run.json")) as fh:
        run = json.load(fh)
    data = {
        "meta": run["meta"],
        "params": run.get("params") or {},
        "epochs": read("epochs.csv"),
        "receivers": read("receivers.csv"),
        "pairs": read("pairs.csv"),
        "events": read("events.csv"),
        "samples": read("bwe_samples.csv"),
    }
    ids = sorted(data["receivers"]["identity"].unique()) if not data["receivers"].empty else []
    data["identities"] = ids
    data["colors"] = {ident: IDENTITY_SLOTS[i % len(IDENTITY_SLOTS)] for i, ident in enumerate(ids)}
    # b_hat rows where the raw estimate is 0 carry the SFU's synthetic
    # 100 Mbps "no estimate yet" fallback — mask it so it doesn't wreck the
    # y-scale (the raw 0 stays visible and honest)
    rc = data["receivers"]
    if not rc.empty:
        data["receivers"] = rc.assign(bHatKbps=rc.bHatKbps.where(rc.bweRawBps > 0))
    # default x-range: the active part of the run (idle empty-room epochs at
    # the edges remain reachable via autoscale)
    ep = data["epochs"]
    active = ep[ep.nPairs > 0] if not ep.empty else ep
    src = active if not active.empty else ep
    data["xrange"] = [src.t.min() - 1.5, src.t.max() + 1.5] if not src.empty else None
    return data


def _legend(fig_state: set, group: str) -> bool:
    """Show each legend group exactly once across the whole figure."""
    if group in fig_state:
        return False
    fig_state.add(group)
    return True


def panel_bwe(fig, row, d, state) -> None:
    ev = d["events"]
    for ident in d["identities"]:
        rc = d["receivers"][d["receivers"].identity == ident]
        c = d["colors"][ident]
        fig.add_trace(go.Scatter(
            x=rc.t, y=rc.bweRawBps / 1000.0, name=ident, legendgroup=ident,
            showlegend=_legend(state, ident), mode="lines",
            line=dict(color=c, width=2),
            hovertemplate=f"{ident} BWE %{{y:.0f}} kbps<extra></extra>"), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=rc.t, y=rc.bHatKbps, name=f"{ident} b̂", legendgroup=ident, showlegend=False,
            mode="lines", line=dict(color=c, width=1.2, dash="dash"),
            hovertemplate=f"{ident} b̂ %{{y:.0f}} kbps<extra></extra>"), row=row, col=1)
        cong = rc[rc.congestionState != "NONE"]
        if not cong.empty:
            fig.add_trace(go.Scatter(
                x=cong.t, y=cong.bweRawBps / 1000.0, name="congested",
                legendgroup="congested", showlegend=_legend(state, "congested"),
                mode="markers", marker=dict(symbol="x-thin", size=9, line=dict(color=INK, width=1.6)),
                hovertemplate=f"{ident} %{{customdata}}<extra></extra>",
                customdata=cong.congestionState), row=row, col=1)
    if not ev.empty:
        bc = ev[ev.type == "bwe-change"].copy()
        if not bc.empty:
            bc["y"] = [json.loads(f).get("newBWEBps", 0) / 1000.0 for f in bc.fields]
            fig.add_trace(go.Scatter(
                x=bc.t, y=bc.y, name="bwe-change", legendgroup="bwe-change",
                showlegend=_legend(state, "bwe-change"), visible="legendonly",
                mode="markers",
                marker=dict(symbol="diamond", size=8,
                            color=[d["colors"].get(i, INK_2) for i in bc.identity]),
                hovertemplate="bwe-change %{customdata}<extra></extra>",
                customdata=[f"{i}: Δ{json.loads(f).get('bweDeltaBps', 0) / 1000:+.0f} kbps"
                            for i, f in zip(bc.identity, bc.fields)]), row=row, col=1)
        pf = ev[ev.type == "probe-finalized"].copy()
        if not pf.empty:
            pf["y"] = [json.loads(f).get("channelCapacityBps", 0) / 1000.0 for f in pf.fields]
            fig.add_trace(go.Scatter(
                x=pf.t, y=pf.y, name="probe finalized", legendgroup="probes",
                showlegend=_legend(state, "probes"), visible="legendonly",
                mode="markers",
                marker=dict(symbol="circle-open", size=7,
                            color=[d["colors"].get(i, INK_2) for i in pf.identity]),
                hovertemplate="probe %{customdata}<extra></extra>",
                customdata=[f"{i}: {json.loads(f).get('probeSignal', '?')}"
                            for i, f in zip(pf.identity, pf.fields)]), row=row, col=1)
    fig.update_yaxes(title_text="kbps", rangemode="tozero", row=row, col=1)


def panel_alloc(fig, row, d, state) -> None:
    for ident in d["identities"]:
        rc = d["receivers"][d["receivers"].identity == ident]
        c = d["colors"][ident]
        fig.add_trace(go.Scatter(
            x=rc.t, y=rc.allocatedKbps, name=ident, legendgroup=ident, showlegend=False,
            mode="lines", line=dict(color=c, width=2, shape="hv"),
            hovertemplate=f"{ident} allocated %{{y:.0f}} kbps<extra></extra>"), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=rc.t, y=rc.bHatKbps, name=f"{ident} b̂", legendgroup=ident, showlegend=False,
            mode="lines", line=dict(color=c, width=1.2, dash="dash"),
            hovertemplate=f"{ident} b̂ %{{y:.0f}} kbps<extra></extra>"), row=row, col=1)
    s = d["samples"]
    if not s.empty:
        subs = s[s.kind == "sub"]
        for ident in d["identities"]:
            ss = subs[subs.identity == ident]
            if ss.empty:
                continue
            fig.add_trace(go.Scatter(
                x=ss.t, y=ss.expectedUsageBps / 1000.0, name="SFU expected usage",
                legendgroup="expected", showlegend=_legend(state, "expected"),
                visible="legendonly", mode="lines",
                line=dict(color=d["colors"][ident], width=1), opacity=0.45,
                hovertemplate=f"{ident} expected %{{y:.0f}} kbps<extra></extra>"), row=row, col=1)
    fig.update_yaxes(title_text="kbps", rangemode="tozero", row=row, col=1)


def panel_bridge(fig, row, d, state) -> None:
    ep = d["epochs"]
    budget = d["params"].get("budgetKbps")
    fig.add_trace(go.Scatter(
        x=ep.t, y=ep.totalKbps, name="bridge allocated", legendgroup="bridge",
        showlegend=_legend(state, "bridge"), mode="lines",
        line=dict(color=C_BRIDGE, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(74,58,167,0.10)",
        hovertemplate="allocated %{y:.0f} kbps<extra></extra>"), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=ep.t, y=ep.demandKbps, name="greedy demand", legendgroup="demand",
        showlegend=_legend(state, "demand"), mode="lines",
        line=dict(color=INK_2, width=1.2, dash="dot"),
        hovertemplate="demand %{y:.0f} kbps<extra></extra>"), row=row, col=1)
    if budget:
        fig.add_trace(go.Scatter(
            x=[ep.t.min(), ep.t.max()], y=[budget, budget], name="budget B",
            legendgroup="budget", showlegend=_legend(state, "budget"), mode="lines",
            line=dict(color=INK, width=1.4, dash="dash"),
            hovertemplate="B %{y:.0f} kbps<extra></extra>"), row=row, col=1)
    fig.update_yaxes(title_text="kbps", rangemode="tozero", row=row, col=1)


def panel_levels(fig, row, d, sub_ident, state) -> None:
    pr = d["pairs"][d["pairs"].subIdentity == sub_ident]
    for pub in sorted(pr.publisherIdentity.unique()):
        pp = pr[pr.publisherIdentity == pub]
        c = d["colors"].get(pub, INK_2)
        fig.add_trace(go.Scatter(
            x=pp.t, y=pp.level, name=pub, legendgroup=pub, showlegend=False,
            mode="lines", line=dict(color=c, width=2, shape="hv"),
            customdata=list(zip(pp.layer, pp.bitrateKbps.round(0), pp.tau, pp.w.round(3))),
            hovertemplate=(f"{pub} → {sub_ident}: %{{customdata[0]}}"
                           " (%{customdata[1]} kbps, τ=%{customdata[2]}, w=%{customdata[3]})<extra></extra>")),
            row=row, col=1)
    fig.update_yaxes(
        title_text=sub_ident.replace("webrtcperf-", "sub "),
        tickmode="array", tickvals=list(range(6)), ticktext=LEVEL_LABELS,
        range=[-0.4, 5.4], row=row, col=1)


def panel_quality(fig, row, d, state) -> None:
    for ident in d["identities"]:
        rc = d["receivers"][d["receivers"].identity == ident]
        fig.add_trace(go.Scatter(
            x=rc.t, y=rc.qCorr, name=ident, legendgroup=ident, showlegend=False,
            mode="lines", line=dict(color=d["colors"][ident], width=2, shape="hv"),
            hovertemplate=f"{ident} Q %{{y:.3f}}<extra></extra>"), row=row, col=1)
    ep = d["epochs"]
    fig.add_trace(go.Scatter(
        x=ep.t, y=ep.jainQ, name="Jain(Q)", legendgroup="jain",
        showlegend=_legend(state, "jain"), visible="legendonly",
        mode="lines", line=dict(color=INK_2, width=1.2, dash="dash"),
        hovertemplate="Jain(Q) %{y:.3f}<extra></extra>"), row=row, col=1)
    fig.update_yaxes(title_text="Q̃", row=row, col=1)


def panel_churn(fig, row, d, state) -> None:
    ep = d["epochs"]
    free = ep.levelChanges - ep.penalizedChanges
    fig.add_trace(go.Bar(
        x=ep.t, y=ep.penalizedChanges, name="changes in stability window",
        legendgroup="penalized", showlegend=_legend(state, "penalized"),
        marker=dict(color=C_PENALIZED, line=dict(color=SURFACE, width=1)),
        hovertemplate="penalized %{y}<extra></extra>"), row=row, col=1)
    fig.add_trace(go.Bar(
        x=ep.t, y=free, name="free changes (τ ≥ T_stab)",
        legendgroup="freechg", showlegend=_legend(state, "freechg"),
        marker=dict(color=C_FREE, line=dict(color=SURFACE, width=1)),
        hovertemplate="free %{y}<extra></extra>"), row=row, col=1)
    fig.update_yaxes(title_text="level changes", rangemode="tozero", row=row, col=1)


def panel_cells(fig, row, d, state) -> None:
    ep = d["epochs"]
    fig.add_trace(go.Scatter(
        x=ep.t, y=ep.spentCells, name="spent cells", legendgroup="cells",
        showlegend=_legend(state, "cells"), mode="lines",
        line=dict(color=INK_2, width=1.6, shape="hv"),
        hovertemplate="spent %{y} cells<extra></extra>"), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=ep.t, y=ep.budgetCells, name="budget cells", legendgroup="cellsb",
        showlegend=_legend(state, "cellsb"), mode="lines",
        line=dict(color=INK, width=1.2, dash="dash"),
        hovertemplate="budget %{y} cells<extra></extra>"), row=row, col=1)
    fig.update_yaxes(title_text="δ-cells", rangemode="tozero", row=row, col=1)


PANEL_ORDER = ["bwe", "alloc", "bridge", "levels", "quality", "churn", "cells"]
PANEL_TITLES = {
    "bwe": "Bandwidth estimates per receiver (solid: BWE, dashed: b̂ = ρ·BWE)",
    "alloc": "Decided allocation vs downlink capacity per receiver",
    "bridge": "Bridge egress (decided) vs budget B",
    "levels": "Allocated quality levels (per subscriber, colored by publisher)",
    "quality": "Corrected reception quality Q̃ per receiver",
    "churn": "Level changes per epoch",
    "cells": "Solver cell budget",
}


def build_report(d: dict) -> go.Figure:
    n_subs = len(d["identities"])
    rows = 3 + n_subs + 3
    weights = [3.0, 3.0, 2.2] + [1.8] * n_subs + [3.0, 1.6, 1.4]
    titles = [PANEL_TITLES["bwe"], PANEL_TITLES["alloc"], PANEL_TITLES["bridge"]]
    titles += [PANEL_TITLES["levels"] if i == 0 else "" for i in range(n_subs)]
    titles += [PANEL_TITLES["quality"], PANEL_TITLES["churn"], PANEL_TITLES["cells"]]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.022,
        row_heights=[w / sum(weights) for w in weights], subplot_titles=titles)
    state: set = set()
    panel_bwe(fig, 1, d, state)
    panel_alloc(fig, 2, d, state)
    panel_bridge(fig, 3, d, state)
    for i, ident in enumerate(d["identities"]):
        panel_levels(fig, 4 + i, d, ident, state)
    panel_quality(fig, 4 + n_subs, d, state)
    panel_churn(fig, 5 + n_subs, d, state)
    panel_cells(fig, 6 + n_subs, d, state)

    p = d["params"]
    subtitle = (f"B = {p.get('budgetKbps', '?')} kbps · δ = {p.get('deltaKbps', '?')} kbps · "
                f"α = {p.get('alpha', '?')} · T_stab = {p.get('tStab', '?')} · "
                f"φ_k = {p.get('phiK', '?')} · ρ = {p.get('rho', '?')} · "
                f"epoch = {p.get('epochMs', '?')} ms · room {d['meta'].get('room', '?')} · "
                f"{d['meta'].get('epochs', '?')} epochs")
    stem = os.path.basename(d["meta"].get("log", "run")).removesuffix(".log")
    height = 170 * rows + 230
    fig.update_layout(
        title=dict(text=f"SVC allocation run — {stem}", font=dict(color=INK, size=19),
                   yref="container", y=1 - 18 / height, yanchor="top"),
        height=height, width=1360,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=INK_2, size=12),
        barmode="stack", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.004, xanchor="left", x=0,
                    font=dict(size=12, color=INK)),
        margin=dict(l=70, r=30, t=175, b=80),
    )
    fig.add_annotation(  # params footer (sits in the bottom margin)
        text=subtitle, xref="paper", yref="paper", x=0, y=-0.028,
        xanchor="left", yanchor="top", showarrow=False, font=dict(size=12, color=INK_2))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    if d.get("xrange"):
        fig.update_xaxes(range=d["xrange"])
    fig.update_xaxes(title_text="t [s since first epoch]", row=rows, col=1)
    fig.update_annotations(font=dict(size=13, color=INK))
    return fig


def build_panel(d: dict, name: str) -> go.Figure:
    """Standalone figure for one panel (used by --export)."""
    state: set = set()
    if name == "levels":
        n = len(d["identities"])
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for i, ident in enumerate(d["identities"]):
            panel_levels(fig, 1 + i, d, ident, state)
        height = 190 * n + 130
    else:
        fig = make_subplots(rows=1, cols=1)
        {"bwe": panel_bwe, "alloc": panel_alloc, "bridge": panel_bridge,
         "quality": panel_quality, "churn": panel_churn, "cells": panel_cells}[name](fig, 1, d, state)
        height = 420
    fig.update_layout(
        title=dict(text=PANEL_TITLES[name], font=dict(color=INK, size=16)),
        height=height, width=1100, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=INK_2, size=12), barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK)),
        margin=dict(l=70, r=30, t=90, b=50))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if d.get("xrange"):
        fig.update_xaxes(range=d["xrange"])
    fig.update_xaxes(title_text="t [s]", row=(len(d["identities"]) if name == "levels" else 1), col=1)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="directory written by parse_epochs.py")
    ap.add_argument("--out", default="report.html", help="output HTML name (inside results_dir)")
    ap.add_argument("--export", choices=["png", "svg"], help="also write static images (needs kaleido)")
    ap.add_argument("--open", action="store_true", help="open the report in the default browser")
    args = ap.parse_args()

    d = load_run(args.results_dir)
    if d["epochs"].empty or not d["identities"]:
        print(f"error: no epochs/receivers in {args.results_dir}", file=sys.stderr)
        return 1
    if len(d["identities"]) > 3:
        print(f"note: {len(d['identities'])} participants — categorical palette is "
              "all-pairs-validated for 3; interpret colors with care", file=sys.stderr)

    fig = build_report(d)
    out_html = os.path.join(args.results_dir, args.out)
    fig.write_html(out_html, include_plotlyjs=True)
    print(f"wrote {out_html}")

    if args.export:
        try:
            fig.write_image(os.path.join(args.results_dir, f"report.{args.export}"),
                            width=fig.layout.width, height=fig.layout.height, scale=2)
            for name in PANEL_ORDER:
                pfig = build_panel(d, name)
                pfig.write_image(os.path.join(args.results_dir, f"fig-{name}.{args.export}"),
                                 width=pfig.layout.width, height=pfig.layout.height, scale=2)
            print(f"wrote report.{args.export} + {len(PANEL_ORDER)} panel images")
        except Exception as exc:  # kaleido missing or renderer failure
            print(f"static export failed ({exc}); install kaleido", file=sys.stderr)

    if args.open:
        webbrowser.open("file://" + os.path.abspath(out_html))
    return 0


if __name__ == "__main__":
    sys.exit(main())
