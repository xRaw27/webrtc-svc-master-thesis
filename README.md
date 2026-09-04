# webrtc-svc-master-thesis

Experiment harness for a master's-thesis proof-of-concept: **adaptive SVC layer
allocation in a LiveKit SFU**. A modified LiveKit SFU distributes VP9 `L3T3`
video layers across all subscribers in a room under a single global outgoing
bandwidth budget (`B_SFU`), using a max-min round-robin allocator
(`RoomBandwidthAllocator`) fed by per-subscriber bandwidth estimates. This repo
holds everything *around* the SFU: load-test scenarios, headless-client
automation, run configuration, and log/metric analysis.

The SFU-side code lives in the sibling [`livekit/`](../livekit) fork. For an
architectural overview and per-component documentation, see the
[`docs/`](../docs) folder at the workspace root (sibling of this repo).

## Required directory layout

This repo expects to be cloned alongside the other repositories:

```
/
├── webrtc-svc-master-thesis/   <- this repo
├── livekit/                    <- LiveKit SFU fork    https://github.com/xRaw27/livekit
├── webrtcperf/                 <- webrtcperf          https://github.com/vpalmisano/webrtcperf
├── client-sdk-js/              <- LiveKit client SDK   https://github.com/livekit/client-sdk-js
└── docs/                       <- component documentation
```

Default paths in scenario YAMLs and the Makefile are relative to this layout.

## Setup

```bash
# Build the SFU (Go) — produces ../livekit/bin/livekit-server
cd ../livekit && mage build && cd -

# Node deps (used by URL handlers — JWT generation)
pnpm install

# Python poetry env (used for plotting and analysis)
poetry install

# Test media (Big Buck Bunny + generated testsrc-720p) — requires ffmpeg with libx264
make fetch-media
```

## Running

Three terminals, in this order:

```bash
# Terminal 1 — LiveKit SFU (sets BWE_LOG_INTERVAL=100ms, tees full log to logs/sfu-<ts>.log)
make sfu

# Terminal 2 — LiveKit client-sdk demo app (Vite on :8080)
make demo

# Terminal 3 — test scenario (default SCENARIO=multi-client-throttled)
make scenario SCENARIO=multi-client-throttled
```

Available scenarios (in [`scenarios/`](scenarios)): `multi-client-throttled`,
`multi-client-upload-throttled`, `one-publisher-two-subscribers`,
`local-livekit-basic`. `make help` lists all targets.

The global bandwidth budget `B_SFU` and the BWE/probe parameters are set in
[`livekit-config.yaml`](livekit-config.yaml) (`room.room_bandwidth_limit`,
`rtc.congestion_control.stream_allocator.*`). With `room_bandwidth_limit: 0` the
SFU falls back to stock LiveKit per-subscriber greedy allocation.

## Result analysis

After a scenario finishes (operates on the newest `logs/sfu-*.log`):

```bash
# Room/per-subscriber/per-track summary: utilisation, Jain's fairness,
# congestion, layer occupancy, anomaly flags. SCENARIO unlocks throttle stats.
make summarize SCENARIO=multi-client-throttled

# BWE plot for one participant (or a range)
make plot PARTICIPANT=0
make plot PARTICIPANT=0-2

# Cross-source verify: SFU bwe-log vs client-side received metrics.
# `verify` reads from Prometheus; `verify-csv` reads webrtcperf's results.csv.
make verify PARTICIPANT=0-2          # needs the Prometheus stack running
make verify-csv PARTICIPANT=0-2      # offline, from results/results.csv

# Split the latest SFU log into per-participant files (logs/sfu-<ts>-participant-<N>.log)
make filter-logs

# Live tail of PoC observability logs from the newest SFU log
tail -f $(ls -r logs/sfu-*.log | head -1) | grep -E 'rba-log|bwe-log|probe-log'
```

See [`docs/04-observability-logging.md`](../docs/04-observability-logging.md) for
the `rba-log` / `bwe-log` / `probe-log` line schema and
[`docs/06-analysis-and-metrics.md`](../docs/06-analysis-and-metrics.md) for the
analysis tooling.

## Prometheus / Grafana (optional)

```bash
make webrtcperf-stack-up        # starts in the background
# Pushgateway: http://localhost:9091
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3001 (admin/admin)
make webrtcperf-stack-down
```

Scenarios push client-side metrics to the Pushgateway (`prometheusPushgateway`
in each scenario YAML) — they show up automatically once the stack and a
scenario are running. (On Docker Desktop for macOS the bundled `node-exporter`
container crashes on a bind-mount limitation; it is irrelevant to these metrics
and can be ignored.)

## Manual browser test token

```bash
make token   # 24h token for browser-user-1
```

Requires `lk` (LiveKit CLI): `brew install livekit-cli`.
