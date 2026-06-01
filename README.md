# webrtc-svc-master-thesis

## Required directory layout

This repo expects to be cloned alongside the other repositories:

```
/
├── webrtc-svc-master-thesis/   <- this repo
├── livekit/                    <- livekit SFU fork     https://github.com/xRaw27/livekit
├── webrtcperf/                 <- webrtcperf           https://github.com/vpalmisano/webrtcperf
└── client-sdk-js/              <- livekit client sdk   https://github.com/livekit/client-sdk-js
```

Default paths in scenario YAMLs and the Makefile are relative to this layout.

## Setup

```bash
# Node deps (used by URL handlers — JWT generation)
pnpm install

# Python peotry env (used for plotting and analysis)
poetry install

# Test media (Big Buck Bunny + testsrc-720p) — requires ffmpeg with libx264
make fetch-media
```

## Running

Three terminals, in this order:

```bash
# Terminal 1 — livekit SFU
make sfu

# Terminal 2 — livekit client sdk demo app (Vite on :8080)
make demo

# Terminal 3 — test scenario
make scenario SCENARIO=subscriber-only-throttled
```

`make help` lists all available targets.

## Result analysis

After a scenario finishes:

```bash
# BWE plot for one participant-0
make plot PARTICIPANT=0

# Or a range of participants
make plot PARTICIPANT=0-3

# Live tail of PoC observability logs (bwe-log, rba-log, ...) from the latest SFU log
make watch-sfu-logs
```

## Prometheus / Grafana (optional)

```bash
make webrtcperf-stack-up        # starts in the background
# Pushgateway: http://localhost:9091
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3001 (admin/admin)
make webrtcperf-stack-down
```

Scenarios are configured to push metrics to `localhost:9091` — they show up
automatically once the stack and a scenario are running.

## Manual browser test token

```bash
make token   # 24h token for browser-user-1
```

Requires `lk` (LiveKit CLI): `brew install livekit-cli`.
