SCENARIO          ?= multi-client-throttled
PARTICIPANT       ?= 0

LOG_DIR           := logs
TS                := $(shell date +%Y%m%d-%H%M%S)

SFU_BINARY        := ../livekit/bin/livekit-server
SFU_CONFIG        := livekit-config.yaml

CLIENT_SDK_JS_DIR := ../client-sdk-js
WEBRTCPERF_DIR    := ../webrtcperf

BBB_URL           := https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov

.PHONY: help sfu demo demo-lan scenario scenario-lan plot summarize token filter-logs webrtcperf-stack-up webrtcperf-stack-down fetch-media

LAN_HOST_IP       := 192.168.1.28

## Show available targets
help:
	@awk 'BEGIN{FS=":"} /^##/{d=substr($$0,4); next} /^[a-zA-Z_-]+:/{if(d) printf "  %-22s %s\n", $$1, d; d=""}' $(MAKEFILE_LIST)

## Run livekit SFU (log to logs/sfu-<ts>.log)
sfu:
	@mkdir -p $(LOG_DIR)
	BWE_LOG_INTERVAL=100ms $(SFU_BINARY) --config $(SFU_CONFIG) --dev --bind 0.0.0.0 2>&1 | tee $(LOG_DIR)/sfu-$(TS).log

## Run the Vite demo client app on port 8080 (localhost only)
demo:
	cd $(CLIENT_SDK_JS_DIR) && pnpm examples:demo

## Run the Vite demo client app exposed on the LAN (0.0.0.0:8080)
demo-lan:
	cd $(CLIENT_SDK_JS_DIR) && pnpm examples:demo --host 0.0.0.0

## Run a webrtcperf scenario (make scenario SCENARIO=<name>)
scenario:
	cd $(WEBRTCPERF_DIR) && PUBLISHER_SESSIONS=0 yarn start ../webrtc-svc-master-thesis/scenarios/$(SCENARIO).yaml

## Run a webrtcperf scenario against a remote SFU on the LAN (LAN_HOST_IP)
scenario-lan:
	cd $(WEBRTCPERF_DIR) && PUBLISHER_SESSIONS=0 \
	  LIVEKIT_URL=ws://$(LAN_HOST_IP):7880 \
	  APP_URL=http://$(LAN_HOST_IP):8080 \
	  yarn start ../webrtc-svc-master-thesis/scenarios/$(SCENARIO).yaml

## Generate BWE plot for a participant index or range (make plot PARTICIPANT=2 | 0-3)
plot:
	@log=$$(ls -r $(LOG_DIR)/sfu-*.log | head -1); \
	from=$(firstword $(subst -, ,$(PARTICIPANT))); \
	to=$(lastword $(subst -, ,$(PARTICIPANT))); \
	for i in $$(seq $$from $$to); do \
	  echo ">> plotting webrtcperf-$$i"; \
	  .venv/bin/python analysis/plot-bwe.py $$log webrtcperf-$$i; \
	  open -a "Google Chrome" results/bwe-webrtcperf-$$i.html; \
	done

## Summarise the latest run (room-level, per-sub, per-track, anomalies). SCENARIO is optional but unlocks throttle stats.
summarize:
	@log=$$(ls -r $(LOG_DIR)/sfu-*.log | head -1); \
	echo ">> summarising $$log"; \
	if [ -n "$(SCENARIO)" ]; then \
	  .venv/bin/python analysis/summarize_run.py $$log --scenario scenarios/$(SCENARIO).yaml; \
	else \
	  .venv/bin/python analysis/summarize_run.py $$log; \
	fi

## Generate a 24h token for browser-user-1 (devkey/secret)
token:
	lk token create --api-key devkey --api-secret secret --join \
	  --room test-room --identity browser-user-1 --valid-for 24h

## Filter PoC observability lines (bwe-log, rba-log, ...) from each SFU log into sfu-<ts>-filtered.log
filter-logs:
	@for log in $(LOG_DIR)/sfu-*.log; do \
	  case "$$log" in *-filtered.log) continue ;; esac; \
	  filtered=$${log%.log}-filtered.log; \
	  if [ -f "$$filtered" ]; then continue; fi; \
	  echo ">> filtering $$log -> $$filtered"; \
	  grep -E '[a-z-]+-log:' "$$log" | while IFS= read -r line; do \
	    prefix=$${line%%\{*}; \
	    json=\{$${line#*\{}; \
	    printf '%s\n' "$$prefix" | sed 's/[[:space:]]*$$//'; \
	    printf '%s' "$$json" | jq .; \
	    printf '\n'; \
	  done > "$$filtered"; \
	done

## Start webrtcperf Prometheus/Grafana stack in the background
webrtcperf-stack-up:
	cd $(WEBRTCPERF_DIR)/prometheus-stack && docker compose up -d

## Stop webrtcperf Prometheus/Grafana stack
webrtcperf-stack-down:
	cd $(WEBRTCPERF_DIR)/prometheus-stack && docker compose down

## Fetch test media (Big Buck Bunny + generate testsrc-720p)
fetch-media: media/big_buck_bunny_1080p_h264.mp4 media/testsrc-720p.mp4

media/big_buck_bunny_1080p_h264.mp4:
	@mkdir -p media
	curl -L -o /tmp/big_buck_bunny.mov $(BBB_URL)
	ffmpeg -y -i /tmp/big_buck_bunny.mov -c copy $@
	rm /tmp/big_buck_bunny.mov

media/testsrc-720p.mp4:
	@mkdir -p media
	ffmpeg -y -f lavfi -i 'testsrc=size=1280x720:rate=30:duration=120' \
	  -c:v libx264 -pix_fmt yuv420p $@

.DEFAULT_GOAL := help
