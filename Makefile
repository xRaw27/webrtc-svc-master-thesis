SCENARIO          ?= multi-client-throttled
PARTICIPANT       ?= 0

LOG_DIR           := logs
TS                := $(shell date +%Y%m%d-%H%M%S)

SFU_BINARY        := ../livekit/bin/livekit-server
SFU_CONFIG        := livekit-config.yaml

CLIENT_SDK_JS_DIR := ../client-sdk-js
WEBRTCPERF_DIR    := ../webrtcperf

BBB_URL           := https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov.zip

.PHONY: help sfu demo demo-lan scenario scenario-lan clean-chrome epochs report plot verify summarize solve token filter-logs webrtcperf-stack-up webrtcperf-stack-down fetch-media

LAN_HOST_IP       := 192.168.1.11

## Show available targets
help:
	@awk 'BEGIN{FS=":"} /^##/{d=substr($$0,4); next} /^[a-zA-Z_-]+:/{if(d) printf "  %-22s %s\n", $$1, d; d=""}' $(MAKEFILE_LIST)

## Run livekit SFU (full log to logs/sfu-<ts>.log; only warn/error on terminal)
sfu:
	@mkdir -p $(LOG_DIR)
	@echo ">> sfu logging to $(LOG_DIR)/sfu-$(TS).log"
	@BWE_LOG_INTERVAL=100ms $(SFU_BINARY) --config $(SFU_CONFIG) --dev --bind 0.0.0.0 2>&1 \
	  | tee $(LOG_DIR)/sfu-$(TS).log \
	  | grep --line-buffered -iE 'WARN|ERROR|FATAL|PANIC' || true

## Run the Vite demo client app on port 8080 (localhost only)
demo:
	cd $(CLIENT_SDK_JS_DIR) && pnpm examples:demo

## Run the Vite demo client app exposed on the LAN (0.0.0.0:8080)
demo-lan:
	cd $(CLIENT_SDK_JS_DIR) && pnpm examples:demo --host 0.0.0.0

## Kill zombie webrtcperf Chrome instances (they squat debugging ports 9000+N
## after an aborted run, so later sessions time out and never start)
clean-chrome:
	-@pkill -f "/.webrtcperf/chrome/" 2>/dev/null; sleep 1; \
	  echo ">> leftover webrtcperf chrome procs: $$(pgrep -f '/.webrtcperf/chrome/' | wc -l | tr -d ' ')"

## Run a webrtcperf scenario (make scenario SCENARIO=<name>)
scenario: clean-chrome
	cd $(WEBRTCPERF_DIR) && PUBLISHER_SESSIONS=0 yarn start ../webrtc-svc-master-thesis/scenarios/$(SCENARIO).yaml

## Run a webrtcperf scenario against a remote SFU on the LAN (LAN_HOST_IP)
scenario-lan: clean-chrome
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

## Parse the latest SFU log (or LOG=path) into results/<stem>/ (run.json + CSVs)
epochs:
	@log="$(LOG)"; [ -n "$$log" ] || log=$$(ls -r $(LOG_DIR)/sfu-*.log | head -1); \
	echo ">> parsing $$log"; \
	.venv/bin/python analysis/parse_epochs.py "$$log" --include-bwe-log

## Epoch report for the latest run (or LOG=path): parse + charts + open in browser
report:
	@log="$(LOG)"; [ -n "$$log" ] || log=$$(ls -r $(LOG_DIR)/sfu-*.log | head -1); \
	stem=$$(basename "$$log" .log); \
	echo ">> report for $$log"; \
	.venv/bin/python analysis/parse_epochs.py "$$log" --include-bwe-log && \
	.venv/bin/python analysis/plot_epochs.py "results/$$stem" --open

## Cross-source verify: SFU bwe-log vs client-side metrics in Prometheus
## (make verify PARTICIPANT=2 | 0-3, env PROM_URL=http://localhost:9090).
verify:
	@log=$$(ls -r $(LOG_DIR)/sfu-*.log | head -1); \
	from=$(firstword $(subst -, ,$(PARTICIPANT))); \
	to=$(lastword $(subst -, ,$(PARTICIPANT))); \
	for i in $$(seq $$from $$to); do \
	  echo ">> verifying webrtcperf-$$i"; \
	  .venv/bin/python analysis/verify_run.py $$log webrtcperf-$$i; \
	  open -a "Google Chrome" results/verify-webrtcperf-$$i.html; \
	done

## Cross-source verify from webrtcperf CSV (no Prometheus).
## CSV defaults to results/results.csv (webrtcperf detailedStatsPath).
## make verify-csv PARTICIPANT=0-2 [CSV=results/results.csv]
CSV ?= results/results.csv
verify-csv:
	@log=$$(ls -r $(LOG_DIR)/sfu-*.log | head -1); \
	from=$(firstword $(subst -, ,$(PARTICIPANT))); \
	to=$(lastword $(subst -, ,$(PARTICIPANT))); \
	for i in $$(seq $$from $$to); do \
	  echo ">> verifying webrtcperf-$$i from $(CSV)"; \
	  .venv/bin/python analysis/verify_run_csv.py $$log $(CSV) webrtcperf-$$i; \
	  open -a "Google Chrome" results/verify-csv-webrtcperf-$$i.html; \
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

## Split each SFU log into per-participant files (sfu-<ts>-participant-<N>.log
## for webrtcperf-<N>, sfu-<ts>-<identity>.log otherwise). Identities are
## auto-detected from each log. Already-split logs are skipped. Idempotent.
filter-logs:
	@for log in $(LOG_DIR)/sfu-*.log; do \
	  case "$$log" in *-participant-*.log|*-webrtcperf-*.log|*-filtered.log) continue ;; esac; \
	  identities=$$(grep -hoE '"participant":[[:space:]]*"[^"]+"' "$$log" | sed -E 's/.*"([^"]+)"$$/\1/' | sort -u); \
	  for id in $$identities; do \
	    case "$$id" in \
	      webrtcperf-*) suffix="participant-$${id#webrtcperf-}" ;; \
	      *)            suffix="$$id" ;; \
	    esac; \
	    out="$${log%.log}-$${suffix}.log"; \
	    if [ -f "$$out" ]; then continue; fi; \
	    echo ">> filtering $$log -> $$out"; \
	    grep -F "$$id" "$$log" > "$$out" 2>/dev/null; \
	    [ -s "$$out" ] || rm -f "$$out"; \
	  done; \
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
	curl -L -o /tmp/big_buck_bunny.zip $(BBB_URL)
	unzip -o /tmp/big_buck_bunny.zip -d /tmp/big_buck_bunny
	ffmpeg -y -i /tmp/big_buck_bunny/*.mov -an -c copy $@
	rm -rf /tmp/big_buck_bunny.zip /tmp/big_buck_bunny

media/testsrc-720p.mp4:
	@mkdir -p media
	ffmpeg -y -f lavfi -i 'testsrc=size=1280x720:rate=30:duration=120' \
	  -c:v libx264 -pix_fmt yuv420p $@

.DEFAULT_GOAL := help
