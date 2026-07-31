SHELL := /bin/bash

# The orchestrator is implemented in Python under agent/.
# Historical Go targets were removed 2026-07-17 (the Go tree is not part of this package).

.PHONY: help run-orchestrator run-ai run-ui \
        test test-agent test-ai \
        lint typecheck \
        desktop-deps desktop-dev desktop-build

help:
	@echo ""
	@echo "  Martinus — available make targets"
	@echo "  ─────────────────────────────────────────────"
	@echo "  run-orchestrator   Start the Python orchestrator on :8080 (uvicorn)"
	@echo "  run-ai             Start the FastAPI AI service on :8090 (uvicorn)"
	@echo "  run-ui             Start the NiceGUI dashboard on :8091"
	@echo "  test               Run both Python test suites (agent + ai-service)"
	@echo "  test-agent         Run agent/tests only"
	@echo "  test-ai            Run ai-service/tests only"
	@echo "  lint               flake8 over agent/ and ai-service/app/"
	@echo "  typecheck          mypy over agent/ (per mypy.ini)"
	@echo "  desktop-deps       Install Node deps for the desktop app (npm ci)"
	@echo "  desktop-dev        Run the Tauri desktop app in dev mode"
	@echo "  desktop-build      Bundle the Tauri desktop app for release"
	@echo ""

run-orchestrator:
	python -m uvicorn agent.server:app --host 127.0.0.1 --port 8080

run-ai:
	cd ai-service && python -m uvicorn app.main:app --host 127.0.0.1 --port 8090

run-ui:
	cd ui && python main.py

# The two suites need different import roots.
test-agent:
	python -m pytest agent/tests -q

test-ai:
	cd ai-service && python -m pytest tests -q

test: test-agent test-ai

lint:
	python -m flake8 agent ai-service/app

typecheck:
	python -m mypy agent

desktop-deps:
	cd desktop && npm ci

desktop-dev:
	cd desktop && npm run tauri:dev

desktop-build:
	cd desktop && npm run tauri:build
