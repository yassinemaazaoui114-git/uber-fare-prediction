.PHONY: help setup install clean lint format security test train train-fast predict server-start server-stop server-restart server-status logs open-browser full-start stop-all

# ============================================================================
# 🎯 UBER FARE PREDICTION - MLOPS PROJECT
# ============================================================================

PYTHON := python3
PIP := $(PYTHON) -m pip
VENV := venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP_VENV := $(VENV_BIN)/pip

# Ports
API_PORT := 8000
SERVER_PID_FILE := .api_server.pid

# Colors
COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_CYAN := \033[36m
COLOR_YELLOW := \033[33m
COLOR_RED := \033[31m
COLOR_BLUE := \033[34m

# ============================================================================
# 📖 HELP & DOCUMENTATION
# ============================================================================

help: ## 📖 Show this help message
	@echo "$(COLOR_CYAN)"
	@echo "╔════════════════════════════════════════════════════════════════════╗"
	@echo "║           🚕 UBER FARE PREDICTION - MLOPS TOOLKIT 🚕              ║"
	@echo "╚════════════════════════════════════════════════════════════════════╝"
	@echo "$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)🚀 QUICK START:$(COLOR_RESET)"
	@echo "  make full-start      - 🎬 Setup + Train + Start Server + Open Browser"
	@echo "  make stop-all        - 🛑 Stop all services"
	@echo ""
	@echo "$(COLOR_CYAN)📦 SETUP & INSTALLATION:$(COLOR_RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /SETUP|INSTALLATION/ {print "  make " $$1 " - " $$2}'
	@echo ""
	@echo "$(COLOR_YELLOW)🤖 MODEL TRAINING:$(COLOR_RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /TRAINING/ {print "  make " $$1 " - " $$2}'
	@echo ""
	@echo "$(COLOR_BLUE)🌐 SERVER MANAGEMENT:$(COLOR_RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /SERVER/ {print "  make " $$1 " - " $$2}'
	@echo ""
	@echo "$(COLOR_GREEN)🎨 CODE QUALITY:$(COLOR_RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /QUALITY/ {print "  make " $$1 " - " $$2}'
	@echo ""
	@echo "$(COLOR_RED)🧹 CLEANUP:$(COLOR_RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /CLEANUP/ {print "  make " $$1 " - " $$2}'
	@echo ""

# ============================================================================
# 📦 SETUP & INSTALLATION
# ============================================================================

setup: ## 📦 [SETUP] Complete project setup (venv + dependencies)
	@echo "$(COLOR_CYAN)📦 Setting up project environment...$(COLOR_RESET)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(COLOR_GREEN)Creating virtual environment...$(COLOR_RESET)"; \
		$(PYTHON) -m venv $(VENV); \
	else \
		echo "$(COLOR_YELLOW)Virtual environment already exists$(COLOR_RESET)"; \
	fi
	@echo "$(COLOR_GREEN)Installing dependencies...$(COLOR_RESET)"
	@$(PIP_VENV) install --upgrade pip
	@$(PIP_VENV) install -r requirements.txt
	@mkdir -p models data static logs
	@echo "$(COLOR_GREEN)✅ Setup complete!$(COLOR_RESET)"

install: setup ## 📦 [SETUP] Alias for setup

# ============================================================================
# 🤖 MODEL TRAINING
# ============================================================================

train: ## 🤖 [TRAINING] Train all models (9 models, ~5 minutes)
	@echo "$(COLOR_CYAN)🚀 Training all models (fast mode)...$(COLOR_RESET)"
	@$(PYTHON_VENV) main.py
	@echo "$(COLOR_GREEN)✅ Training complete!$(COLOR_RESET)"

train-fast: train ## 🤖 [TRAINING] Alias for train

# ============================================================================
# 🌐 SERVER MANAGEMENT (BACKGROUND PROCESS)
# ============================================================================

server-start: ## 🌐 [SERVER] Start FastAPI server in background
	@echo "$(COLOR_CYAN)🌐 Starting FastAPI server in background...$(COLOR_RESET)"
	@if [ -f $(SERVER_PID_FILE) ]; then \
		if ps -p $$(cat $(SERVER_PID_FILE)) > /dev/null 2>&1; then \
			echo "$(COLOR_YELLOW)⚠️  Server already running (PID: $$(cat $(SERVER_PID_FILE)))$(COLOR_RESET)"; \
			exit 0; \
		else \
			rm -f $(SERVER_PID_FILE); \
		fi; \
	fi
	@nohup $(VENV_BIN)/uvicorn api.app:app --host 0.0.0.0 --port $(API_PORT) > logs/server.log 2>&1 & echo $$! > $(SERVER_PID_FILE)
	@sleep 2
	@if ps -p $$(cat $(SERVER_PID_FILE)) > /dev/null 2>&1; then \
		echo "$(COLOR_GREEN)✅ Server started successfully!$(COLOR_RESET)"; \
		echo "$(COLOR_CYAN)   PID: $$(cat $(SERVER_PID_FILE))$(COLOR_RESET)"; \
		echo "$(COLOR_CYAN)   URL: http://127.0.0.1:$(API_PORT)$(COLOR_RESET)"; \
		echo "$(COLOR_CYAN)   Logs: tail -f logs/server.log$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_RED)❌ Failed to start server$(COLOR_RESET)"; \
		rm -f $(SERVER_PID_FILE); \
		exit 1; \
	fi

server-stop: ## 🌐 [SERVER] Stop FastAPI server
	@echo "$(COLOR_CYAN)🛑 Stopping FastAPI server...$(COLOR_RESET)"
	@if [ -f $(SERVER_PID_FILE) ]; then \
		if ps -p $$(cat $(SERVER_PID_FILE)) > /dev/null 2>&1; then \
			kill $$(cat $(SERVER_PID_FILE)); \
			rm -f $(SERVER_PID_FILE); \
			echo "$(COLOR_GREEN)✅ Server stopped$(COLOR_RESET)"; \
		else \
			echo "$(COLOR_YELLOW)⚠️  Server not running$(COLOR_RESET)"; \
			rm -f $(SERVER_PID_FILE); \
		fi; \
	else \
		echo "$(COLOR_YELLOW)⚠️  Server not running$(COLOR_RESET)"; \
	fi

server-restart: server-stop server-start ## 🌐 [SERVER] Restart FastAPI server

server-status: ## 🌐 [SERVER] Check server status
	@echo "$(COLOR_CYAN)📊 Checking server status...$(COLOR_RESET)"
	@if [ -f $(SERVER_PID_FILE) ]; then \
		if ps -p $$(cat $(SERVER_PID_FILE)) > /dev/null 2>&1; then \
			echo "$(COLOR_GREEN)✅ Server is running$(COLOR_RESET)"; \
			echo "   PID: $$(cat $(SERVER_PID_FILE))"; \
			echo "   URL: http://127.0.0.1:$(API_PORT)"; \
			echo "   Memory: $$(ps -o rss= -p $$(cat $(SERVER_PID_FILE)) | awk '{printf "%.1f MB", $$1/1024}')"; \
		else \
			echo "$(COLOR_RED)❌ Server not running (stale PID file)$(COLOR_RESET)"; \
			rm -f $(SERVER_PID_FILE); \
		fi; \
	else \
		echo "$(COLOR_RED)❌ Server not running$(COLOR_RESET)"; \
	fi

logs: ## 🌐 [SERVER] Show server logs (live)
	@echo "$(COLOR_CYAN)📋 Server logs (Ctrl+C to exit):$(COLOR_RESET)"
	@if [ -f logs/server.log ]; then \
		tail -f logs/server.log; \
	else \
		echo "$(COLOR_RED)❌ No logs found. Start server first with 'make server-start'$(COLOR_RESET)"; \
	fi

open-browser: ## 🌐 [SERVER] Open web interface in browser
	@echo "$(COLOR_CYAN)🌐 Opening browser...$(COLOR_RESET)"
	@if command -v xdg-open > /dev/null; then \
		xdg-open http://127.0.0.1:$(API_PORT); \
	elif command -v open > /dev/null; then \
		open http://127.0.0.1:$(API_PORT); \
	else \
		echo "$(COLOR_YELLOW)⚠️  Please open manually: http://127.0.0.1:$(API_PORT)$(COLOR_RESET)"; \
	fi

# ============================================================================
# 🎬 QUICK START COMMANDS
# ============================================================================

full-start: setup train server-start open-browser ## 🎬 Complete setup + train + start server + open browser
	@echo ""
	@echo "$(COLOR_GREEN)╔════════════════════════════════════════════════════════════════════╗$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)║              🎉 PROJECT FULLY STARTED! 🎉                         ║$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)╚════════════════════════════════════════════════════════════════════╝$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_CYAN)✅ Environment ready$(COLOR_RESET)"
	@echo "$(COLOR_CYAN)✅ Models trained$(COLOR_RESET)"
	@echo "$(COLOR_CYAN)✅ Server running at: http://127.0.0.1:$(API_PORT)$(COLOR_RESET)"
	@echo "$(COLOR_CYAN)✅ Browser opened$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)📝 Useful commands:$(COLOR_RESET)"
	@echo "   make logs          - View server logs"
	@echo "   make server-status - Check server status"
	@echo "   make server-stop   - Stop server"
	@echo "   make stop-all      - Stop everything"
	@echo ""

stop-all: server-stop ## 🛑 Stop all services
	@echo "$(COLOR_GREEN)✅ All services stopped$(COLOR_RESET)"

# ============================================================================
# 🎨 CODE QUALITY
# ============================================================================

lint: ## 🎨 [QUALITY] Check code style with flake8
	@echo "$(COLOR_CYAN)🔍 Checking code style with flake8...$(COLOR_RESET)"
	@$(VENV_BIN)/flake8 main.py pipelines/ utils/ api/ || true
	@echo "$(COLOR_GREEN)✅ Lint complete!$(COLOR_RESET)"

format: ## 🎨 [QUALITY] Format code with black
	@echo "$(COLOR_CYAN)🎨 Formatting code with black...$(COLOR_RESET)"
	@$(VENV_BIN)/black main.py pipelines/ utils/ api/
	@echo "$(COLOR_GREEN)✅ Code formatted!$(COLOR_RESET)"

security: ## 🎨 [QUALITY] Security scan with bandit
	@echo "$(COLOR_CYAN)🔒 Running security scan with bandit...$(COLOR_RESET)"
	@$(VENV_BIN)/bandit -r pipelines/ utils/ api/ -f txt || true
	@echo "$(COLOR_GREEN)✅ Security scan complete!$(COLOR_RESET)"

test: ## 🎨 [QUALITY] Run all tests
	@echo "$(COLOR_CYAN)🧪 Running tests...$(COLOR_RESET)"
	@$(VENV_BIN)/pytest tests/ -v
	@echo "$(COLOR_GREEN)✅ Tests complete!$(COLOR_RESET)"

# ============================================================================
# 🧹 CLEANUP
# ============================================================================

clean: ## 🧹 [CLEANUP] Remove cache files
	@echo "$(COLOR_CYAN)🧹 Cleaning cache files...$(COLOR_RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf catboost_info/
	@echo "$(COLOR_GREEN)✅ Cleanup complete!$(COLOR_RESET)"

clean-all: clean server-stop ## 🧹 [CLEANUP] Remove everything (venv, models, cache, server)
	@echo "$(COLOR_RED)⚠️  This will delete venv and trained models!$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(COLOR_CYAN)🧹 Removing virtual environment and models...$(COLOR_RESET)"; \
		rm -rf $(VENV); \
		rm -rf models/*.joblib; \
		rm -rf logs/*.log; \
		rm -f $(SERVER_PID_FILE); \
		echo "$(COLOR_GREEN)✅ Deep clean complete!$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)Cancelled$(COLOR_RESET)"; \
	fi

# ============================================================================
# 📊 PROJECT INFORMATION
# ============================================================================

info: ## 📊 Show project information
	@echo "$(COLOR_CYAN)╔════════════════════════════════════════════════════════════════════╗$(COLOR_RESET)"
	@echo "$(COLOR_CYAN)║           🚕 UBER FARE PREDICTION PROJECT INFO 🚕                 ║$(COLOR_RESET)"
	@echo "$(COLOR_CYAN)╚════════════════════════════════════════════════════════════════════╝$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)📁 Project Structure:$(COLOR_RESET)"
	@echo "   • 9 ML Models (CatBoost, LightGBM, XGBoost, etc.)"
	@echo "   • 13 Engineered Features"
	@echo "   • 72.6% Best Accuracy (CatBoost)"
	@echo "   • FastAPI Backend + HTML/CSS/JS Frontend"
	@echo ""
	@echo "$(COLOR_GREEN)🔧 Components:$(COLOR_RESET)"
	@if [ -d "$(VENV)" ]; then echo "   ✅ Virtual Environment"; else echo "   ❌ Virtual Environment (run 'make setup')"; fi
	@if [ -f "models/catboost_model.joblib" ]; then echo "   ✅ Trained Models"; else echo "   ❌ Trained Models (run 'make train')"; fi
	@if [ -f $(SERVER_PID_FILE) ] && ps -p $$(cat $(SERVER_PID_FILE)) > /dev/null 2>&1; then echo "   ✅ Server Running"; else echo "   ❌ Server Stopped"; fi
	@echo ""
	@echo "$(COLOR_CYAN)🌐 URLs:$(COLOR_RESET)"
	@echo "   • Home:      http://127.0.0.1:$(API_PORT)/"
	@echo "   • Predict:   http://127.0.0.1:$(API_PORT)/predict-page"
	@echo "   • Retrain:   http://127.0.0.1:$(API_PORT)/retrain-page"
	@echo "   • API Docs:  http://127.0.0.1:$(API_PORT)/docs"
	@echo ""
	@echo "$(COLOR_YELLOW)👨‍💻 Developer: Yassine Maazaoui$(COLOR_RESET)"
	@echo ""
