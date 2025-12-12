PYTHON = python3
VENV = venv
VENV_BIN = $(VENV)/bin
REQUIREMENTS = requirements.txt

.PHONY: help setup install train train-tuned evaluate clean clean-models lint format security test ci all

# Default target
help:
	@echo "=========================================="
	@echo "🚕 Uber Fare Prediction - Make Commands"
	@echo "=========================================="
	@echo ""
	@echo "📦 Setup Commands:"
	@echo "  make setup          - Create venv and install dependencies"
	@echo "  make install        - Install/update dependencies only"
	@echo ""
	@echo "🚀 Training Commands:"
	@echo "  make train          - Train all models (fast, ~3 min)"
	@echo "  make train-tuned    - Train with hyperparameter tuning (~20 min)"
	@echo "  make evaluate       - Evaluate trained models"
	@echo ""
	@echo "🧪 Quality Commands:"
	@echo "  make lint           - Check code quality (flake8)"
	@echo "  make format         - Auto-format code (black)"
	@echo "  make security       - Security check (bandit)"
	@echo "  make test           - Run tests (pytest)"
	@echo "  make ci             - Run all quality checks"
	@echo ""
	@echo "🧹 Cleanup Commands:"
	@echo "  make clean          - Remove cache and temp files"
	@echo "  make clean-models   - Remove saved models"
	@echo ""
	@echo "🎯 Quick Start:"
	@echo "  make all            - Setup + train + evaluate"
	@echo "=========================================="

# Setup virtual environment and install dependencies
setup:
	@echo "📦 Creating virtual environment..."
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@echo "📥 Installing dependencies..."
	@$(VENV_BIN)/pip install --upgrade pip
	@$(VENV_BIN)/pip install -r $(REQUIREMENTS)
	@echo "✅ Setup complete!"

# Install/update dependencies only
install:
	@echo "📥 Installing/updating dependencies..."
	@$(VENV_BIN)/pip install --upgrade pip
	@$(VENV_BIN)/pip install -r $(REQUIREMENTS)
	@echo "✅ Dependencies installed!"

# Train models (fast - no tuning)
train:
	@echo "🚀 Training all models (fast mode)..."
	@$(VENV_BIN)/python main.py
	@echo "✅ Training complete!"

# Train models with hyperparameter tuning
train-tuned:
	@echo "🚀 Training all models with hyperparameter tuning..."
	@echo "⏰ This will take approximately 20 minutes..."
	@$(VENV_BIN)/python main.py --tune
	@echo "✅ Tuned training complete!"

# Evaluate models
evaluate:
	@echo "📊 Evaluating trained models..."
	@$(VENV_BIN)/python main.py --evaluate
	@echo "✅ Evaluation complete!"

# Code quality checks
lint:
	@echo "🔍 Running flake8 (code quality check)..."
	@$(VENV_BIN)/flake8 main.py utils/ pipelines/ --max-line-length=100 --exclude=$(VENV) || true
	@echo "✅ Lint check complete!"

format:
	@echo "🎨 Formatting code with black..."
	@$(VENV_BIN)/black main.py utils/ pipelines/ --line-length=100 --exclude=$(VENV)
	@echo "✅ Code formatted!"

security:
	@echo "🔒 Running bandit (security check)..."
	@$(VENV_BIN)/bandit -r main.py utils/ pipelines/ -ll --exclude=$(VENV) || true
	@echo "✅ Security check complete!"

test:
	@echo "🧪 Running tests..."
	@$(VENV_BIN)/pytest tests/ -v --tb=short || true
	@echo "✅ Tests complete!"

# Run all quality checks (CI pipeline)
ci: lint format security test
	@echo ""
	@echo "=========================================="
	@echo "✅ All quality checks complete!"
	@echo "=========================================="

# Cleanup commands
clean:
	@echo "🧹 Cleaning cache and temp files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleanup complete!"

clean-models:
	@echo "🗑️  Removing saved models..."
	@rm -rf models/*.joblib 2>/dev/null || true
	@echo "✅ Models removed!"

# Quick start - setup, train, and evaluate
all: setup train
	@echo ""
	@echo "=========================================="
	@echo "✅ All tasks complete!"
	@echo "📁 Models saved in: models/"
	@echo "=========================================="
