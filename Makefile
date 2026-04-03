.PHONY: help install train test clean format lint

# Default target
help:
	@echo "TriSG-KT Makefile Commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Run training with default settings"
	@echo "  make test        - Run unit tests (pytest)"
	@echo "  make smoke-test  - Quick training smoke test"
	@echo "  make clean       - Clean output files"
	@echo "  make format      - Format Python code"
	@echo "  make lint        - Run linting"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

# Quick training
train:
	@echo "Starting training..."
	./scripts/train.sh full

# Full training with all innovations
train-full:
	@echo "Starting full TriSG-KT training..."
	./scripts/2_train.sh full

# Unit tests
test:
	@echo "Running unit tests..."
	python -m pytest DTransformer/tests/ -v

# Quick smoke test (training script in test mode)
smoke-test:
	@echo "Running smoke test..."
	./scripts/2_train.sh test

# Clean outputs
clean:
	@echo "Cleaning output files..."
	rm -rf output/* logs/* checkpoints/* results/*
	@echo "✓ Clean complete"

# Format code
format:
	@echo "Formatting Python code..."
	black *.py **/*.py
	@echo "✓ Format complete"

# Run linting
lint:
	@echo "Running linting..."
	pylint DTransformer/ baselines/ scripts/
	@echo "✓ Lint complete"
