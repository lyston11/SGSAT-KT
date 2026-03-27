.PHONY: help install train test clean format lint

# Default target
help:
	@echo "SGSAT-KT Makefile Commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Run training with default settings"
	@echo "  make test        - Run tests"
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
	./train.sh full

# Full training with all innovations
train-full:
	@echo "Starting full SGSAT-KT training..."
	./scripts/2_train.sh full

# Test with small dataset
test:
	@echo "Running tests..."
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
