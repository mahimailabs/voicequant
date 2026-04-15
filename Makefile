.PHONY: install dev test lint serve bench clean

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

serve:
	voicequant serve

bench:
	voicequant bench --all

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
