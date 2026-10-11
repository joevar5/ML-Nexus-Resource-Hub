.PHONY: setup data test
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
data:
	.venv/bin/python -m src.data
test:
	.venv/bin/python -m pytest -q
