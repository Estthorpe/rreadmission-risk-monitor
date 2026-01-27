.PHONY: install lint test ingest
install:
	pip install -e .".[dev]"

lint:
	pytest -q

ingest:
	python ingest_data.py