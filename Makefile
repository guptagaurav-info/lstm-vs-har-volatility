PY ?= python
export PYTHONPATH := src

.PHONY: all ingest run test
all: ingest run
ingest:
	$(PY) -m volnet.ingest
run:
	$(PY) -m volnet.pipeline
test:
	$(PY) -m pytest -q tests
