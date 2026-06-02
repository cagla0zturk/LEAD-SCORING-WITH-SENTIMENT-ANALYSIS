.PHONY: install data train test serve docker-build docker-run clean

# Use python3 by default (many systems do not ship a bare `python`). Override with:
#   make train PYTHON=python
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) -m scripts.prepare_data

train: data
	$(PYTHON) -m scripts.train_all

train-quick: data
	$(PYTHON) -m scripts.train_all --quick --no-plots

test:
	$(PYTHON) -m pytest -q

serve:
	uvicorn lead_priority.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t lead-priority:latest .

docker-run:
	docker run --rm -p 8000:8000 lead-priority:latest

clean:
	rm -rf models/*.joblib models/*.json data/processed/*.csv data/processed/*.json reports/*.png
