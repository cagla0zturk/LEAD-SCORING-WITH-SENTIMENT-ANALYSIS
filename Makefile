.PHONY: install data train test serve docker-build docker-run clean

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python -m scripts.prepare_data

train: data
	python -m scripts.train_all

train-quick: data
	python -m scripts.train_all --quick --no-plots

test:
	python -m pytest -q

serve:
	uvicorn lead_priority.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t lead-priority:latest .

docker-run:
	docker run --rm -p 8000:8000 lead-priority:latest

clean:
	rm -rf models/*.joblib models/*.json data/processed/*.csv data/processed/*.json reports/*.png
