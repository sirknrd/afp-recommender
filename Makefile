.PHONY: install train test docker lint clean

install:
	pip install -r requirements.txt

train:
	python train_fixed.py --epochs 50 --batch-size 64 --lr 1e-3

test:
	pytest tests/ -v --cov=quantum_lstm --cov-report=html

lint:
	black .
	isort .

docker:
	docker build -t quantum-lstm-fixed -f docker/Dockerfile .
	docker run --gpus all -p 8888:8888 quantum-lstm-fixed

clean:
	rm -rf runs/ __pycache__/ .pytest_cache/ htmlcov/
