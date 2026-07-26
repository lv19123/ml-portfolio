install:
	pip install -r requirements.txt

test:
	pytest

run-backend:
	uvicorn backend.app.main:app --reload

run-frontend:
	streamlit run frontend/app.py

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down
