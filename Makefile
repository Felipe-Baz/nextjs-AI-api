
install:
	pipenv install --dev

clean:
	pipenv clean

shell:
	pipenv shell

dev:
	pipenv run uvicorn app.main:app --reload

check:
	black --check .

format:
	black .

tests:
	pipenv run pytest

cov:
	pipenv run pytest --cov-report=html --cov .

docker_image:
	docker build -t felipebaz/sentiment-analisys .

docker_run:
	docker run -d --name sentiment-analisys -p 5000:5000 felipebaz/sentiment-analisys

docker_push:
	docker push felipebaz/sentiment-analisys

requirements:
	pipenv requirements > requirements.txt

kube_start:
	minikube start

kube_deploy:
	kubectl apply -f kubernetes.yaml

kube_access:
	minikube service sentiment-analisys-service

kube_dash:
	minikube dashboard

.PHONY: install clean shell dev check format tests cov docker_image docker_run docker_push requirements kube_start kube_deploy kube_access kube_dash
