.PHONY: deps server

deps:
	pip install -r requirements.txt

server:
	FLASK_APP=main.py flask run
