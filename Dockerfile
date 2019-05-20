FROM docker.io/entilzha/quizbowl:0.1

RUN python -m pip install --upgrade torch && \
	python -c "import nltk; nltk.download('punkt')"