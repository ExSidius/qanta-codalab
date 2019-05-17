import click
from flask import Flask, jsonify, request
from .guesser_model import Model

from .guesser import Guesser
from .util import download


def create_app(enable_batch=True):
    app = Flask(__name__)

    guesser = Guesser()

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guesser.guess_and_buzz(question)
        return jsonify({
            'guess': guess,
            'buzz': buzz,
        })

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]

        return jsonify([
            {'guess': guess, 'buzz': buzz}
            for guess, buzz in guesser.batch_guess_and_buzz(questions)
        ])

    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    guesser.train(save=True)


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()
