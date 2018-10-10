from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset


MODEL_PATH = 'models/tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(model, question_text):
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


def create_app():
    tfidf_guesser = TfidfGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.form['question_text']
        guess, buzz = guess_and_buzz(tfidf_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
def web(host, port):
    """
    Start web server wrapping tfidf model
    """
    app = create_app()
    app.run(host=host, port=port, debug=False)


@cli.command()
@click.argument('input_file')
@click.argument('output_file')
def batch(input_file, output_file):
    """
    Run batch mode where input files conform to:

    input_file: each line is a question in json format
        {"char_position":int, "question_text": str, "Incremental_text":str,"is_new_sent":false/true}
    output_file: each line is a question in json format
        {"guess":str, "buzz": false/true}
    """
    tfidf_guesser = TfidfGuesser.load()

    def _guess(question_text):
        guesses = tfidf_guesser.guess([question_text], BUZZ_NUM_GUESSES)[0]
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        return guesses[0][0], buzz

    with open(path.join('data', output_file), 'w') as outh:
        with open(path.join('data', input_file)) as inh:
            for question_json in tqdm(inh):
                guess, buzz = _guess(json.loads(question_json)['question_text'])
                outh.write(json.dumps({'guess': guess, 'buzz': True if buzz else False}) + '\n')


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(dataset.training_data())
    tfidf_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
