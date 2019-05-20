import torch
import pickle
import sys
import nltk
from os.path import dirname, abspath, join, exists

from qanta.guesser_model import Model

MODEL_PATH = join(dirname(abspath(__file__)), 'dan_bias.pt')
IND_LABEL_PATH = join(dirname(abspath(__file__)), 'word_maps.pkl')
kUNK = '<unk>'
kPAD = '<pad>'
# arbitrary, temp buzzer
# BUZZ_THRESHOLD = 0.01
BUZZ_THRESHOLD = -200

assert exists(MODEL_PATH)
assert exists(IND_LABEL_PATH)

# how to load model & data so it's initialized once?



def vectorize(word2index, tokenized_text):
	return [word2index[word] if word in word2index else word2index[kUNK] for word in tokenized_text]


class Guesser:
	def __init__(self):
		self.model = torch.load(MODEL_PATH, map_location='cpu')
		with open(IND_LABEL_PATH, 'rb') as f:
			self.ind_and_labels = pickle.load(f)
		self.word2index = self.ind_and_labels['word2index']
		self.index2class = self.ind_and_labels['index2class']

	def guess_and_buzz(self, question):
		tokens = nltk.word_tokenize(question)
		vectorized = vectorize(self.word2index, tokens)

		input_text = torch.Tensor([vectorized])
		text_length = torch.Tensor([len(vectorized)])
		logits = self.model(input_text, text_length)

		top_n, top_i = logits.topk(5)
		# print(top_n, top_i, file=sys.stderr)
		# print('\n', file=sys.stderr)
		buzz = False
		if top_n[0][0].item() > BUZZ_THRESHOLD:
			buzz = True
		return self.index2class[top_i[0][0].item()], buzz

	def batch_guess_and_buzz(self, questions):
		return [self.guess_and_buzz(question) for question in questions]

	def train(self, save=True):
		pass
