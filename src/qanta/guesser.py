import torch
import pickle
import nltk
from guesser_model import Model

MODEL_PATH = 'dan_debias.pt'
IND_LABEL_PATH = 'word_maps.pkl'
kUNK = '<unk>'
kPAD = '<pad>'
# arbitrary, temp buzzer
BUZZ_THRESHOLD = 0.01

# how to load model & data so it's initialized once?
model = torch.load(MODEL_PATH)
with open(IND_LABEL_PATH, 'rb') as f:
	ind_and_labels = pickle.load(f)
word2index = ind_and_labels['word2index']
index2class = ind_and_labels['index2class']


def vectorize(tokenized_text):
	return [word2index[word] if word in word2index else word2index[kUNK] for word in tokenized_text]


class Guesser:
	def guess_and_buzz(self, question):
		tokens = nltk.word_tokenize(question)
		vectorized = vectorize(tokens)
		logits = model(torch.FloatTensor([vectorized]))
		top_n, top_i = logits.topk(5)
		buzz = False
		if top_n[0] > BUZZ_THRESHOLD:
			buzz = True
		return index2class[top_i[0]], buzz

	def batch_guess_and_buzz(self, questions):
		return [self.guess_and_buzz(question) for question in questions]

	def train(self, save=True):
		pass


guesser = Guesser()
