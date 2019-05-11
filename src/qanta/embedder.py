from os.path import dirname, abspath, join, exists
import json
import torch

EMBEDDING_PATH = join(dirname(dirname(dirname(abspath(__file__)))),
					  'data', 'debiased_embeddings.txt')
EMBEDDING_JSON_PATH = join(dirname(dirname(dirname(abspath(__file__)))),
					  'data', 'debiased_embeddings.json')
EMBEDDING_LENGTH = 300


def generate_embeddings():
	assert exists(EMBEDDING_PATH)

	embeddings = {}
	with open(EMBEDDING_PATH, 'r') as file:
		while True:
			line = file.readline()

			if not line:
				break

			line = line.split()
			word = line[0]
			embedding = [float(emb.strip()) for emb in line[1:]]
			if len(embedding) == EMBEDDING_LENGTH:
				embeddings[word] = embedding

	if not exists(EMBEDDING_JSON_PATH):
		with open(EMBEDDING_JSON_PATH, 'w') as file:
			json.dump(embeddings, file)

	return embeddings


class Embedder:
	"""
	This could cause memory problems down the line.
	We should consider alternatives.
	"""
	def __init__(self, index2word):
		self.embeddings = generate_embeddings()
		self.index2word = index2word

	def __call__(self, text: torch.Tensor):
		new_text = []
		for row in text:
			new_text.append([self.vectorize_word(self.index2word[index]) for index in row])
		return torch.Tensor(new_text)

	def vectorize_word(self, word):
		if word in self.embeddings:
			return self.embeddings[word]

		return [0 for _ in range(EMBEDDING_LENGTH)]