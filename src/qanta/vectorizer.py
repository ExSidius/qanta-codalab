from os.path import dirname, abspath, join, exists

EMBEDDING_PATH = join(dirname(dirname(dirname(abspath(__file__)))),
					  'data', 'debiased_embeddings.txt')
EMBEDDING_LENGTH = 300


def generate_embeddings():
	assert exists(EMBEDDING_PATH)

	embeddings = {}
	with open('debiased_embeddings.txt', 'r') as file:
		while True:
			line = file.readline()

			if not line:
				break

			line = line.split()
			word = line[0]
			embedding = [float(emb.strip()) for emb in line[1:]]
			if len(embedding) == EMBEDDING_LENGTH:
				embeddings[word] = embedding
	return embeddings


class Vectorizer:
	"""
	This could cause memory problems down the line.
	We should consider alternatives.
	"""
	def __init__(self):
		self.embeddings = generate_embeddings()

	def transform(self, data):
		return [self.vectorize_document(doc) for doc in data]

	def vectorize(self, text):
		length = len(text.split())
		if length > 1:
			return self.vectorize_document(text)
		elif length == 1:
			return self.vectorize_word(text)
		else:
			return []

	def vectorize_document(self, document):
		return [self.vectorize(word) for word in document.split()]

	def vectorize_word(self, word):
		if word in self.embeddings:
			return self.embeddings[word]

		return [0 for _ in range(EMBEDDING_LENGTH)]