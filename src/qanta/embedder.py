from os.path import dirname, abspath, join, exists
import torch

EMBEDDING_PATH = join(dirname(dirname(dirname(abspath(__file__)))),
					  'data', 'debiased_embeddings.txt')
EMBEDDING_LENGTH = 300
TORCH_EMBEDDER_PATH = join(dirname(dirname(dirname(abspath(__file__)))),
					  'data', 'torch_embedder.pt')


def generate_embeddings():
	assert exists(EMBEDDING_PATH)

	print('Starting to load pre-trained embeddings')
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
	print('Done loading pre-trained embeddings')
	return embeddings


class Embedder:
	"""
	This could cause memory problems down the line.
	We should consider alternatives.
	"""
	def __init__(self, index2word):
		self.embeddings = generate_embeddings()
		self.index2word = index2word

	def get_embedding(self):
		if exists(TORCH_EMBEDDER_PATH):
			embedding = torch.load(TORCH_EMBEDDER_PATH)
		else:
			embedder = torch.zeros(len(self.index2word), EMBEDDING_LENGTH)
			for ind, word in self.index2word.items():
				if word in self.embeddings:
					embedder[ind] = torch.FloatTensor(self.embeddings[word])
			embedding = torch.nn.Embedding.from_pretrained(embedder)
			torch.save(embedding, TORCH_EMBEDDER_PATH)
		return embedding
	#
	# def __call__(self, text: torch.Tensor):
	# 	new_text = []
	# 	for row in text:
	# 		new_text.append([self.vectorize_word(self.index2word[index]) for index in row])
	# 	return torch.Tensor(new_text)
	#
	# def vectorize_word(self, word):
	# 	if word in self.embeddings:
	# 		return self.embeddings[word]
	#
	# 	return [0 for _ in range(EMBEDDING_LENGTH)]


if __name__ == '__main__':
	pass