from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch

from typing import NamedTuple, List, Dict, Tuple, Optional, Union
import argparse
import json
import nltk
import pickle

from helpers import logger
from embedder import EMBEDDING_LENGTH, Embedder

kUNK = '<unk>'
kPAD = '<pad>'


class Example(NamedTuple):
	tokenized_text: List[str]
	label: str


@logger('loading data')
def load_data(filename: str, limit: Optional[int] = None) -> List[Example]:
	data = []
	with open(filename) as json_data:
		questions = json.load(json_data)["questions"][:limit]
		for question in questions:
			tokenized_text = nltk.word_tokenize(question['text'])
			label = question['page']
			if label:
				data.append(Example(tokenized_text, label))
	return data


@logger('creating class labels')
def class_labels(examples: List[Example]) -> Tuple[
	Dict[str, int], Dict[int, str]]:
	classes = set([example.label for example in examples])
	index2class = dict(enumerate(classes))
	class2index = {v: k for k, v in index2class.items()}
	return class2index, index2class


@logger('loading words')
def load_words(examples: List[Example]) -> Tuple[
	List[str], Dict[str, int], Dict[int, str]]:
	words = {kPAD, kUNK}

	tokenized_texts, _ = zip(*examples)
	for tokenized_text in tokenized_texts:
		for token in tokenized_text:
			if token not in words:
				words.add(token)
	words = list(words)

	index2word = dict(enumerate(words))
	word2index = {v: k for k, v in index2word.items()}

	return words, word2index, index2word


class QuestionDataset(Dataset):
	def __init__(self,
				 examples: List[Example],
				 word2index: Dict[str, int],
				 num_classes: int,
				 class2index: Optional[Dict[str, int]] = None):
		self.tokenized_questions = []
		self.labels = []

		tokenized_questions, labels = zip(*examples)
		self.tokenized_questions = list(tokenized_questions)
		self.labels = list(labels)

		new_labels = []
		for label in self.labels:
			new_label = class2index[
				label] if label in class2index else num_classes
			new_labels.append(new_label)
		self.labels = new_labels

		self.word2index = word2index

	def __getitem__(self, index) -> Tuple[List[int], int]:
		return self.vectorize(self.tokenized_questions[index]), self.labels[
			index]

	def __len__(self):
		return len(self.tokenized_questions)

	def vectorize(self, tokenized_text: List[str]) -> List[int]:
		return [self.word2index[word] if word in self.word2index else
				self.word2index[kUNK]
				for word in tokenized_text]


def batchify(batch: List[Tuple[List[int], int]]) -> Dict[
	str, Union[torch.LongTensor, torch.FloatTensor]]:
	"""
	Create a batch of examples which includes the
	question text, question length and labels.
	"""

	questions, labels = zip(*batch)
	questions = list(questions)
	question_lens = [len(q) for q in questions]
	labels = list(labels)

	labels = torch.LongTensor(labels)
	x1 = torch.LongTensor(len(questions), max(question_lens)).zero_()
	for i, (question, q_len) in enumerate(zip(questions, question_lens)):
		x1[i, :q_len].copy_(torch.LongTensor(question))

	return {
		'text': x1,
		'len': torch.FloatTensor(question_lens),
		'labels': labels,
	}


class Model(nn.Module):
	def __init__(self,
				 n_classes,
				 vocab_size,
				 embedding_dimension=EMBEDDING_LENGTH,
	             embedder=None,
				 n_hidden=50,
				 dropout_rate=.5):
		super(Model, self).__init__()

		self.n_classes = n_classes
		self.vocab_size = vocab_size

		self.n_hidden = n_hidden

		self.embedding_dimension = embedding_dimension
		if embedder:
			self.embeddings = embedder
		else:
			self.embeddings = nn.Embedding(self.vocab_size,
										   self.embedding_dimension,
										   padding_idx=0)

		self.dropout_rate = dropout_rate

		self.layer1 = nn.Linear(embedding_dimension, n_hidden)
		self.layer2 = nn.Linear(n_hidden, n_classes)

		self.classifier = nn.Sequential(
			self.layer1,
			nn.ReLU(),
			nn.Dropout(p=dropout_rate, inplace=False),
			self.layer2,
		)
		self._softmax = nn.Softmax(dim=1)

	def forward(self, input_text: torch.Tensor, text_len: torch.Tensor, is_prob=False):

		logits = torch.LongTensor([0.0] * self.n_classes)

		embedding = self.embeddings(input_text)
		average_embedding = embedding.sum(1) / text_len.view(embedding.size(0),
															 -1)

		if is_prob:
			logits = self._softmax(logits)
		else:
			logits = self.classifier(average_embedding)

		return logits


def evaluate(data_loader: torch.utils.data.DataLoader,
			 model: Model,
			 device: torch.device) -> float:
	model.eval()
	num_examples = 0
	error = 0
	for i, batch in enumerate(data_loader):
		question_text = batch['text'].to(device)
		question_len = batch['len'].to(device)
		labels = batch['labels'].to(device)

		logits = model(question_text, question_len)

		top_n, top_i = logits.topk(1)
		num_examples += question_text.size(0)
		error += torch.nonzero(top_i.squeeze() - labels).size(0)

	accuracy = 1 - error / num_examples
	print(accuracy)
	return accuracy


def train(args: argparse.Namespace,
		  model: Model,
		  train_data_loader: torch.utils.data.DataLoader,
		  dev_data_loader: torch.utils.data.DataLoader,
		  accuracy: float,
		  device: torch.device) -> float:
	model.train()
	optimizer = torch.optim.Adamax(model.parameters())
	criterion = nn.CrossEntropyLoss()
	# print_loss_total = 0
	# epoch_loss_total = 0
	# start = time.time()

	for i, batch in enumerate(train_data_loader):
		question_text = batch['text'].to(device)
		question_len = batch['len'].to(device)
		labels = batch['labels'].to(device)

		optimizer.zero_grad()
		result = model(question_text, question_len)
		loss = criterion(result, labels)
		loss.backward()
		optimizer.step()

		clip_grad_norm_(model.parameters(), args.grad_clipping)
		# print_loss_total += loss.data.numpy()
		# epoch_loss_total += loss.data.numpy()

		if i % args.checkpoint == 0 and i > 0:
			# print_loss_avg = print_loss_total / args.checkpoint

			# print(
			# 	f'number of steps: {i}, loss: {print_loss_avg} time: {time.time() - start}')
			# print_loss_total = 0
			curr_accuracy = evaluate(dev_data_loader, model, device)
			if curr_accuracy > accuracy:
				accuracy = curr_accuracy
				torch.save(model, args.save_model)

	return accuracy


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Question Type')
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--train-file', type=str, default='../../data/qanta.train.json')
	parser.add_argument('--dev-file', type=str, default='../../data/qanta.dev.json')
	parser.add_argument('--test-file', type=str, default='../../data/qanta.test.json')
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--num-epochs', type=int, default=20)
	parser.add_argument('--grad-clipping', type=int, default=5)
	parser.add_argument('--resume', action='store_true', default=False)
	parser.add_argument('--test', action='store_true', default=False)
	parser.add_argument('--save-model', type=str, default='dan.pt')
	parser.add_argument('--load-model', type=str, default='dan.pt')
	parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
	parser.add_argument('--checkpoint', type=int, default=50)
	parser.add_argument('--use-pretrained-embeddings', action='store_true', default=False)
	parser.add_argument('--store-word-maps', action='store_true', default=False)

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if args.cuda else "cpu")

	training_examples = load_data(args.train_file, args.limit)
	dev_examples = load_data(args.dev_file)
	test_examples = load_data(args.test_file)

	voc, word2index, index2word = load_words(training_examples + dev_examples)

	class2index, index2class = class_labels(training_examples + dev_examples)
	num_classes = len(class2index)

	if args.store_word_maps:
		with open('word_maps.pkl', 'wb') as f:
			pickle.dump({'voc': voc, 'word2index': word2index, 'index2word': index2word,
			             'class2index': class2index, 'index2class': index2class}, f)

	embedder = None
	if args.use_pretrained_embeddings:
		embedder = Embedder(index2word).get_embedding()

	print(f'Number of classes in dataset: {num_classes}')
	print()

	if args.test:
		model = torch.load(args.load_model)
		test_dataset = QuestionDataset(test_examples, word2index, num_classes,
									   class2index)
		test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
		test_loader = torch.utils.data.DataLoader(test_dataset,
												  batch_size=args.batch_size,
												  sampler=test_sampler,
												  num_workers=0,
												  collate_fn=batchify)
		evaluate(test_loader, model, device)
	else:
		if args.resume:
			model = torch.load(args.load_model)
		else:
			model = Model(num_classes, len(voc), embedder=embedder)
			model.to(device)

		print(model)

		train_dataset = QuestionDataset(training_examples, word2index,
										num_classes, class2index)
		train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

		dev_dataset = QuestionDataset(dev_examples, word2index, num_classes,
									  class2index)
		dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
		dev_loader = torch.utils.data.DataLoader(dev_dataset,
												 batch_size=args.batch_size,
												 sampler=dev_sampler,
												 num_workers=0,
												 collate_fn=batchify)
		accuracy = 0
		for epoch in range(args.num_epochs):
			print(f'Start Epoch {epoch}')
			train_loader = torch.utils.data.DataLoader(train_dataset,
													   batch_size=args.batch_size,
													   sampler=train_sampler,
													   num_workers=0,
													   collate_fn=batchify)
			accuracy = train(args, model, train_loader, dev_loader, accuracy,
							 device)
		print('Start Testing:\n')

		test_dataset = QuestionDataset(test_examples, word2index, num_classes,
									   class2index)
		test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
		test_loader = torch.utils.data.DataLoader(test_dataset,
												  batch_size=args.batch_size,
												  sampler=test_sampler,
												  num_workers=0,
												  collate_fn=batchify)
		evaluate(test_loader, model, device)