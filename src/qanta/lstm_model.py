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
TRAIN_DATASET = 'train_dataset.pkl'
DEV_DATASET = 'dev_dataset.pkl'
TEST_DATASET = 'test_dataset.pkl'
WORD_MAPS = 'word_maps.pkl'

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
			label = question['category']
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

	# creating labels for every time step
	label_list = [[labels[i]]*len(question) for i, question in enumerate(questions)]
	# padding to be the same length
	target_labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(y) for y in label_list], padding_value=-1).t()

	x1 = torch.LongTensor(len(questions), max(question_lens)).zero_()
	for i, (question, q_len) in enumerate(zip(questions, question_lens)):
		x1[i, :q_len].copy_(torch.LongTensor(question))

	return {
		'text': x1,
		'len': torch.FloatTensor(question_lens),
		'labels': target_labels,
	}


def accuracy_fn(logits, labels, num_classes):
	# reshape labels to give a flat vector of length batch_size*seq_len
	labels = labels.contiguous()
	labels = labels.view(-1)

	# flatten all predictions
	logits = logits.contiguous()
	logits = logits.view(-1, num_classes)

	# create mask - remember, we padded using -1 for our labels in batchify
	mask = (labels > -1).float()

	# these are the actual number of examples ignoring padded stuff
	num_examples = int(torch.sum(mask).data)

	# get the non-zero indices of the mask - these are the corresponding indices in logits/labels
	# that contain data of value (rest is just padded)
	indices = torch.nonzero(mask.data).squeeze()

	# get the logits corresponding to non-padded values as given by non-zero indices of mask
	logits = torch.index_select(logits, 0, indices)

	# get the logits corresponding to non-padded values as given by non-zero indices of mask
	labels = torch.index_select(labels, 0, indices)

	top_n, top_i = logits.topk(1)
	error = torch.nonzero(top_i.squeeze() - labels).size(0)

	return error, num_examples


def loss_fn(outputs, labels, num_classes):
	# to compute cross entropy loss
	outputs = torch.nn.functional.log_softmax(outputs, dim=1)

	# reshape labels to give a flat vector of length batch_size*seq_len
	labels = labels.contiguous()
	labels = labels.view(-1)

	# flatten all predictions
	outputs = outputs.contiguous()
	outputs = outputs.view(-1, num_classes)

	# mask out 'PAD' tokens
	mask = (labels > -1).float()

	# the number of tokens is the sum of elements in mask
	num_tokens = int(torch.sum(mask).data)

	# pick the values corresponding to labels and multiply by mask
	outputs = outputs[range(outputs.shape[0]), labels] * mask

	# cross entropy loss for all non 'PAD' tokens
	return -torch.sum(outputs) / num_tokens



class LSTMModel(nn.Module):
	def __init__(self,
				 n_classes,
				 vocab_size,
				 embedding_dimension=EMBEDDING_LENGTH,
	             embedder=None,
				 n_hidden=50):
		super(LSTMModel, self).__init__()

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

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = torch.nn.LSTM(self.embedding_dimension, n_hidden)

		# The linear layer that maps from hidden state space to class space
		self.hidden2class = nn.Linear(n_hidden, n_classes)

	def forward(self, input_text: torch.Tensor, text_len: torch.Tensor, is_prob=False):
		embedding = self.embeddings(input_text)
		lstm_out, _ = self.lstm(embedding)
		logits = self.hidden2class(lstm_out)

		return logits


def evaluate(data_loader, model, device):
	"""
	evaluate the current model, get the accuracy for dev/test set
	Keyword arguments:
	data_loader: pytorch build-in data loader output
	model: model to be evaluated
	device: cpu or gpu
	"""

	model.eval()
	total_num_examples = 0
	total_error = 0
	for idx, batch in enumerate(data_loader):
		question_text = batch['text'].to(device)
		question_len = batch['len'].to(device)
		labels = batch['labels'].to(device)

		####Your code here ---

		# get the output from the model
		logits = model(question_text, question_len)

		# get error, num_examples using accuracy_fn defined previously
		error, num_examples = accuracy_fn(logits, labels, model.n_classes)

		# update total_error and total_num_examples
		total_error += error
		total_num_examples += num_examples

	accuracy = 1 - total_error / total_num_examples
	return accuracy



def train(args: argparse.Namespace,
		  model,
		  train_data_loader: torch.utils.data.DataLoader,
		  dev_data_loader: torch.utils.data.DataLoader,
		  accuracy: float,
		  device: torch.device,
          learning_rate: float) -> float:
	"""
		Train the current model
		Keyword arguments:
		args: arguments, here we use checkpoint value
		model: model to be trained
		train_data_loader: pytorch build-in data loader output for training examples
		dev_data_loader: pytorch build-in data loader output for dev examples
		device: cpu or gpu
	"""
	if args.optim == 'adamax':
		optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
	elif args.optim == 'rprop':
		optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)

	print_loss_total = 0
	epoch_loss_total = 0

	best_train_acc, best_dev_acc = 0.0, 0.0

	for i, batch in enumerate(train_data_loader):
		model.train()
		question_text = batch['text'].to(device)
		question_len = batch['len'].to(device)
		labels = batch['labels'].to(device)

		# zero out
		optimizer.zero_grad()

		# get output from model
		logits = model(question_text, question_len)

		# use loss_fn defined above to calculate loss
		loss = loss_fn(logits, labels, model.n_classes)

		# backprop
		loss.backward()
		optimizer.step()

		clip_grad_norm_(model.parameters(), 5)
		print_loss_total += loss.cpu().data.numpy()
		epoch_loss_total += loss.cpu().data.numpy()

		if i % args.checkpoint == 0 and i > 0:
			# use accuracy_fn defined above to calculate 'error' and number of examples ('num_examples') used to
			# calculate accuracy below.
			error, num_examples = accuracy_fn(logits, labels, model.n_classes)
			accuracy = 1 - error / num_examples

			print_loss_avg = print_loss_total / args.checkpoint
			dev_acc = evaluate(dev_data_loader, model, device)
			print('number of steps: %d, train loss: %.5f, train acc: %.3f, dev acc: %.3f' % (i + 1, print_loss_avg,
			                                                                                 accuracy, dev_acc))
			print_loss_total = 0
			if accuracy > best_train_acc:
				best_train_acc = accuracy
			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				torch.save(model, args.save_model)

	return best_dev_acc


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
	parser.add_argument('--save-model', type=str, default='lstm.pt')
	parser.add_argument('--load-model', type=str, default='lstm.pt')
	parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
	parser.add_argument('--checkpoint', type=int, default=50)
	parser.add_argument('--use-pretrained-embeddings', action='store_true', default=False)
	parser.add_argument('--store-word-maps', action='store_true', default=False)
	parser.add_argument('--optim', type=str, default='adamax')
	parser.add_argument('--save-qdataset', action='store_true', default=False)
	parser.add_argument('--load-qdataset', action='store_true', default=False)
	parser.add_argument('--learning-rate', type=float, default=0.001)

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	print(args.cuda)
	device = torch.device("cuda" if args.cuda else "cpu")
	print(device)

	assert not (args.save_qdataset and args.load_qdataset)

	if args.load_qdataset:
		print('Loading saved datasets')
		train_dataset = pickle.load(open(TRAIN_DATASET, 'rb'))
		dev_dataset = pickle.load(open(DEV_DATASET, 'rb'))
		test_dataset = pickle.load(open(TEST_DATASET, 'rb'))
		word_maps = pickle.load(open(WORD_MAPS, 'rb'))

		voc = word_maps['voc']
		word2index = word_maps['word2index']
		index2word = word_maps['index2word']
		class2index = word_maps['class2index']
		index2class = word_maps['index2class']
	else:
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
		if not args.load_qdataset:
			test_dataset = QuestionDataset(test_examples, word2index, num_classes,
									   class2index)
			if args.save_qdataset:
				print('Saving test dataset')
				pickle.dump(test_dataset, open(TEST_DATASET, 'wb'))
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
			model.to(device)
		else:
			model = LSTMModel(num_classes, len(voc), embedder=embedder)
			model.to(device)

		print(model)

		if not args.load_qdataset:
			train_dataset = QuestionDataset(training_examples, word2index,
										num_classes, class2index)
			dev_dataset = QuestionDataset(dev_examples, word2index, num_classes,
			                              class2index)
			if args.save_qdataset:
				print('Saving train & dev datasets')
				pickle.dump(train_dataset, open(TRAIN_DATASET, 'wb'))
				pickle.dump(dev_dataset, open(DEV_DATASET, 'wb'))

		train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
		dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
		dev_loader = torch.utils.data.DataLoader(dev_dataset,
												 batch_size=args.batch_size,
												 sampler=dev_sampler,
												 num_workers=0,
												 collate_fn=batchify)
		accuracy = 0
		for epoch in range(args.num_epochs):
			print(f'Start Epoch {epoch}')
			learning_rate = args.learning_rate * ((0.5) ** (epoch//100))
			train_loader = torch.utils.data.DataLoader(train_dataset,
													   batch_size=args.batch_size,
													   sampler=train_sampler,
													   num_workers=0,
													   collate_fn=batchify)
			accuracy = train(args, model, train_loader, dev_loader, accuracy,
							 device, learning_rate)
		print('Start Testing:\n')

		if not args.load_qdataset:
			test_dataset = QuestionDataset(test_examples, word2index, num_classes,
									   class2index)
		test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
		test_loader = torch.utils.data.DataLoader(test_dataset,
												  batch_size=args.batch_size,
												  sampler=test_sampler,
												  num_workers=0,
												  collate_fn=batchify)
		evaluate(test_loader, model, device)
