from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch


from typing import NamedTuple, List, Dict, Tuple, Optional
import argparse
import json
import time
import nltk

from helpers import logger

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
            label = question['category']
            if label:
                data.append(Example(tokenized_text, label))
    return data


@logger('creating class labels')
def class_labels(examples: List[Example]) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes = [example.label for example in examples]
    index2class = dict(enumerate(classes))
    class2index = {v: k for k, v in index2class.items()}
    return class2index, index2class


@logger('loading words')
def load_words(examples: List[Example]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    words = {kPAD, kUNK}
    for tokenized_text, _ in examples:
        words = words.union(set(tokenized_text))
    words = sorted(words)

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
            new_label = class2index[label] if label in class2index else num_classes
            new_labels.append(new_label)
        self.labels = new_labels

        self.word2index = word2index

    def __getitem__(self, index) -> Tuple[List[int], int]:
        return self.vectorize(self.tokenized_questions[index]), self.labels[index]

    def __len__(self):
        return len(self.tokenized_questions)

    def vectorize(self, tokenized_text: List[str]) -> List[int]:
        return [self.word2index[word] if word in self.word2index else self.word2index[kUNK]
                for word in tokenized_text]


def batchify(batch):
    """
    Gather a batch of individual examples into one batch,
    which includes the question text, question length and labels
    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        ####Your code here
        logits = model(question_text, question_len)

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy


def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model
    Keyword arguments:
    args: arguments
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### modify the following code to complete the training funtion

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        #### Your code here
        optimizer.zero_grad()
        result = model(question_text, question_len)
        loss = criterion(result, labels)
        loss.backward()
        optimizer.step()

        clip_grad_norm_(model.parameters(), args.grad_clipping)
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()

        if idx % args.checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / args.checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time() - start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, args.save_model)
                accuracy = curr_accuracy
    return accuracy


class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model for passing tests, might need to tinker to improve performance/handle
    #### pretrained word embeddings/for your project code.

    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        # Create the actual prediction framework for the DAN classifier.

        # You'll need combine the two linear layers together, probably
        # with the Sequential function.  The first linear layer takes
        # word embeddings into the representation space, and the
        # second linear layer makes the final prediction.  Other
        # layers / functions to consider are Dropout, ReLU.
        # For test cases, the network we consider is - linear1 -> ReLU() -> Dropout(0.5) -> linear2

        #### Your code here
        self.classifier = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            nn.Dropout(p=nn_dropout, inplace=False),
            self.linear2,
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.

        Keyword arguments:
        input_text : vectorized question text
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """

        logits = torch.LongTensor([0.0] * self.n_classes)

        # Complete the forward funtion.  First look up the word embeddings.
        embedding = self.embeddings(input_text)

        # Then average them
        average_embedding = embedding.sum(1) / text_len.view(embedding.size(0), -1)

        # Before feeding them through the network

        logits = self.classifier(average_embedding)
        if is_prob:
            logits = self._softmax(logits)

        return logits


# You basically do not need to modify the below code
# But you may need to add funtions to support error analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--train-file', type=str, default='../../data/qanta.train.json')
    parser.add_argument('--dev-file', type=str, default='../../data/qanta.dev.json')
    parser.add_argument('--test-file', type=str, default='../../data/qanta.test.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument('--load-model', type=str, default='q_type.pt')
    parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
    parser.add_argument('--checkpoint', type=int, default=50)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    training_examples = load_data(args.train_file, args.limit)
    dev_examples = load_data(args.dev_file)
    test_examples = load_data(args.test_file)

    voc, word2index, index2word = load_words(training_examples)

    class2index, index2class = class_labels(training_examples + dev_examples)
    num_classes = len(class2index)

    print(f'Number of classes in dataset: {num_classes}')
    print()

    if args.test:
        model = torch.load(args.load_model)
        test_dataset = QuestionDataset(test_examples, word2index, num_classes, class2index)
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
            model = DanModel(num_classes, len(voc))
            model.to(device)
        print(model)
        #### Load batchifed dataset
        train_dataset = QuestionDataset(training_examples, word2index, num_classes, class2index)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = QuestionDataset(dev_examples, word2index, num_classes, class2index)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                                 sampler=dev_sampler, num_workers=0,
                                                 collate_fn=batchify)
        accuracy = 0
        for epoch in range(args.num_epochs):
            print('start epoch %d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       sampler=train_sampler, num_workers=0,
                                                       collate_fn=batchify)
            accuracy = train(args, model, train_loader, dev_loader, accuracy, device)
        print('start testing:\n')

        test_dataset = QuestionDataset(test_examples, word2index, num_classes, class2index)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  sampler=test_sampler, num_workers=0,
                                                  collate_fn=batchify)
        evaluate(test_loader, model, device)



