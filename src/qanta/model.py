import argparse

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import numpy as np

import json
import time
import nltk


class Model(nn.Module):
    def __init__(self,
                 n_classes: int,
                 vocabulary_size: int,
                 embedded_dimension: int = 50,
                 n_hidden: int = 50,
                 dropout_rate: float = .5):
        super(Model, self).__init__()
        self.n_classes = n_classes
        self.vocabulary_size = vocabulary_size
        self.embedded_dimension = embedded_dimension
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.embeddings = nn.Embedding(self.vocabulary_size, self.embedded_dimension, padding_idx=0)

        self.layer1 = nn.Linear(embedded_dimension, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_classes)

        self.classifier = nn.Sequential(self.layer1,
                                        nn.ReLU(),
                                        nn.Dropout(p=self.dropout_rate, inplace=False),
                                        self.layer2,)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, input_text, text_length, is_prob=False):
        logits = torch.LongTensor([0.0] * self.n_classes)

        embedding = self.embeddings(input_text)
        average_embedding = embedding.sum(1) / text_length.view(embedding.size(0), -1)

        logits = self.classifier(average_embedding) if not is_prob else self._softmax(logits)

        return logits


def evaluate(*args):
    pass


def train(args,
          model: Model,
          train_data_loader,
          dev_data_loader,
          accuracy,
          device):

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    for i, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        optimizer.zero_grad()
        result = model(question_text, question_len)
        loss = criterion(result, labels)
        loss.backward()
        optimizer.step()

        clip_grad_norm_(model.parameters(), args.grad_clipping)
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()

        if not i % 50:
            print('number of steps: %d, loss: %.5f time: %.5f' % (i, print_loss_total / 50, time.time()- start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, args.save_model)
                accuracy = curr_accuracy
    return accuracy