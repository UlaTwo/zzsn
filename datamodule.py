import torch
import torchtext

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class PennTreebankDataModule():

    def __init__(self, device, batch_size):

        self.device = device
        self.tokenizer = get_tokenizer(None)
        self.batch_size = batch_size

        self.training_batched, self.vocabulary = self.get_training_batched()
        self.testing_batched = self.get_testing_batched()
        self.validation_batched = self.get_validation_batched()
        self.num_tokens = len(self.vocabulary)

    def get_training_batched(self):
        training_set = torchtext.datasets.PennTreebank(split='train')
        training_tokens = self.tokenize_dataset(training_set)
        vocabulary = build_vocab_from_iterator(
            iter([training_tokens]), specials=['<unk>', '<eos>'])
        vocabulary.set_default_index(vocabulary['<unk>'])
        training_data = torch.tensor(
            vocabulary.forward(training_tokens)).type(torch.int64)
        training_batched = self.create_data_batches(
            training_data).to(self.device)

        return training_batched, vocabulary

    def get_testing_batched(self):
        testing_set = torchtext.datasets.PennTreebank(split='test')
        testing_tokens = self.tokenize_dataset(testing_set)
        testing_data = torch.tensor(
            self.vocabulary.forward(testing_tokens)).type(torch.int64)
        testing_batched = self.create_data_batches(
            testing_data).to(self.device)

        return testing_batched

    def get_validation_batched(self):
        validation_set = torchtext.datasets.PennTreebank(split='valid')
        validation_tokens = self.tokenize_dataset(validation_set)
        validation_data = torch.tensor(
            self.vocabulary.forward(validation_tokens)).type(torch.int64)
        validation_batched = self.create_data_batches(
            validation_data).to(self.device)

        return validation_batched

    def tokenize_dataset(self, dataset):
        tokens = []
        for sentence in dataset:
            tokenized = self.tokenizer(sentence)
            tokenized.append('<eos>')
            tokens += tokenized

        return tokens

    def create_data_batches(self, data):
        num_batches = data.size(0) // self.batch_size
        data_size_trimmed = num_batches * self.batch_size
        data = data.narrow(0, 0, data_size_trimmed)
        data = data.view(self.batch_size, -1).t().contiguous()
        return data

    def get_datasets(self):
        return self.training_batched, self.testing_batched, self.validation_batched

    def get_num_tokens(self):
        return self.num_tokens
