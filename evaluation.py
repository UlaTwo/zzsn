import math
import time
import os

from datetime import datetime
from tqdm import tqdm

import torch
import wandb


class ModelEvaluation():

    def __init__(self, training_batched, validation_batched, testing_batched,
                 model, config, log_interval, criterion, optimizer, scheduler):

        self.training_batched = training_batched
        self.validation_batched = validation_batched
        self.testing_batched = testing_batched
        self.model = model
        self.config = config
        self.log_interval = log_interval
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def run(self):
        wandb.watch(self.model)
        best_validation_loss = None
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_directory = os.path.join(
            'models', f'model_{timestamp}_{self.config.embedding_size}em_{self.config.num_hidden}hu_{self.config.num_layers}l')
        os.makedirs(model_directory)

        try:
            for epoch in range(1, self.config.epochs + 1):
                print(f'Training epoch {epoch} out of {self.config.epochs}...')
                wandb.log({"Epoch": epoch})

                current_lr = self.optimizer.param_groups[0]['lr']
                wandb.log({"Learning rate": current_lr})

                epoch_start_time = time.time()

                self.train_step(epoch)

                validation_loss = self.evaluate(self.validation_batched)
                validation_perplexity = math.exp(validation_loss)

                elapsed_time = time.time() - epoch_start_time

                print('-' * 89)
                print(
                    f'Finished epoch {epoch} | elapsed: {elapsed_time:5.2f}s | validation loss {validation_loss:5.2f} | perplexity {validation_perplexity:5.2f}')
                print('-' * 89)
                wandb.log({"Validation Loss": validation_loss})
                wandb.log({"Validation Perplexity": validation_perplexity})

                if best_validation_loss is None or validation_loss < best_validation_loss:
                    model_path = os.path.join(
                        model_directory, f'model_{epoch}.pth')
                    with open(model_path, 'wb') as f:
                        torch.save(self.model, f)
                    best_validation_loss = validation_loss

                self.scheduler.step(validation_loss)
                wandb.log({"Epoch": epoch})
        except KeyboardInterrupt:
            print('Stopped training...')

        print('Finished training! Evaluating model on testing dataset...')
        test_loss = self.evaluate(self.testing_batched)
        test_perplexity = math.exp(test_loss)

        print(
            f'Finished testing! Test loss: {test_loss:.2f} Perplexity: {test_perplexity:.2f}')

        wandb.log({"Test Final Perplexity": test_perplexity})
        wandb.log({"Test Final Loss": test_loss})
        wandb.finish()

    def get_ith_batch(self, batched_data, i):

        seq_len_adjusted = min(self.config.seq_len, len(batched_data) - 1 - i)
        input = batched_data[i:i+seq_len_adjusted]
        target = batched_data[i+1:i+1+seq_len_adjusted].view(-1)
        return input, target

    def repackage_hidden(self, hidden):
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in hidden)

    def evaluate(self, dataset):

        self.model.eval()

        total_loss = 0.
        hidden = self.model.init_hidden(self.config.batch_size)

        with torch.no_grad():
            for i in range(0, dataset.size(0) - 1, self.config.seq_len):
                input, targets = self.get_ith_batch(dataset, i)
                output, hidden = self.model(input, hidden)
                hidden = self.repackage_hidden(hidden)
                loss = self.criterion(output, targets)
                total_loss += len(input) * loss.item()
        return total_loss / (len(dataset) - 1)

    def train_step(self, epoch):

        self.model.train()
        total_loss = 0.
        start_time = time.time()
        hidden = self.model.init_hidden(self.config.batch_size)
        for batch, i in enumerate(tqdm(range(0, self.training_batched.size(0) - 1, self.config.seq_len))):
            input, targets = self.get_ith_batch(self.training_batched, i)

            self.model.zero_grad()
            hidden = self.repackage_hidden(hidden)
            output, hidden = self.model(input, hidden)
            loss = self.criterion(output, targets)
            wandb.log({"Training Item Loss": loss.item()})

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip)
            self.optimizer.step()

            total_loss += loss.item()

            if batch % self.log_interval == 0 and batch > 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / self.log_interval
                perplexity = math.exp(avg_loss)

                total_batches = len(
                    self.training_batched) // self.config.seq_len
                ms_per_batch = 1000.0 * elapsed / self.log_interval

                print(
                    f'Epoch {epoch} | {batch}/{total_batches} batches | ms/batch {ms_per_batch:5.2f} | avg. loss {avg_loss:5.2f}| perplexity {perplexity:5.2f}')
                wandb.log({"Training Avg Loss": avg_loss})
                wandb.log({"Training Preplexity": perplexity})
                total_loss = 0
                start_time = time.time()
