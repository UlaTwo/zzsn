import argparse

import torch
import torch.nn as nn

import datamodule
import evaluation
from models import ModelConfig, RNNModel, ModelType

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Language Model')
    parser.add_argument('--model', type=str, default='GRU',
                        help='type of network (GRU/LSTM/RTransformer)')
    parser.add_argument('--em', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--hu', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--attention_heads', type=int, default=32,
                        help='number of heads in multihead attention (used only in R-Transformer)')
    parser.add_argument('--layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--seqlen', type=int, default=32,
                        help='sequence length')
    parser.add_argument('--window_size', type=int, default=9,
                        help='size of local RNN (used only in R-Transformer)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval (in batches)')
    args = parser.parse_args()
    return args


def parse_model_type(model_name: str) -> ModelType:
    model_name_norm = model_name.lower()
    if model_name_norm == 'gru':
        return ModelType.GRU
    elif model_name_norm == 'lstm':
        return ModelType.LSTM
    elif model_name_norm == 'rtransformer':
        return ModelType.R_TRANSFORMER
    else:
        return None


def get_model_config(args):
    return ModelConfig(args.epochs, args.batch_size, args.em,
                       args.hu, args.layers, args.seqlen,
                       args.dropout, args.lr, args.clip)


def main():
    args = parse_arguments()
    config = get_model_config(args)
    model_type = parse_model_type(args.model)

    if model_type is None:
        print('Unknown model type! Please select one of the following model: GRU, LSTM, RTransformer')
        sys.exit(1)
    elif model_type == ModelType.R_TRANSFORMER:
        if config.embedding_size % config.attention_heads != 0:
            print('Embedding size must be divisible by number of attention heads!')
            sys.exit(1)


    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = datamodule.PennTreebankDataModule(device, config.batch_size)
    training_batched, testing_batched, validation_batched = data.get_datasets()

    if model_type == ModelType.GRU:
        model = RNNModel(data.get_num_tokens(), config.embedding_size,
                         config.num_hidden, config.num_layers, config.dropout).to(device)
    if model_type == ModelType.LSTM:
        model = RNNModel('LSTM', data.get_num_tokens(), config.embedding_size,
                         config.num_hidden, config.num_layers, config.dropout).to(device)
    elif model_type == ModelType.R_TRANSFORMER:
        model = RTransformerModel(data.get_num_tokens(), config.embedding_size, config.window_size,
                                  config.attention_heads, config.num_layers,
                                  config.dropout, device).to(device)
    else:
        raise NotImplementedError()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.25, patience=1, verbose=True)

    run_name = f'{args.model.lower()}_{config.num_layers}l_{config.num_hidden}hu_{config.embedding_size}em'
    
    if model_type == ModelType.R_TRANSFORMER:
        run_name = run_name + \
            f'_{config.window_size}w_{config.attention_heads}ah'
    
    wandb.init(project='zzsn_experiments', entity ='zzsn-r-transformer' , config=config.get_dict(), name=run_name)

    model_evaluation = evaluation.ModelEvaluation(training_batched, validation_batched, testing_batched,
                                                  model, config, args.log_interval, criterion, optimizer, scheduler)

    model_evaluation.run()


if __name__ == '__main__':
    main()
