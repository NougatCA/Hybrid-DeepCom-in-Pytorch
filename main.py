from torch.utils.tensorboard import SummaryWriter

import os

import config
import train
import eval


def _train(vocab_file_path=None, model_file_path=None):
    print('\nStarting the training process......\n')

    if vocab_file_path:
        code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
        print('Vocabulary will be built by given file path.')
        print('\tsource code vocabulary path:\t', os.path.join(config.vocab_dir, code_vocab_path))
        print('\tast of code vocabulary path:\t', os.path.join(config.vocab_dir, ast_vocab_path))
        print('\tcode comment vocabulary path:\t', os.path.join(config.vocab_dir, nl_vocab_path))
    else:
        print('Vocabulary will be built according to dataset.')

    if model_file_path:
        print('Model will be built by given state dict file path:', os.path.join(config.model_dir, model_file_path))
    else:
        print('Model will be created by program.')

    print('\nInitializing the training environments......\n')
    train_instance = train.Train(vocab_file_path=vocab_file_path, model_file_path=model_file_path)
    print('Environments built successfully.\n')
    print('Size of training dataset:', train_instance.train_dataset_size)
    print('\nSize of source code vocabulary:', train_instance.code_vocab_size)
    print('Size of ast of code vocabulary:', train_instance.ast_vocab_size)
    print('Size of code comment vocabulary:', train_instance.nl_vocab_size)

    if config.validate_during_train:
        print('Validate every', config.validate_every, 'batches and each epoch.')
        print('Size of validation dataset:', train_instance.eval_instance.dataset_size)

    # print('\nStart training......\n')
    # train_instance.run_train()
    # print('\nModel training is done.')

    writer = SummaryWriter('runs/CodePtr')
    for _, batch in enumerate(train_instance.train_dataloader):
        batch_size = len(batch[0][0])
        writer.add_graph(train_instance.model, (batch, batch_size, train_instance.nl_vocab))
        break
    writer.close()


def _test():
    pass


if __name__ == '__main__':
    _train()
