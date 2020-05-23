import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import threading

import utils
import config
import data
import models
import eval


class Train(object):

    def __init__(self, vocab_file_path=None, model_file_path=None):
        """

        :param vocab_file_path: tuple of code vocab, ast vocab, nl vocab, if given, build vocab by given path
        :param model_file_path:
        """

        # dataset
        self.train_dataset = data.CodePtrDataset(code_path=config.train_code_path,
                                                 ast_path=config.train_sbt_path,
                                                 nl_path=config.train_nl_path)
        self.train_dataset_size = len(self.train_dataset)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                            code_vocab=self.code_vocab,
                                                                                            ast_vocab=self.ast_vocab,
                                                                                            nl_vocab=self.nl_vocab))

        # vocab
        self.code_vocab: utils.Vocab
        self.ast_vocab: utils.Vocab
        self.nl_vocab: utils.Vocab
        # load vocab from given path
        if vocab_file_path:
            code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
            self.code_vocab = utils.load_vocab_pk(code_vocab_path)
            self.ast_vocab = utils.load_vocab_pk(ast_vocab_path)
            self.nl_vocab = utils.load_vocab_pk(nl_vocab_path)
        # new vocab
        else:
            self.code_vocab = utils.Vocab('code_vocab')
            self.ast_vocab = utils.Vocab('ast_vocab')
            self.nl_vocab = utils.Vocab('nl_vocab')
            codes, asts, nls = self.train_dataset.get_dataset()
            for code, ast, nl in zip(codes, asts, nls):
                self.code_vocab.add_sentence(code)
                self.ast_vocab.add_sentence(ast)
                self.nl_vocab.add_sentence(nl)

            self.origin_code_vocab_size = len(self.code_vocab)
            self.origin_nl_vocab_size = len(self.nl_vocab)

            # trim vocabulary
            self.code_vocab.trim(config.code_vocab_size)
            self.nl_vocab.trim(config.nl_vocab_size)
            # save vocabulary
            self.code_vocab.save(config.code_vocab_path)
            self.ast_vocab.save(config.ast_vocab_path)
            self.nl_vocab.save(config.nl_vocab_path)
            self.code_vocab.save_txt(config.code_vocab_txt_path)
            self.ast_vocab.save_txt(config.ast_vocab_txt_path)
            self.nl_vocab.save_txt(config.nl_vocab_txt_path)

        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab_size = len(self.nl_vocab)

        # model
        self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                  ast_vocab_size=self.ast_vocab_size,
                                  nl_vocab_size=self.nl_vocab_size,
                                  model_file_path=model_file_path)
        self.params = list(self.model.code_encoder.parameters()) + \
            list(self.model.ast_encoder.parameters()) + \
            list(self.model.reduce_hidden.parameters()) + \
            list(self.model.decoder.parameters())

        # optimizer
        self.optimizer = Adam([
            {'params': self.model.code_encoder.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.ast_encoder.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.reduce_hidden.parameters(), 'lr': config.reduce_hidden_lr},
            {'params': self.model.decoder.parameters(), 'lr': config.decoder_lr},
            
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        if config.use_lr_decay:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=config.lr_decay_every,
                                                    gamma=config.lr_decay_rate)

        # best score and model(state dict)
        self.min_loss: float = 1000
        self.best_model: dict = {}
        self.best_epoch_batch: (int, int) = (None, None)

        # eval instance
        self.eval_instance = eval.Eval(self.get_cur_state_dict())

        config.model_dir = os.path.join(config.model_dir, utils.get_timestamp())
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)

    def run_train(self):
        """
        start training
        :return:
        """
        self.train_iter()
        return self.best_model

    def train_one_batch(self, batch, batch_size, criterion):
        """
        train one batch
        :param batch: get from collate_fn of corresponding dataloader
        :param batch_size: batch size
        :param criterion: loss function
        :return: avg loss
        """
        nl_batch = batch[4]

        self.optimizer.zero_grad()

        decoder_outputs = self.model(batch, batch_size, self.nl_vocab)     # [T, B, nl_vocab_size]

        decoder_outputs = decoder_outputs.view(-1, config.nl_vocab_size)
        nl_batch = nl_batch.view(-1)

        loss = criterion(decoder_outputs, nl_batch)
        loss.backward()

        # address over fit
        torch.nn.utils.clip_grad_norm_(self.params, 5)

        self.optimizer.step()

        return loss

    def train_iter(self):
        start_time = time.time()

        plot_loss = 0

        criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))

        for epoch in range(config.n_epochs):
            print_loss = 0
            last_print_index = 0
            for index_batch, batch in enumerate(self.train_dataloader):

                batch_size = len(batch[0][0])

                loss = self.train_one_batch(batch, batch_size, criterion)
                print_loss += loss.item()
                plot_loss += loss.item()

                # print train progress details
                if index_batch % config.print_every == 0:
                    cur_time = time.time()
                    utils.print_train_progress(start_time=start_time, cur_time=cur_time, epoch=epoch,
                                               n_epochs=config.n_epochs, index_batch=index_batch, batch_size=batch_size,
                                               dataset_size=self.train_dataset_size, loss=print_loss,
                                               last_print_index=last_print_index)
                    print_loss = 0
                    last_print_index = index_batch

                # plot train progress details
                if index_batch % config.plot_every == 0:
                    pass

                # save check point
                if config.use_check_point and index_batch % config.save_check_point_every == 0:
                    pass

                # validate on the valid dataset every config.valid_every batches
                if config.validate_during_train and index_batch % config.validate_every == 0 and index_batch != 0:
                    print('\nValidating the model at epoch {}, batch {} on valid dataset......'.format(
                        epoch, index_batch))
                    config.logger.info('Validating the model at epoch {}, batch {} on valid dataset.'.format(
                        epoch, index_batch))
                    self.valid_state_dict(state_dict=self.get_cur_state_dict(), epoch=epoch, batch=index_batch)

            # validate on the valid dataset every epoch
            if config.validate_during_train:
                print('\nValidating the model at the end of epoch {} on valid dataset......'.format(epoch))
                config.logger.info('Validating the model at the end of epoch {} on valid dataset.'.format(epoch))
                self.valid_state_dict(self.get_cur_state_dict(), epoch=epoch)

            if config.use_lr_decay:
                self.lr_scheduler.step()

        # save the best model
        if config.save_best_model:
            best_model_name = 'best_epoch-{}_batch-{}.pt'.format(
                self.best_epoch_batch[0], self.best_epoch_batch[1] if self.best_epoch_batch[1] != -1 else 'last')
            self.save_model(name=best_model_name, state_dict=self.best_model)

    def save_model(self, name=None, state_dict=None):
        """
        save current model
        :param name: if given, name the model file by given name, else by current time
        :param state_dict: if given, save the given state dict, else save current model
        :return:
        """
        if state_dict is None:
            state_dict = self.get_cur_state_dict()
        if name is None:
            model_save_path = os.path.join(config.model_dir, 'model_{}.pt'.format(utils.get_timestamp()))
        else:
            model_save_path = os.path.join(config.model_dir, name)
        torch.save(state_dict, model_save_path)

    def save_check_point(self):
        pass

    def get_cur_state_dict(self) -> dict:
        """
        get current state dict of model
        :return:
        """
        state_dict = {
                'code_encoder': self.model.code_encoder.state_dict(),
                'ast_encoder': self.model.ast_encoder.state_dict(),
                'reduce_hidden': self.model.reduce_hidden.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        return state_dict

    def valid_state_dict(self, state_dict, epoch, batch=-1):
        self.eval_instance.set_state_dict(state_dict)
        loss = self.eval_instance.run_eval()

        if config.save_valid_model:
            model_name = 'model_valid-loss-{:.4f}_epoch-{}_batch-{}.pt'.format(loss, epoch, batch)
            save_thread = threading.Thread(target=self.save_model, args=(model_name, state_dict))
            save_thread.start()

        if loss < self.min_loss:
            self.min_loss = loss
            self.best_model = state_dict
            self.best_epoch_batch = (epoch, batch)
