import torch
import torch.nn as nn
from torch.optim import Adam
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
        _, _, _, _, nl_batch, _ = batch

        self.optimizer.zero_grad()

        decoder_outputs = self.model(batch, batch_size, self.nl_vocab)     # [T, B, nl_vocab_size]

        decoder_outputs = decoder_outputs.view(-1, config.nl_vocab_size)
        nl_batch = nl_batch.view(-1)

        loss = criterion(decoder_outputs, nl_batch)
        loss.backward()

        # address over fit
        # torch.nn.utils.clip_grad_norm(self.params, 5)

        self.optimizer.step()

        return loss

        # self.optimizer.zero_grad()
        #
        # # batch: [T, B]
        # # code_batch, code_seq_lens, code_pos, \
        # #     ast_batch, ast_seq_lens, ast_pos, \
        # #     nl_batch, nl_seq_lens = batch
        # code_batch, code_seq_lens, \
        #     ast_batch, ast_seq_lens, \
        #     nl_batch, nl_seq_lens = batch
        #
        # # print(code_batch)
        # # print(code_seq_lens)
        # # print(code_pos)
        # # print(ast_batch)
        # # print(ast_seq_lens)
        # # print(ast_pos)
        # # print(nl_batch)
        # # print(nl_seq_lens)
        #
        # print(code_batch)
        # print(code_seq_lens)
        # print(ast_batch)
        # print(ast_seq_lens)
        # print(nl_batch)
        # print(nl_seq_lens)
        #
        # # encode
        # # outputs: [T, B, H]
        # # hidden: [2, B, H]
        # code_outputs, code_hidden = self.model.code_encoder(code_batch, code_seq_lens)
        # ast_outputs, ast_hidden = self.model.ast_encoder(ast_batch, ast_seq_lens)
        # # print('encoder outputs shape:', code_outputs.shape, ast_outputs.shape)
        # # print('encoder hidden shape:', code_hidden.shape, ast_hidden.shape)
        #
        # # restore the code outputs and ast outputs to match the sequence of nl
        # # code_outputs = utils.restore_encoder_outputs(code_outputs, code_pos)
        # # code_hidden = utils.restore_encoder_outputs(code_hidden, code_pos)
        # # ast_outputs = utils.restore_encoder_outputs(ast_outputs, ast_pos)
        # # ast_hidden = utils.restore_encoder_outputs(ast_hidden, ast_pos)
        # # print('restore outputs shape:', code_outputs.shape, ast_outputs.shape)
        # # print('restore hidden shape:', code_hidden.shape, ast_hidden.shape)
        #
        # # data for decoder
        # code_hidden = code_hidden[:1]   # [1, B, H]
        # ast_hidden = ast_hidden[:1]     # [1, B, H]
        # # decoder_hidden = torch.cat([code_hidden, ast_hidden], dim=2)    # [1, B, 2*H]
        # decoder_hidden = self.model.reduce_hidden(code_hidden, ast_hidden)  # [1, B, H]
        #
        # target_length = max(nl_seq_lens)
        # decoder_inputs = utils.init_decoder_inputs(batch_size=batch_size, vocab=self.nl_vocab)  # [B]
        # # print('decoder inputs shape:', decoder_inputs.shape)
        # # print('decoder hidden shape:', decoder_hidden.shape)
        #
        # # decode
        # loss: torch.Tensor = torch.tensor(0.).to(config.device)
        #
        # # whether use teacher forcing
        # if config.use_teacher_forcing and random.random() < config.use_teacher_forcing:
        #     for step in range(min(config.max_decode_steps, target_length)):
        #         # decoder_outputs: [B, nl_vocab_size]
        #         # decoder_hidden: [1, B, H]
        #         # attn_weights: [B, 1, T]
        #         decoder_outputs, decoder_hidden, \
        #             code_attn_weights, ast_attn_weights = self.model.decoder(inputs=decoder_inputs,
        #                                                                      last_hidden=decoder_hidden,
        #                                                                      code_outputs=code_outputs,
        #                                                                      ast_outputs=ast_outputs)
        #         # use ground truth to be the next input of decoder
        #         decoder_inputs = nl_batch[step]
        #         # decoder_hidden = torch.cat([decoder_hidden, decoder_hidden], dim=2)
        #         loss += criterion(decoder_outputs, nl_batch[step])
        # else:
        #     for step in range(min(config.max_decode_steps, target_length)):
        #         # decoder_outputs: [B, nl_vocab_size]
        #         # decoder_hidden: [1, B, H]
        #         # attn_weights: [B, 1, T]
        #         decoder_outputs, decoder_hidden, \
        #             code_attn_weights, ast_attn_weights = self.model.decoder(inputs=decoder_inputs,
        #                                                                      last_hidden=decoder_hidden,
        #                                                                      code_outputs=code_outputs,
        #                                                                      ast_outputs=ast_outputs)
        #         # use the output of decoder to be the next input of decoder
        #         _, indices = decoder_outputs.topk(1)  # [B, 1]
        #         decoder_inputs = indices.squeeze().detach()  # [B]
        #         decoder_inputs = decoder_inputs.to(config.device)
        #         # decoder_hidden = torch.cat([decoder_hidden, decoder_hidden], dim=2)
        #         loss += criterion(decoder_outputs, nl_batch[step])
        #
        # loss.backward()
        #
        # # address over fit
        # # torch.nn.utils.clip_grad_norm(self.params, 5)
        #
        # # optimize
        # self.optimizer.step()
        #
        # return loss.item() / target_length
        # # return 0

    def train_iter(self):
        start_time = time.time()

        plot_loss = 0

        criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))

        for epoch in range(config.n_epochs):
            print_loss = 0
            last_print_index = 0
            for index_batch, batch in enumerate(self.train_dataloader):

                # if index_batch == 1:
                #     break

                batch_size = len(batch[0][0])
                # print('batch_size:', batch_size)

                loss = self.train_one_batch(batch, batch_size, criterion)
                # print('loss:', loss)
                # print()

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
                    self.valid_state_dict(state_dict=self.get_cur_state_dict(), epoch=epoch, batch=index_batch)

            # validate on the valid dataset every epoch
            if config.validate_during_train:
                print('\nValidating the model at the end of epoch {} on valid dataset......'.format(epoch))
                self.valid_state_dict(self.get_cur_state_dict(), epoch=epoch)

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
