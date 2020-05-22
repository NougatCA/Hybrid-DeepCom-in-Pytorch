import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import os
import time
import threading

import models
import data
import utils
import config


class Eval(object):

    def __init__(self, model):

        # vocabulary
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)

        # dataset
        self.dataset = data.CodePtrDataset(code_path=config.valid_code_path,
                                           ast_path=config.valid_sbt_path,
                                           nl_path=config.valid_nl_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.eval_batch_size,
                                     collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                      code_vocab=self.code_vocab,
                                                                                      ast_vocab=self.ast_vocab,
                                                                                      nl_vocab=self.nl_vocab))

        # model
        if isinstance(model, str):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_file_path=os.path.join(config.model_dir, model),
                                      is_eval=True)
        elif isinstance(model, dict):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_state_dict=model,
                                      is_eval=True)
        else:
            raise Exception('Parameter \'model\' for class \'Eval\' must be file name or state_dict of the model.')

    def run_eval(self):
        loss = self.eval_iter()
        return loss

    def eval_one_batch(self, batch, batch_size, criterion):
        """
        evaluate one batch
        :param batch:
        :param batch_size:
        :param criterion:
        :return:
        """
        with torch.no_grad():

            # code_batch and ast_batch: [T, B]
            # nl_batch is raw data, [B, T] in list
            # nl_seq_lens is None
            nl_batch = batch[4]

            decoder_outputs = self.model(batch, batch_size, self.nl_vocab)  # [T, B, nl_vocab_size]

            decoder_outputs = decoder_outputs.view(-1, config.nl_vocab_size)
            nl_batch = nl_batch.view(-1)

            loss = criterion(decoder_outputs, nl_batch)

            return loss

    def eval_iter(self):
        """
        evaluate model on self.dataset
        :return: scores
        """
        epoch_loss = 0
        criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))

        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch[0].shape[1]

            loss = self.eval_one_batch(batch, batch_size, criterion=criterion)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.dataloader)

        print('Validate completed, avg loss: {:.4f}.\n'.format(avg_loss))
        config.logger.info('Validate completed, avg loss: {:.4f}.\n'.format(avg_loss))
        return avg_loss

    def set_state_dict(self, state_dict):
        self.model.set_state_dict(state_dict)


class BeamNode(object):

    def __init__(self, sentence_indices, log_probs, hidden):
        """

        :param sentence_indices: indices of words of current sentence (from root to current node)
        :param log_probs: log prob of node of sentence
        :param hidden: [1, 1, H]
        """
        self.sentence_indices = sentence_indices
        self.log_probs = log_probs
        self.hidden = hidden

    def extend_node(self, word_index, log_prob, hidden):
        return BeamNode(sentence_indices=self.sentence_indices + [word_index],
                        log_probs=self.log_probs + [log_prob],
                        hidden=hidden)

    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.sentence_indices)

    def word_index(self):
        return self.sentence_indices[-1]


class Test(object):

    def __init__(self, model):

        # vocabulary
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)

        # dataset
        self.dataset = data.CodePtrDataset(code_path=config.test_code_path,
                                           ast_path=config.test_sbt_path,
                                           nl_path=config.test_nl_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.test_batch_size,
                                     collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                      code_vocab=self.code_vocab,
                                                                                      ast_vocab=self.ast_vocab,
                                                                                      nl_vocab=self.nl_vocab,
                                                                                      raw_nl=True))

        # model
        if isinstance(model, str):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_file_path=os.path.join(config.model_dir, model),
                                      is_eval=True)
        elif isinstance(model, dict):
            self.model = models.Model(code_vocab_size=self.code_vocab_size,
                                      ast_vocab_size=self.ast_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_state_dict=model,
                                      is_eval=True)
        else:
            raise Exception('Parameter \'model\' for class \'Test\' must be file name or state_dict of the model.')

    def run_test(self) -> dict:
        """
        start test
        :return: scores dict, key is name and value is score
        """
        c_bleu, avg_s_bleu, avg_meteor = self.test_iter()
        scores_dict = {
            'c_bleu': c_bleu,
            's_bleu': avg_s_bleu,
            'meteor': avg_meteor
        }
        utils.print_test_scores(scores_dict)
        return scores_dict

    def test_one_batch(self, batch, batch_size):
        """

        :param batch:
        :param batch_size:
        :return:
        """
        with torch.no_grad():
            nl_batch = batch[4]

            # outputs: [T, B, H]
            # hidden: [1, B, H]
            code_outputs, ast_outputs, decoder_hidden = \
                self.model(batch, batch_size, self.nl_vocab, is_test=True)

            # decode
            batch_sentences = self.beam_decode(batch_size=batch_size,
                                               code_outputs=code_outputs,
                                               ast_outputs=ast_outputs,
                                               decoder_hidden=decoder_hidden)

            # translate indices into words both for candidates
            candidates = self.translate_indices(batch_sentences)

            # measure
            s_blue_score, meteor_score = utils.measure(batch_size, references=nl_batch, candidates=candidates)

            return nl_batch, candidates, s_blue_score, meteor_score

    def test_iter(self):
        """
        evaluate model on self.dataset
        :return: scores
        """
        start_time = time.time()
        total_references = []
        total_candidates = []
        total_s_bleu = 0
        total_meteor = 0

        out_file = None
        if config.save_test_details:
            try:
                out_file = open(os.path.join(config.out_dir, 'test_details_{}.txt'.format(utils.get_timestamp())),
                                encoding='utf-8',
                                mode='w')
            except IOError:
                print('Test details file open failed.')

        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch[0].shape[1]

            references, candidates, s_blue_score, meteor_score = self.test_one_batch(batch, batch_size)
            total_s_bleu += s_blue_score
            total_meteor += meteor_score
            total_references += references
            total_candidates += candidates

            if index_batch % config.print_every == 0:
                cur_time = time.time()
                utils.print_eval_progress(start_time=start_time, cur_time=cur_time, index_batch=index_batch,
                                          batch_size=batch_size, dataset_size=self.dataset_size,
                                          batch_s_bleu=s_blue_score, batch_meteor=meteor_score)

            if config.save_test_details:
                for index in range(len(references)):
                    out_file.write(' '.join(['Reference:'] + references[index]) + '\n')
                    out_file.write(' '.join(['Candidate:'] + candidates[index]) + '\n')
                    out_file.write('\n')

        # corpus level bleu score
        c_bleu = utils.corpus_bleu_score(references=total_references, candidates=total_candidates)

        avg_s_bleu = total_s_bleu / self.dataset_size
        avg_meteor = total_meteor / self.dataset_size

        if out_file:
            out_file.write('c_bleu: ' + str(c_bleu) + '\n')
            out_file.write('s_bleu: ' + str(avg_s_bleu) + '\n')
            out_file.write('meteor: ' + str(avg_meteor) + '\n')
            out_file.flush()
            out_file.close()

        return c_bleu, avg_s_bleu, avg_meteor

    def greedy_decode(self, batch_size, code_outputs: torch.Tensor,
                      ast_outputs: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        decode for one batch, sentence by sentence
        :param batch_size:
        :param code_outputs: [T, B, H]
        :param ast_outputs: [T, B, H]
        :param decoder_hidden: [1, B, H]
        :return: batch_sentences, [B, config.beam_top_sentence]
        """
        batch_sentences = []
        for index_batch in range(batch_size):
            batch_hidden = decoder_hidden[:, index_batch, :].unsqueeze(1)  # [1, 1, H]
            batch_code_output = code_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]
            batch_ast_output = ast_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]

            decoded_indices = []
            decoder_inputs = torch.tensor([utils.get_sos_index(self.nl_vocab)], device=config.device).long()    # [1]

            for step in range(config.max_decode_steps):
                # batch_output: [1, nl_vocab_size]
                # batch_hidden: [1, H]
                # attn_weights: [1, 1, T]
                # decoder_outputs: [1, nl_vocab_size]
                decoder_outputs, batch_hidden, \
                    code_attn_weights, ast_attn_weights = self.model.decoder(inputs=decoder_inputs,
                                                                             last_hidden=batch_hidden,
                                                                             code_outputs=batch_code_output,
                                                                             ast_outputs=batch_ast_output)
                # log_prob, word_index: [1, 1]
                _, word_index = decoder_outputs.topk(1)
                word_index = word_index[0][0].item()

                decoded_indices.append(word_index)
                if word_index == utils.get_eos_index(self.nl_vocab):
                    break

            batch_sentences.append([decoded_indices])

        return batch_sentences

    def beam_decode(self, batch_size, code_outputs: torch.Tensor,
                    ast_outputs: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        beam decode for one batch, feed one batch for decoder
        :param batch_size:
        :param code_outputs: [T, B, H]
        :param ast_outputs: [T, B, H]
        :param decoder_hidden: [1, B, H]
        :return: batch_sentences, [B, config.beam_top_sentence]
        """
        batch_sentences = []

        # B = 1
        for index_batch in range(batch_size):
            # for each input sentence
            single_decoder_hidden = decoder_hidden[:, index_batch, :].unsqueeze(1)  # [1, 1, H]
            single_code_output = code_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]
            single_ast_output = ast_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]

            root = BeamNode(sentence_indices=[utils.get_sos_index(self.nl_vocab)],
                            log_probs=[0.0],
                            hidden=single_decoder_hidden)

            current_nodes = [root]  # list of nodes to be further extended
            final_nodes = []  # list of end nodes

            for step in range(config.max_decode_steps):
                if len(current_nodes) == 0:
                    break

                candidate_nodes = []  # list of nodes to be extended next step

                feed_inputs = []
                feed_hidden = []

                # B = len(current_nodes) except eos
                extend_nodes = []
                for node in current_nodes:
                    # if current node is EOS
                    if node.word_index() == utils.get_eos_index(self.nl_vocab):
                        final_nodes.append(node)
                        # if number of final nodes reach the beam width
                        if len(final_nodes) >= config.beam_width:
                            break
                        continue

                    extend_nodes.append(node)

                    decoder_inputs = node.word_index()
                    single_decoder_hidden = node.hidden.clone().detach()     # [1, 1, H]

                    feed_inputs.append(decoder_inputs)  # [B]
                    feed_hidden.append(single_decoder_hidden)   # B x [1, 1, H]

                if len(extend_nodes) == 0:
                    break

                feed_batch_size = len(feed_inputs)
                feed_code_outputs = single_code_output.repeat(1, feed_batch_size, 1)
                feed_ast_outputs = single_ast_output.repeat(1, feed_batch_size, 1)

                feed_inputs = torch.tensor(feed_inputs, device=config.device)   # [B]
                feed_hidden = torch.stack(feed_hidden, dim=2).squeeze(0)    # [1, B, H]

                # decoder_outputs: [B, nl_vocab_size]
                # new_decoder_hidden: [1, B, H]
                # attn_weights: [B, 1, T]
                decoder_outputs, new_decoder_hidden, \
                    code_attn_weights, ast_attn_weights = self.model.decoder(inputs=feed_inputs,
                                                                             last_hidden=feed_hidden,
                                                                             code_outputs=feed_code_outputs,
                                                                             ast_outputs=feed_ast_outputs)

                # get top k words
                # log_probs: [B, beam_width]
                # word_indices: [B, beam_width]
                batch_log_probs, batch_word_indices = decoder_outputs.topk(config.beam_width)

                for index_node, node in enumerate(extend_nodes):
                    log_probs = batch_log_probs[index_node]
                    word_indices = batch_word_indices[index_node]
                    hidden = new_decoder_hidden[:, index_node, :].unsqueeze(1)

                    for i in range(config.beam_width):
                        log_prob = log_probs[i]
                        word_index = word_indices[i].item()

                        new_node = node.extend_node(word_index=word_index,
                                                    log_prob=log_prob,
                                                    hidden=hidden)
                        candidate_nodes.append(new_node)

                # sort candidate nodes by log_prb and select beam_width nodes
                candidate_nodes = sorted(candidate_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
                current_nodes = candidate_nodes[: config.beam_width]

            final_nodes += current_nodes
            final_nodes = sorted(final_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
            final_nodes = final_nodes[: config.beam_top_sentences]

            sentences = []
            for final_node in final_nodes:
                sentences.append(final_node.sentence_indices)

            batch_sentences.append(sentences)

        return batch_sentences

    def translate_indices(self, batch_sentences):
        """
        translate indices to words for one batch
        :param batch_sentences: [B, config.beam_top_sentences, sentence_length]
        :return:
        """
        batch_words = []
        for sentences in batch_sentences:
            words = []
            for indices in sentences:
                for index in indices:
                    word = self.nl_vocab.index2word[index]
                    if utils.is_unk(word) or not utils.is_special_symbol(word):
                        words.append(word)
            batch_words.append(words)
        return batch_words
