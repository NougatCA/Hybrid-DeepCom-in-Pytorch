import torch
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

    def __init__(self, model, is_test=False):

        # vocabulary
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab = utils.load_vocab_pk(config.ast_vocab_path)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)

        # dataset
        if not is_test:
            self.dataset = data.CodePtrDataset(code_path=config.valid_code_path,
                                               ast_path=config.valid_sbt_path,
                                               nl_path=config.valid_nl_path)
        else:
            self.dataset = data.CodePtrDataset(code_path=config.test_code_path,
                                               ast_path=config.test_sbt_path,
                                               nl_path=config.test_nl_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     collate_fn=lambda *args: utils.collate_fn(*args,
                                                                               code_vocab=self.code_vocab,
                                                                               ast_vocab=self.ast_vocab,
                                                                               nl_vocab=self.nl_vocab,
                                                                               is_eval=True))

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

    def run_eval(self) -> dict:
        """
        start evaluation
        :return: scores dict, key is name and value is score
        """
        c_bleu, avg_s_bleu, avg_meteor = self.eval_iter()
        scores_dict = {
            'c_bleu': c_bleu,
            's_bleu': avg_s_bleu,
            'meteor': avg_meteor
        }
        return scores_dict

    def eval_one_batch(self, batch, batch_size):
        """
        evaluate one batch
        :param batch:
        :param batch_size
        :return: scores measured by nmt measures
        """
        # code_batch and ast_batch: [T, B]
        # nl_batch is raw data, [B, T]
        # nl_seq_lens is None
        code_batch, code_seq_lens, code_pos, \
            ast_batch, ast_seq_lens, ast_pos, \
            nl_batch, nl_seq_lens = batch

        # encode
        code_outputs, code_hidden = self.model.code_encoder(code_batch, code_seq_lens)
        ast_outputs, ast_hidden = self.model.ast_encoder(ast_batch, ast_seq_lens)

        # restore the code outputs and ast outputs to match the sequence of nl
        code_outputs = utils.restore_encoder_outputs(code_outputs, code_pos)
        code_hidden = utils.restore_encoder_outputs(code_hidden, code_pos)
        ast_outputs = utils.restore_encoder_outputs(ast_outputs, ast_pos)
        ast_hidden = utils.restore_encoder_outputs(ast_hidden, ast_pos)

        # merge the outputs and hidden states of two encoders
        # outputs: [T, B, H] + [T, B, H] -> [T, B, 2*H]
        # hidden: [2, B, H] + [2, B, H] -> [B, 2*H]
        code_outputs, ast_outputs = utils.align_encoder_outputs(code_outputs, ast_outputs)
        encoder_outputs = torch.cat((code_outputs, ast_outputs), dim=2)  # [T, B, 2*H]
        encoder_hidden = self.model.reduce_hidden(code_hidden, ast_hidden)  # [B, 2*H]
        decoder_hidden = encoder_hidden.unsqueeze(0)    # [1, B, 2*H]

        # decode
        batch_sentences = self.beam_decode(batch_size=batch_size,
                                           encoder_outputs=encoder_outputs,
                                           decoder_hidden=decoder_hidden)

        # translate indices into words both for candidates
        candidates = self.translate_indices(batch_sentences)

        # measurement
        s_blue_score, meteor_score = utils.measure(batch_size, references=nl_batch, candidates=candidates)

        return nl_batch, candidates, s_blue_score, meteor_score

    def eval_iter(self):
        """
        evaluate model on self.dataset
        :return: scores
        """
        start_time = time.time()
        total_references = []
        total_candidates = []
        total_s_bleu = 0
        total_meteor = 0
        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch[0].shape[1]

            references, candidates, s_blue_score, meteor_score = self.eval_one_batch(batch, batch_size)
            total_s_bleu += s_blue_score
            total_meteor += meteor_score
            total_references += references
            total_candidates += candidates

            if index_batch % config.print_every == 0:
                cur_time = time.time()
                # print_thread = threading.Thread(target=utils.print_eval_progress, args=(start_time,
                #                                                                         cur_time,
                #                                                                         index_batch,
                #                                                                         batch_size,
                #                                                                         self.dataset_size,
                #                                                                         total_s_bleu,
                #                                                                         total_meteor))
                # print_thread.start()
                utils.print_eval_progress(start_time=start_time, cur_time=cur_time, index_batch=index_batch,
                                          batch_size=batch_size, dataset_size=self.dataset_size,
                                          batch_s_bleu=s_blue_score, batch_meteor=meteor_score)

        # corpus level bleu score
        c_bleu = utils.corpus_bleu_score(references=total_references, candidates=total_candidates)

        avg_s_bleu = total_s_bleu / self.dataset_size
        avg_meteor = total_meteor / self.dataset_size

        return c_bleu, avg_s_bleu, avg_meteor

    def beam_decode(self, batch_size, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        beam decode for one batch, sentence by sentence
        :param batch_size:
        :param encoder_outputs: [T, B, 2*H]
        :param decoder_hidden: [1, B, 2*H]
        :return:
        """
        batch_sentences = []

        # B = 1
        for index_batch in range(batch_size):
            # for each input sentence
            batch_hidden = decoder_hidden[:, index_batch, :].unsqueeze(1)       # [1, 1, 2*H]
            batch_output = encoder_outputs[:, index_batch, :].unsqueeze(1)      # [T, 1, 2*H]

            # (word index, log prob, prev sentence)
            root: (int, float, list) = (utils.get_sos_index(self.nl_vocab), 1., [])

            current_nodes = [root]      # list of nodes to be further extended
            final_nodes = []      # list of end nodes

            for step in range(config.max_decode_steps):
                if len(current_nodes) == 0:
                    break

                candidate_nodes = []    # list of nodes to be extended next step

                for node in current_nodes:
                    # if current node is EOS
                    if node[0] == utils.get_eos_index(self.nl_vocab):
                        final_nodes.append(node)
                        # if number of final nodes reach the beam width
                        if len(final_nodes) >= config.beam_width:
                            break
                        continue

                    decoder_inputs = torch.tensor([node[0]], device=config.device).long()     # [1]

                    # batch_output: [1, nl_vocab_size]
                    # batch_hidden: [1, H]
                    # attn_weights: [1, 1, T]
                    # decoder_outputs: [1, nl_vocab_size]
                    decoder_outputs, batch_hidden, attn_weights = self.model.decoder(inputs=decoder_inputs,
                                                                                     last_hidden=batch_hidden,
                                                                                     encoder_outputs=batch_output)

                    # get top k words
                    # log_probs: [1, beam_width]
                    # word_indices: [1, beam_width]
                    log_probs, word_indices = decoder_outputs.topk(config.beam_width)
                    log_probs = log_probs[0]
                    word_indices = word_indices[0]

                    for i in range(config.beam_width):
                        log_prob = log_probs[i]
                        word_index = word_indices[i].item()
                        new_sentence = node[2].copy()
                        new_sentence.append(node[0])
                        new_node = (word_index, node[1] + log_prob, new_sentence)
                        candidate_nodes.append(new_node)

                # sort candidate nodes by log_prb and select beam_width nodes
                candidate_nodes = sorted(candidate_nodes, key=lambda item: item[1], reverse=True)
                current_nodes = candidate_nodes[: config.beam_width]

            final_nodes += current_nodes
            final_nodes = sorted(final_nodes, key=lambda item: item[1], reverse=True)
            final_nodes = final_nodes[: config.beam_top_sentences]

            sentences = []
            for final_node in final_nodes:
                sentence = final_node[2].copy()
                sentence.append(final_node[0])
                sentences.append(sentence)

            batch_sentences.append(sentences)

        return batch_sentences

    def beam_decode_batch(self):
        # beam decode once for whole batch
        pass

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
                    if not utils.is_special_symbol(word):
                        words.append(word)
            batch_words.append(words)
        return batch_words

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
