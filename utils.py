import time

import torch
import itertools
import os
import pickle
import numpy as np
import nltk

import config


# special vocabulary symbols

_PAD = '<PAD>'
_SOS = '<s>'    # start of sentence
_EOS = '</s>'   # end of sentence
_UNK = '<UNK>'  # OOV word

_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]


class Vocab(object):

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        self.add_sentence(_START_VOCAB)     # add special symbols

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, max_vocab_size=None):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        # trim according to minimum count of words
        if config.trim_vocab_min_count:
            keep_words += _START_VOCAB
            # filter words
            for word, count in self.word2count.items():
                if count >= config.vocab_min_count:
                    keep_words.append(word)

        # trim according to maximum size of vocabulary
        if config.trim_vocab_max_size:
            if max_vocab_size is None:
                raise Exception('Parameter \'max_vocab_size\'must be passed if \'config.trim_vocab_max_size\' is True')
            if self.num_words <= max_vocab_size:
                return
            for special_symbol in _START_VOCAB:
                self.word2count.pop(special_symbol)
            keep_words = list(self.word2count.items())
            keep_words = sorted(keep_words, key=lambda item: item[1], reverse=True)
            keep_words = keep_words[: max_vocab_size - len(_START_VOCAB)]
            keep_words = _START_VOCAB + [word for word, _ in keep_words]

        # reinitialize
        self.word2index.clear()
        self.word2count.clear()
        self.index2word.clear()
        self.num_words = 0
        self.add_sentence(keep_words)

    def save(self, name):
        """
        save self as pickle file named as given name
        :param name: file name
        :return:
        """
        path = os.path.join(config.vocab_dir, name)
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def save_txt(self, name):
        """
        save self vocabulary as a txt file named by self.name and timestamp
        :return:
        """
        txt_path = os.path.join(config.vocab_dir, name)
        with open(txt_path, 'w', encoding='utf-8') as file:
            for word, _ in self.word2index.items():
                file.write(word + '\n')

    def __len__(self):
        return self.num_words


class EarlyStopping(object):

    def __init__(self, patience=config.early_stopping_patience, verbose=False, delta=0):
        """

        :param patience: How long to wait after last time validation loss improved
        :param verbose: If True, prints a message for each validation loss improvement
        :param delta: Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.min_valid_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, valid_loss):

        if self.min_valid_loss is None:
            self.min_valid_loss = valid_loss
        elif valid_loss > self.min_valid_loss - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}.\n'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print('Early stop.\n')
        else:
            self.min_valid_loss = valid_loss
            self.counter = 0


def load_vocab_pk(file_name) -> Vocab:
    """
    load pickle file by given file name
    :param file_name:
    :return:
    """
    path = os.path.join(config.vocab_dir, file_name)
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    if not isinstance(vocab, Vocab):
        raise Exception('Pickle file: \'{}\' is not an instance of class \'Vocab\''.format(path))
    return vocab


def get_timestamp():
    """
    return the current timestamp, eg. 20200222_151420
    :return: current timestamp
    """
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def load_dataset(dataset_path) -> list:
    """
    load the dataset from given path
    :param dataset_path: path of dataset
    :return: lines from the dataset
    """
    lines = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            words = line.strip().split(' ')
            lines.append(words)
    return lines


def filter_data(codes, asts, nls):
    """
    filter the data according to the rules
    :param codes: list of tokens of source codes
    :param asts: list of tokens of sequence asts
    :param nls: list of tokens of comments
    :return: filtered codes, asts and nls
    """
    assert len(codes) == len(asts)
    assert len(asts) == len(nls)

    new_codes = []
    new_asts = []
    new_nls = []
    for i in range(len(codes)):
        code = codes[i]
        ast = asts[i]
        nl = nls[i]
        if len(code) > config.max_code_length or len(nl) > config.max_nl_length or len(nl) < config.min_nl_length:
            continue
        new_codes.append(code)
        new_asts.append(ast)
        new_nls.append(nl)
    return new_codes, new_asts, new_nls


def init_vocab(name, lines, trim=False, min_count=None):
    """
    initialize the vocab by given name and dataset, trim if necessary
    :param name: name of vocab
    :param lines: dataset
    :param trim: whether trim
    :param min_count: minimum count of word
    :return: vocab
    """
    vocab = Vocab(name)
    for line in lines:
        vocab.add_sentence(line)
    if trim:
        vocab.trim(min_count)
    return vocab


def init_decoder_inputs(batch_size, vocab: Vocab) -> torch.Tensor:
    """
    initialize the input of decoder
    :param batch_size:
    :param vocab:
    :return: initial decoder input, torch tensor, [batch_size]
    """
    return torch.tensor([vocab.word2index[_SOS]] * batch_size, device=config.device)


def filter_oov(inputs, vocab: Vocab):
    """
    replace the oov words with UNK token
    :param inputs: inputs, [time_step, batch_size]
    :param vocab: corresponding vocab
    :return: filtered inputs, numpy array, [time_step, batch_size]
    """
    unk = vocab.word2index[_UNK]
    for index_step, step in enumerate(inputs):
        for index_word, word in enumerate(step):
            if word >= vocab.num_words:
                inputs[index_step][index_word] = unk
    return inputs


def get_seq_lens(batch: list) -> list:
    """
    get sequence lengths of given batch
    :param batch: [B, T]
    :return: sequence lengths
    """
    seq_lens = []
    for seq in batch:
        seq_lens.append(len(seq))
    return seq_lens


def pad_one_batch(batch: list, vocab: Vocab) -> torch.Tensor:
    """
    pad batch using _PAD token and get the sequence lengths
    :param batch: one batch, [B, T]
    :param vocab: corresponding vocab
    :return:
    """
    batch = list(itertools.zip_longest(*batch, fillvalue=vocab.word2index[_PAD]))
    batch = [list(b) for b in batch]
    return torch.tensor(batch, device=config.device).long()


def indices_from_batch(batch: list, vocab: Vocab) -> list:
    """
    translate the word in batch to corresponding index by given vocab, then append the EOS token to each sentence
    :param batch: batch to be translated, [B, T]
    :param vocab: Vocab
    :return: translated batch, [B, T]
    """
    indices = []
    for sentence in batch:
        indices_sentence = []
        for word in sentence:
            if word not in vocab.word2index:
                indices_sentence.append(vocab.word2index[_UNK])
            else:
                indices_sentence.append(vocab.word2index[word])
        indices_sentence.append(vocab.word2index[_EOS])
        indices.append(indices_sentence)
    return indices


def sort_batch(batch) -> (list, list, list):
    """
    sort one batch, return indices and sequence lengths
    :param batch: [B, T]
    :return:
    """
    seq_lens = get_seq_lens(batch)
    pos = np.argsort(seq_lens)[::-1]
    batch = [batch[index] for index in pos]
    seq_lens.sort(reverse=True)
    return batch, seq_lens, pos


def restore_encoder_outputs(outputs: torch.Tensor, pos) -> torch.Tensor:
    """
    restore the outputs or hidden of encoder by given pos
    :param outputs: [T, B, H] or [2, B, H]
    :param pos:
    :return:
    """
    rev_pos = np.argsort(pos)
    outputs = torch.index_select(outputs, 1, torch.tensor(rev_pos, device=config.device))
    return outputs


def get_pad_index(vocab: Vocab) -> int:
    return vocab.word2index[_PAD]


def get_sos_index(vocab: Vocab) -> int:
    return vocab.word2index[_SOS]


def get_eos_index(vocab: Vocab) -> int:
    return vocab.word2index[_EOS]


def collate_fn(batch, code_vocab, ast_vocab, nl_vocab, is_eval=False) -> \
        (torch.Tensor, list, list, torch.Tensor, list, list, torch.Tensor, list):
    """
    process the batch
    :param batch: one batch, first dimension is batch, [B]
    :param code_vocab: [B, T]
    :param ast_vocab: [B, T]
    :param nl_vocab: [B, T]
    :param is_eval: if True then nl_batch will not be translated and returns the raw data
    :return:
    """
    batch = batch[0]
    code_batch = []
    ast_batch = []
    nl_batch = []
    for b in batch:
        code_batch.append(b[0])
        ast_batch.append(b[1])
        nl_batch.append(b[2])

    # transfer words to indices including oov words, and append EOS token to each sentence, list
    code_batch = indices_from_batch(code_batch, code_vocab)  # [B, T]
    ast_batch = indices_from_batch(ast_batch, ast_vocab)  # [B, T]
    if not is_eval:
        nl_batch = indices_from_batch(nl_batch, nl_vocab)  # [B, T]

    # sort each batch in decreasing order and get sequence lengths
    code_batch, code_seq_lens, code_pos = sort_batch(code_batch)
    ast_batch, ast_seq_lens, ast_pos = sort_batch(ast_batch)
    if not is_eval:
        nl_seq_lens = get_seq_lens(nl_batch)
    else:
        nl_seq_lens = None

    # pad and transpose, [T, B], tensor
    code_batch = pad_one_batch(code_batch, code_vocab)
    ast_batch = pad_one_batch(ast_batch, ast_vocab)
    if not is_eval:
        nl_batch = pad_one_batch(nl_batch, nl_vocab)

    return code_batch, code_seq_lens, code_pos, \
        ast_batch, ast_seq_lens, ast_pos, \
        nl_batch, nl_seq_lens


def unsort_collate_fn(batch, code_vocab, ast_vocab, nl_vocab, raw_nl=False):
    """
    process the batch without sorting
    :param batch: one batch, first dimension is batch, [B]
    :param code_vocab: [B, T]
    :param ast_vocab: [B, T]
    :param nl_vocab: [B, T]
    :param raw_nl: if True then nl_batch will not be translated and returns the raw data
    :return:
    """
    batch = batch[0]
    code_batch = []
    ast_batch = []
    nl_batch = []
    for b in batch:
        code_batch.append(b[0])
        ast_batch.append(b[1])
        nl_batch.append(b[2])

    # transfer words to indices including oov words, and append EOS token to each sentence, list
    code_batch = indices_from_batch(code_batch, code_vocab)  # [B, T]
    ast_batch = indices_from_batch(ast_batch, ast_vocab)  # [B, T]
    if not raw_nl:
        nl_batch = indices_from_batch(nl_batch, nl_vocab)  # [B, T]

    code_seq_lens = get_seq_lens(code_batch)
    ast_seq_lens = get_seq_lens(ast_batch)
    nl_seq_lens = get_seq_lens(nl_batch)

    # pad and transpose, [T, B], tensor
    code_batch = pad_one_batch(code_batch, code_vocab)
    ast_batch = pad_one_batch(ast_batch, ast_vocab)
    if not raw_nl:
        nl_batch = pad_one_batch(nl_batch, nl_vocab)

    return code_batch, code_seq_lens, \
        ast_batch, ast_seq_lens, \
        nl_batch, nl_seq_lens


def to_time(float_time):
    """
    translate float time to h, min, s and ms
    :param float_time: time in float
    :return: h, min, s, ms
    """
    time_s = int(float_time)
    time_ms = int((float_time - time_s) * 1000)
    time_h = time_s // 3600
    time_s = time_s % 3600
    time_min = time_s // 60
    time_s = time_s % 60
    return time_h, time_min, time_s, time_ms


def print_train_progress(start_time, cur_time, epoch, n_epochs, index_batch,
                         batch_size, dataset_size, loss, last_print_index):
    spend = cur_time - start_time
    spend_h, spend_min, spend_s, spend_ms = to_time(spend)

    n_iter = (dataset_size + config.batch_size - 1) // config.batch_size
    len_epoch = len(str(n_epochs))
    len_iter = len(str(n_iter))
    percent_complete = (epoch / n_epochs +
                        (1 / n_epochs) / dataset_size * (index_batch * config.batch_size + batch_size)) * 100

    time_remaining = spend / percent_complete * (100 - percent_complete)
    remain_h, remain_min, remain_s, remain_ms = to_time(time_remaining)

    batch_length = index_batch - last_print_index
    if batch_length != 0:
        loss = loss / batch_length

    print('\033[0;36mtime\033[0m: {:2d}h {:2d}min {:2d}s {:3d}ms, '.format(
        spend_h, spend_min, spend_s, spend_ms), end='')
    print('\033[0;36mremaining\033[0m: {:2d}h {:2d}min {:2d}s {:3d}ms, '.format(
        remain_h, remain_min, remain_s, remain_ms), end='')
    print('\033[0;33mepoch\033[0m: %*d/%*d, \033[0;33mbatch\033[0m: %*d/%*d, ' %
          (len_epoch, epoch + 1, len_epoch, n_epochs, len_iter, index_batch, len_iter, n_iter - 1), end='')
    print('\033[0;32mpercent complete\033[0m: {:6.2f}%, \033[0;31mavg loss\033[0m: {:.4f}'.format(
        percent_complete, loss))

    config.logger.info('epoch: {}/{}, batch: {}/{}, avg loss: {:.4f}'.format(
        epoch + 1, n_epochs, index_batch, n_iter - 1, loss))


def plot_train_progress():
    pass


def is_unk(word):
    if word == _UNK:
        return True
    return False


def is_special_symbol(word):
    if word in _START_VOCAB:
        return True
    else:
        return False


def measure(batch_size, references, candidates) -> (float, float):
    """
    measures the top sentence model generated
    :param batch_size:
    :param references: batch of references
    :param candidates: batch of sentences model generated
    :return: total sentence level bleu score, total meteor score
    """
    total_s_bleu = 0
    total_meteor = 0

    for index_batch in range(batch_size):
        reference = references[index_batch]
        candidate = candidates[index_batch]

        # sentence level bleu score
        sentence_bleu = sentence_bleu_score(reference, candidate)
        total_s_bleu += sentence_bleu

        # meteor score
        meteor = meteor_score(reference, candidate)
        total_meteor += meteor

    return total_s_bleu, total_meteor


def sentence_bleu_score(reference, candidate) -> float:
    """
    calculate the sentence level bleu score, 4-gram with weights(0.25, 0.25, 0.25, 0.25)
    :param reference: tokens of reference sentence
    :param candidate: tokens of sentence generated by model
    :return: sentence level bleu score
    """
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
    return nltk.translate.bleu_score.sentence_bleu(references=[reference],
                                                   hypothesis=candidate,
                                                   smoothing_function=smoothing_function.method5)


def corpus_bleu_score(references, candidates) -> float:
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
    return nltk.translate.bleu_score.corpus_bleu(list_of_references=[[reference] for reference in references],
                                                 hypotheses=[candidate for candidate in candidates],
                                                 smoothing_function=smoothing_function.method5)


def meteor_score(reference, candidate):
    """
    meteor score
    :param reference:
    :param candidate:
    :return:
    """
    return nltk.translate.meteor_score.single_meteor_score(reference=' '.join(reference),
                                                           hypothesis=' '.join(candidate))


def rouge():
    pass


def cider():
    pass


# precision, recall, F-score and F-mean
def ir_score():
    pass


def print_test_progress(start_time, cur_time, index_batch, batch_size, dataset_size, batch_s_bleu, batch_meteor):
    spend = cur_time - start_time
    spend_h, spend_min, spend_s, spend_ms = to_time(spend)

    avg_s_bleu = batch_s_bleu / batch_size
    avg_meteor = batch_meteor / batch_size

    n_iter = (dataset_size + batch_size - 1) // batch_size
    len_iter = len(str(n_iter))
    percent_complete = (index_batch * batch_size + batch_size) / dataset_size * 100

    time_remaining = spend / percent_complete * (100 - percent_complete)
    remain_h, remain_min, remain_s, remain_ms = to_time(time_remaining)

    print('\033[0;36mtime\033[0m: {:2d}h {:2d}min {:2d}s {:3d}ms, '.format(
        spend_h, spend_min, spend_s, spend_ms), end='')
    print('\033[0;36mremaining\033[0m: {:2d}h {:2d}min {:2d}s {:3d}ms, '.format(
        remain_h, remain_min, remain_s, remain_ms), end='')
    print('\033[0;33mbatch\033[0m: %*d/%*d, ' %
          (len_iter, index_batch, len_iter, n_iter), end='')
    print('\033[0;32mpercent complete\033[0m: {:6.2f}%, '.format(
        percent_complete), end='')
    print('\033[0;31mavg s-bleu\033[0m: {:.4f}, \033[0;31mavg meteor\033[0m: {:.4f}'.format(
        avg_s_bleu, avg_meteor))


def print_test_scores(scores_dict):
    print('\nTest completed.', end=' ')
    config.logger.info('Test completed.')
    for name, score in scores_dict.items():
        print('{}: {}.'.format(name, score), end=' ')
        config.logger.info('{}: {}.'.format(name, score))
    print()
