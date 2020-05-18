import torch
import os


# paths
dataset_dir = 'dataset_mini/'

if not os.path.exists(dataset_dir):
    raise Exception('Dataset directory not exist.')

train_code_path = os.path.join(dataset_dir, 'train/train.token.code')
train_sbt_path = os.path.join(dataset_dir, 'train/train.token.sbt')
train_nl_path = os.path.join(dataset_dir, 'train/train.token.nl')
train_ast_path = os.path.join(dataset_dir, 'train/train_ast.json')

valid_code_path = os.path.join(dataset_dir, 'valid/valid.token.code')
valid_sbt_path = os.path.join(dataset_dir, 'valid/valid.token.sbt')
valid_nl_path = os.path.join(dataset_dir, 'valid/valid.token.nl')
valid_ast_path = os.path.join(dataset_dir, 'valid/valid_ast.json')

test_code_path = os.path.join(dataset_dir, 'test/test.token.code')
test_sbt_path = os.path.join(dataset_dir, 'test/test.token.sbt')
test_nl_path = os.path.join(dataset_dir, 'test/test.token.nl')
test_ast_path = os.path.join(dataset_dir, 'test/test_ast.json')

model_dir = 'model/'
best_model_path = 'best_model.pt'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

vocab_dir = 'vocab/'
code_vocab_path = 'code_vocab.pk'
ast_vocab_path = 'ast_vocab.pk'
nl_vocab_path = 'nl_vocab.pk'

code_vocab_txt_path = 'code_vocab.txt'
ast_vocab_txt_path = 'ast_vocab.txt'
nl_vocab_txt_path = 'nl_vocab.txt'

if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)


# device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# features
trim_vocab_min_count = False
trim_vocab_max_size = True
use_coverage = False
use_pointer_gen = False
use_teacher_forcing = True
use_check_point = False
save_model_halfway = False
save_model_every_epoch = False
validate_during_train = True
save_best_model = True


# limitations
max_code_length = 200
max_nl_length = 30
min_nl_length = 4
max_decode_steps = 30


# hyperparameters
vocab_min_count = 5
code_vocab_size = 10000  # 30000
nl_vocab_size = 10000    # 30000
embedding_dim = 256
hidden_size = 256
decoder_dropout_rate = 0.5
teacher_forcing_ratio = 1
batch_size = 8     # 32
code_encoder_lr = 0.001
ast_encoder_lr = 0.001
reduce_hidden_lr = 0.001
decoder_lr = 0.01
n_epochs = 1    # 10
beam_width = 5
beam_top_sentences = 1     # number of sentences beam decoder decode for one input
eval_batch_size = 4    # 16
init_uniform_mag = 0.02
init_normal_std = 1e-4


# visualization and resumes
print_every = 100  # 1000
plot_every = 10     # 100
save_model_every = 20   # 2000
save_check_point_every = 10   # 1000
validate_every = 500     # 2000
