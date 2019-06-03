import json
import logging
import argparse
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import src.evaluation as evaluation
from src.cuda import CUDA
import src.data as data
import src.models as models

outputs_dir = "/written_outputs"

model = "delete_retrieve"
deletion_method = "unigram"
thresh = 5
intermediate_outputs_dir = "data/new_intermediate_outputs/delete/" + model + ".outputs." + deletion_method + "." + str(thresh)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument(
    "--bleu",
    help="do BLEU eval",
    action='store_true'
)
parser.add_argument(
    "--overfit",
    help="train continuously on one batch of data",
    action='store_true'
)
args = parser.parse_args()
config = json.load(open(args.config, 'r'))

working_dir = config['data']['working_dir']

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

if not os.path.exists(working_dir + outputs_dir):
    os.makedirs(working_dir + outputs_dir)

config_path = os.path.join(working_dir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/train_log' % working_dir,
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info('Reading data ...')

# src, tgt, src_test, and tgt_test contains the following information
# also src_dev and tgt_dev
# data: all lines in the file
# content: content for each line
# attribute: attribute markers for each line
# tok2id: a token to id map
# id2tok: an id to token map
# dist_measurer: a CorpusSearcher object which allows calculation of tfidf
src, tgt = data.read_nmt_data(
    src=config['data']['src'],
    config=config,
    tgt=config['data']['tgt'],
    attribute_vocab=config['data']['attribute_vocab']
)

logging.info('Finished reading train data ...')
# adding in a dev dataset because we need something to run log perplexity on
src_dev, tgt_dev = data.read_nmt_data(
    src=config['data']['src_dev'],
    config=config,
    tgt=config['data']['tgt_dev'],
    attribute_vocab=config['data']['attribute_vocab']
)

logging.info('Finished reading dev data ...')

# src_test and tgt_test are different from above
# in that they configure the CorpusSearcher to look for target attributes.
# note that the last two named variables are set
src_test, tgt_test = data.read_nmt_data(
    src=config['data']['src_test'],
    config=config,
    tgt=config['data']['tgt_test'],
    attribute_vocab=config['data']['attribute_vocab'],
    train_src=src,
    train_tgt=tgt
)
logging.info('...done reading all data!')

# writing the intermediate outputs of delete
output_file = open(intermediate_outputs_dir, "w")
data_len = len(src_test['data'])
for i in range(data_len):
    sentence = ' '.join(src_test['data'][i])
    content = ' '.join(src_test['content'][i])
    attribute = ', '.join(src_test['attribute'][i])
    candidates = ', '.join(src_test['attribute_candidates'][i])
    output_file.write(sentence + " ::: " + content + " ::: " + candidates + " ::: " + attribute + "\n")

logging.info('Finished writing delete outputs')

# set some basic variables
batch_size = config['data']['batch_size']
max_length = config['data']['max_len']
src_vocab_size = len(src['tok2id'])
tgt_vocab_size = len(tgt['tok2id'])

# create a mask of length target vocab size (should be the same as source
# vocab size though)
weight_mask = torch.ones(tgt_vocab_size)
weight_mask[tgt['tok2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
if CUDA:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(config['training']['random_seed'])
    weight_mask = weight_mask.cuda()
    loss_criterion = loss_criterion.cuda()

torch.manual_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])

# Create a pretty standard seq2seq model
model = models.SeqModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    pad_id_src=src['tok2id']['<pad>'],
    pad_id_tgt=tgt['tok2id']['<pad>'],
    config=config
)

# get information about number of parameters in the model
# try to load the model from the working directory
logging.info('MODEL HAS %s params' % model.count_params())

writer = SummaryWriter(working_dir)

# we set the appropriate optimizer (in our case, adam)
if config['training']['optimizer'] == 'adam':
    lr = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['learning_rate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

def evaluate_dev(model, src_dev, tgt_dev, config):
    model.eval()
    dev_loss = evaluation.evaluate_lpp(
        model, src_dev, tgt_dev, config)
    model.train()
    return dev_loss

epoch_loss = []
start_since_last_report = time.time()
words_since_last_report = 0
losses_since_last_report = []
best_metric = 0.0
best_epoch = 0
cur_metric = 0.0  # log perplexity
num_batches = len(src['content']) / batch_size

# training the model
STEP = 0
start_epoch = 0

# run for how many ever epochs based on whether we were able to
# load model from a checkpoint
if config['training']['load_checkpoint']:
    model, start_epoch = models.attempt_load_model(
        model=model,
        checkpoint_dir=working_dir)
if CUDA:
    model = model.cuda()

for epoch in range(start_epoch, config['training']['epochs']):
    losses = []
    # we loop through each of the training examples (just the content though)
    for i in range(0, len(src['content']), batch_size):
        batch_idx = i / batch_size

        # get a mini batch with input, attribute and output
        # this allows us to use the model
        input_content, input_aux, output = data.minibatch(
            src, tgt, i, batch_size, max_length, config['model']['model_type'])

        input_lines_src, _, srclens, srcmask, _ = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output

        decoder_logit, decoder_probs = model(
            input_lines_src, input_lines_tgt, srcmask, srclens,
            input_ids_aux, auxlens, auxmask)

        optimizer.zero_grad()

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, tgt_vocab_size),
            output_lines_tgt.view(-1)
        )

        losses.append(loss.item())
        losses_since_last_report.append(loss.item())
        epoch_loss.append(loss.item())
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])

        writer.add_scalar('stats/grad_norm', norm, STEP)

        optimizer.step()

        # this section of code just allows us to print some statistics to the console
        # every couple of iterations
        if batch_idx % config['training']['batches_per_report'] == 0:
            s = float(time.time() - start_since_last_report)
            wps = (batch_size * config['training']['batches_per_report']) / s
            avg_loss = np.mean(losses_since_last_report)
            info = (epoch, batch_idx, num_batches, wps, avg_loss, cur_metric)
            writer.add_scalar('stats/WPS', wps, STEP)
            writer.add_scalar('stats/loss', avg_loss, STEP)
            logging.info('EPOCH: %s ITER: %s/%s WPS: %.2f TRAIN LOSS: %.4f DEV LOSS: %.4f' % info)
            start_since_last_report = time.time()
            words_since_last_report = 0
            losses_since_last_report = []

        # NO SAMPLING!! because weird train-vs-test data stuff would be a pain
        STEP += 1

    logging.info('EPOCH %s COMPLETE. EVALUATING...' % epoch)
    start = time.time()

    # we calculate loss the same way on the evaluation data
    dev_loss = evaluate_dev(model, src_dev, tgt_dev, config)
    cur_metric = dev_loss

    writer.add_scalar('eval/loss', dev_loss, epoch)

    # write the current checkpoint
    logging.info('NEW DEV LOSS: %s. TIME: %.2fs' % (
        cur_metric, (time.time() - start)))
    if epoch > 40:
        logging.info('CHECKPOINTING....')
        torch.save(model.state_dict(), working_dir + '/model.%s.ckpt' % epoch)

    avg_loss = np.mean(epoch_loss)
    epoch_loss = []

# let's test the model now
bleu_score, edit_distance, inputs, preds, golds, auxs = evaluation.inference_metrics(
    model, src_test, tgt_test, config)

with open(working_dir + outputs_dir + '/auxs', 'w') as f:
    f.write('\n'.join(auxs) + '\n')
with open(working_dir + outputs_dir + '/inputs', 'w') as f:
    f.write('\n'.join(inputs) + '\n')
with open(working_dir + outputs_dir + '/preds', 'w') as f:
    f.write('\n'.join(preds) + '\n')
with open(working_dir + outputs_dir + '/golds', 'w') as f:
    f.write('\n'.join(golds) + '\n')

writer.add_scalar('eval/edit_distance', edit_distance, epoch)
writer.add_scalar('eval/bleu', bleu_score, epoch)

writer.close()

logging.info('BLEU SCORE ON REFERENCE DATA: %s.' % (bleu_score))
