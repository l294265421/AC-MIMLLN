# -*- coding: utf-8 -*-

import argparse
import sys
import random
import copy

import torch
import numpy

from nlp_tasks.utils import argument_utils
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import acd_and_sc_train_templates_pytorch \
    as templates

parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='SemEval-2014-Task-4-REST-DevSplits', type=str)
# parser.add_argument('--hard_dataset', help='hard dataset name', default='', type=str)
parser.add_argument('--hard_test', help='extract hard test sets from test sets', default=True,
                    type=argument_utils.my_bool)
parser.add_argument('--task_name', help='task name', default='acd_and_sc', type=str)
parser.add_argument('--data_type', help='different dataset readers correspond to different data types', default='common', type=str)
parser.add_argument('--model_name', help='model name', default='AsMil', type=str)
parser.add_argument('--timestamp', help='timestamp', default=int(1571400646), type=int)
parser.add_argument('--train', help='if train a new model', default=True, type=argument_utils.my_bool)
parser.add_argument('--evaluate', help='if evaluate the new model', default=True, type=argument_utils.my_bool)
parser.add_argument('--evaluation_on_instance_level', default=False, type=argument_utils.my_bool)
parser.add_argument('--error_analysis', help='error analysis', default=False, type=argument_utils.my_bool)
parser.add_argument('--epochs', help='epochs', default=100, type=int)
parser.add_argument('--batch_size', help='batch_size', default=64, type=int)
parser.add_argument('--patience', help='patience', default=10, type=int)
parser.add_argument('--visualize_attention', help='visualize attention', default=False, type=argument_utils.my_bool)
parser.add_argument('--embedding_filepath', help='embedding filepath',
                    default='', type=str)
parser.add_argument('--embed_size', help='embedding dim', default=300, type=int)
parser.add_argument('--seed', default=776, type=int)
parser.add_argument('--repeat', default=1, type=str)
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--lstm_layer_category_classifier', default=False, type=argument_utils.my_bool)
parser.add_argument('--position', default=False, type=argument_utils.my_bool)
parser.add_argument('--position_embeddings_dim', help='position embeddings dim', default=64, type=int)
parser.add_argument('--only_acd', default=False, type=argument_utils.my_bool)
parser.add_argument('--only_sc', default=False, type=argument_utils.my_bool)
parser.add_argument('--debug', default=False, type=argument_utils.my_bool)
parser.add_argument('--early_stopping_by_batch', default=False, type=argument_utils.my_bool)
parser.add_argument('--share_sentiment_classifier', help='share_sentiment_classifier', default=True,
                    type=argument_utils.my_bool)
parser.add_argument('--attention_lamda', default=1, type=float)
parser.add_argument('--sparse_reg', default=False, type=argument_utils.my_bool)
parser.add_argument('--orthogonal_reg', default=False, type=argument_utils.my_bool)
parser.add_argument('--early_stopping', default=True, type=argument_utils.my_bool)


# varients
parser.add_argument('--bert', default=False, type=argument_utils.my_bool)
parser.add_argument('--only_bert', default=False, type=argument_utils.my_bool)
parser.add_argument('--concat_cls_vector', default=True, type=argument_utils.my_bool)
parser.add_argument('--concat_cls_vector_mode', default='sum', type=str, help='',
                    choices=['sum', 'average'])
parser.add_argument('--pair', default=True, type=argument_utils.my_bool)
parser.add_argument('--fixed', default=False, type=argument_utils.my_bool)
parser.add_argument('--max_len', help='max length', default=100, type=int)
parser.add_argument('--lstm_layer_num_in_bert', default=0, type=int)
parser.add_argument('--dropout_in_bert', default=0.5, type=float)
parser.add_argument('--learning_rate_in_bert', default=2e-5, type=float)
parser.add_argument('--l2_in_bert', default=0.00001, type=float)
parser.add_argument('--acd_init_weight', default=1, type=float)
parser.add_argument('--bilstm_hidden_size_in_bert', default=0, type=float)
parser.add_argument('--dropout_after_cls', default=False, type=argument_utils.my_bool)


parser.add_argument('--mil', default=False, type=argument_utils.my_bool)
parser.add_argument('--mil_softmax', default=False, type=argument_utils.my_bool)
parser.add_argument('--acd_sc_mode', default='multi-multi', type=str, help='the acd task mode and the sc task '
                                                                           'mode in joint model',
                    choices=['multi-single', 'multi-multi'])
parser.add_argument('--joint_type', default='joint', type=str)
parser.add_argument('--acd_warmup', help='acd warmup', default=False, type=argument_utils.my_bool)
parser.add_argument('--acd_warmup_epochs', help='acd warmup epochs', default=10, type=int)
parser.add_argument('--acd_warmup_patience', help='acd warmup patience', default=4, type=int)
parser.add_argument('--pipeline', default=False, type=argument_utils.my_bool)
parser.add_argument('--lstm_or_fc_after_embedding_layer', default='lstm', type=str)
parser.add_argument('--lstm_layer_num_in_lstm', default=3, type=int)
parser.add_argument('--sentence_encoder_for_sentiment', default='bilstm', type=str, choices=['bilstm', 'cnn'])
parser.add_argument('--bert_file_path', help='bert_file_path',
                    default=r'', type=str)
parser.add_argument('--bert_vocab_file_path', help='bert_vocab_file_path',
                    default=r'', type=str)

parser.add_argument('--savefig_dir', help='dir to save pictures of visualizing model',
                    default='', type=str)
parser.add_argument('--frozen_all_acsc_parameter_while_pretrain_acd', default=False, type=argument_utils.my_bool)


args = parser.parse_args()

if args.joint_type == 'pipeline':
    args.acd_warmup = True
    args.pipeline = True
elif args.joint_type == 'warmup':
    args.acd_warmup = True
    args.pipeline = False
else:
    args.acd_warmup = False
    args.pipeline = False

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)
gpu_ids = args.gpu_id.split(',')
if len(gpu_ids) == 1:
    args.gpu_id = -1 if int(gpu_ids[0]) == -1 else 0
else:
    args.gpu_id = list(range(len(gpu_ids)))


configuration = args.__dict__

if configuration['seed'] is not None:
    random.seed(configuration['seed'])
    numpy.random.seed(configuration['seed'])
    torch.manual_seed(configuration['seed'])
    torch.cuda.manual_seed(configuration['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model_name_complete_prefix = 'model_name.{model_name}'.format_map(configuration)

configuration_for_this_repeat = copy.deepcopy(configuration)
configuration_for_this_repeat['model_name_complete'] = '%s.repeat.%s' % (model_name_complete_prefix, args.repeat)

if configuration_for_this_repeat['bert']:
    if configuration_for_this_repeat['pair']:
        template = templates.AsMilBert(configuration_for_this_repeat)
    else:
        template = templates.AsMilBertSingle(configuration_for_this_repeat)
else:
    template = templates.AsMil(configuration_for_this_repeat)

if configuration_for_this_repeat['train']:
    template.train()
if configuration_for_this_repeat['evaluate']:
    template.evaluate()
if configuration_for_this_repeat['error_analysis']:
    result = template.error_analysis()
if configuration_for_this_repeat['evaluation_on_instance_level']:
    template.evaluation_on_instance_level()
