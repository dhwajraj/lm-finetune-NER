from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
from helpers import utilities as utils
from helpers.evaluation_helper import eval_wc

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools
import fire

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def train(
        rand_embedding=True, #help='random initialize word embedding'
        emb_file='../../embedding/glove.6B.100d.txt', #help='path to pre-trained embedding'
        train_file='../../data/eng.testa', #help='path to training file'
        dev_file='../../data/eng.testa', #help='path to development file'
        test_file='../../data/eng.testb', #help='path to test file'
        gpu=-1, #help='gpu id'
        batch_size=10, #help='batch_size'
        unk='unk', #help='unknow-token in pre-trained embedding'
        char_hidden=300, #help='dimension of char-level layers'
        word_hidden=300, #help='dimension of word-level layers'
        drop_out=0.55, #help='dropout ratio'
        epoch=200, #help='maximum epoch number'
        start_epoch=0, #help='start point of epoch'
        checkpoint='./checkpoints/', #help='checkpoint path'
        caseless=True, #help='caseless or not'
        char_dim=30, #help='dimension of char embedding'
        word_dim=100, #help='dimension of word embedding'
        char_layers=1, #help='number of char level layers'
        word_layers=1, #help='number of word level layers'
        lr=0.015, #help='initial learning rate'
        lr_decay=0.05, #help='decay ratio of learning rate'
        fine_tune=False, #help='fine tune the diction of word embedding or not'
        load_check_point='', #help='path previous checkpoint that want to be loaded'
        load_opt=True, #help='also load optimizer from the checkpoint'
        update='sgd', #help='optimizer choice' choices=['sgd', 'adam']
        momentum=0.9, #help='momentum for sgd'
        clip_grad=5.0, #help='clip grad at'
        small_crf=False, #help='use small crf instead of large crf, refer model.crf module for more details'
        mini_count=5, #help='thresholds to replace rare words with <unk>'
        lambda0=1, #help='lambda0'
        co_train=True, #help='cotrain language model'
        patience=15, #help='patience for early stop'
        high_way=True, #help='use highway layers'
        highway_layers=1, #help='number of highway layers'
        eva_matrix='fa', #help='use f1 and accuracy or accuracy alone'  choices=['a', 'fa'],
        least_iters=50, #help='at least train how many epochs before stop'
        shrink_embedding=True, #help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus'
):

    args = locals()
    if gpu >= 0:
        torch.cuda.set_device(gpu)

    print('SETTINGS:')
    for k,v in args.items():
        print(str(k),"=",str(v))

    # load corpus
    print('loading corpus')
    with codecs.open(train_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()
    with codecs.open(test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()

    dev_features, dev_labels = utils.read_corpus(dev_lines)
    test_features, test_labels = utils.read_corpus(test_lines)

    if load_check_point:
        if os.path.isfile(load_check_point):
            print("loading checkpoint: '{}'".format(load_check_point))
            checkpoint_file = torch.load(load_check_point)
            start_epoch = checkpoint_file['epoch']
            f_map = checkpoint_file['f_map']
            l_map = checkpoint_file['l_map']
            c_map = checkpoint_file['c_map']
            in_doc_words = checkpoint_file['in_doc_words']
            train_features, train_labels = utils.read_corpus(lines)
        else:
            print("no checkpoint found at: '{}'".format(load_check_point))
    else:
        print('constructing coding table')

        # converting format
        train_features, train_labels, f_map, l_map, c_map = utils.generate_corpus_char(lines, if_shrink_c_feature=True, c_thresholds=mini_count, if_shrink_w_feature=False)
        
        f_set = {v for v in f_map}
        f_map = utils.shrink_features(f_map, train_features, mini_count)

        if rand_embedding:
            print("embedding size: '{}'".format(len(f_map)))
            in_doc_words = len(f_map)
        else:
            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features), f_set)
            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features), dt_f_set)
            print("feature size: '{}'".format(len(f_map)))
            print('loading embedding')
            if fine_tune:  # which means does not do fine-tune
                f_map = {'<eof>': 0}
            f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(emb_file, ' ', f_map, dt_f_set, caseless, unk, word_dim, shrink_to_corpus=shrink_embedding)
            print("embedding size: '{}'".format(len(f_map)))

        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels))
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels), l_set)
        for label in l_set:
            if label not in l_map:
                l_map[label] = len(l_map)
    
    print('constructing dataset')
    # construct dataset
    dataset, forw_corp, back_corp = utils.construct_bucket_mean_vb_wc(train_features, train_labels, l_map, c_map, f_map, caseless)
    dev_dataset, forw_dev, back_dev = utils.construct_bucket_mean_vb_wc(dev_features, dev_labels, l_map, c_map, f_map, caseless)
    test_dataset, forw_test, back_test = utils.construct_bucket_mean_vb_wc(test_features, test_labels, l_map, c_map, f_map, caseless)
    
    dataset_loader = [torch.utils.data.DataLoader(tup, batch_size, shuffle=True, drop_last=False) for tup in dataset]
    dev_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset]

    # build model
    print('building model')
    ner_model = LM_LSTM_CRF(len(l_map), len(c_map), char_dim, char_hidden, char_layers, word_dim, word_hidden, word_layers, len(f_map), drop_out, large_CRF=small_crf, if_highway=high_way, in_doc_words=in_doc_words, highway_layers = highway_layers)

    if load_check_point:
        ner_model.load_state_dict(checkpoint_file['state_dict'])
    else:
        if not rand_embedding:
            ner_model.load_pretrained_word_embedding(embedding_tensor)
        ner_model.rand_init(init_word_embedding=rand_embedding)

    if update == 'sgd':
        optimizer = optim.SGD(ner_model.parameters(), lr=lr, momentum=momentum)
    elif update == 'adam':
        optimizer = optim.Adam(ner_model.parameters(), lr=lr)

    if load_check_point and load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    crit_lm = nn.CrossEntropyLoss()
    crit_ner = CRFLoss_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

    if gpu >= 0:
        if_cuda = True
        print('device: ' + str(gpu))
        torch.cuda.set_device(gpu)
        crit_ner.cuda()
        crit_lm.cuda()
        ner_model.cuda()
        packer = CRFRepack_WC(len(l_map), True)
    else:
        if_cuda = False
        packer = CRFRepack_WC(len(l_map), False)

    tot_length = sum(map(lambda t: len(t), dataset_loader))

    best_f1 = float('-inf')
    best_acc = float('-inf')
    track_list = list()
    start_time = time.time()
    epoch_list = range(start_epoch, start_epoch + epoch)
    patience_count = 0

    evaluator = eval_wc(packer, l_map, eva_matrix)

    for epoch_idx, start_epoch in enumerate(epoch_list):

        epoch_loss = 0
        ner_model.train()
        for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v in tqdm(
                itertools.chain.from_iterable(dataset_loader), mininterval=2,
                desc=' - Tot it %d (epoch %d)' % (tot_length, start_epoch), leave=False, file=sys.stdout):
            f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v)
            ner_model.zero_grad()
            scores = ner_model(f_f, f_p, b_f, b_p, w_f)
            loss = crit_ner(scores, tg_v, mask_v)
            epoch_loss += utils.to_scalar(loss)
            if co_train:
                cf_p = f_p[0:-1, :].contiguous()
                cb_p = b_p[1:, :].contiguous()
                cf_y = w_f[1:, :].contiguous()
                cb_y = w_f[0:-1, :].contiguous()
                cfs, _ = ner_model.word_pre_train_forward(f_f, cf_p)
                loss = loss + lambda0 * crit_lm(cfs, cf_y.view(-1))
                cbs, _ = ner_model.word_pre_train_backward(b_f, cb_p)
                loss = loss + lambda0 * crit_lm(cbs, cb_y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(ner_model.parameters(), clip_grad)
            optimizer.step()
        epoch_loss /= tot_length

        # update lr
        if update == 'sgd':
            utils.adjust_learning_rate(optimizer, lr / (1 + (start_epoch + 1) * lr_decay))

        # eval & save check_point

        if 'f' in eva_matrix:
            dev_result = evaluator.calc_score(ner_model, dev_dataset_loader)
            for label, (dev_f1, dev_pre, dev_rec, dev_acc, msg) in dev_result.items():
                print('DEV : %s : test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f | %s\n' % (label, dev_f1, dev_pre, dev_rec, dev_acc, msg))
            (dev_f1, dev_pre, dev_rec, dev_acc, msg) = dev_result['total']

            if dev_f1 > best_f1:
                patience_count = 0
                best_f1 = dev_f1

                test_result = evaluator.calc_score(ner_model, test_dataset_loader)
                for label, (test_f1, test_pre, test_rec, test_acc, msg) in test_result.items():
                    print('TEST : %s : test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f | %s\n' % (label, test_f1, test_rec, test_pre, test_acc, msg))
                (test_f1, test_rec, test_pre, test_acc, msg) = test_result['total']

                track_list.append(
                    {'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'test_f1': test_f1,
                     'test_acc': test_acc})

                print(
                    '(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f, F1 on test = %.4f, acc on test= %.4f), saving...' %
                    (epoch_loss,
                     start_epoch,
                     dev_f1,
                     dev_acc,
                     test_f1,
                     test_acc))

                try:
                    utils.save_checkpoint({
                        'epoch': start_epoch,
                        'state_dict': ner_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'l_map': l_map,
                        'c_map': c_map,
                        'in_doc_words': in_doc_words
                    }, {'track_list': track_list,
                        'args': vars(args)
                        }, checkpoint + 'cwlm_lstm_crf')
                except Exception as inst:
                    print(inst)

            else:
                patience_count += 1
                print('(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f)' %
                      (epoch_loss,
                       start_epoch,
                       dev_f1,
                       dev_acc))
                track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})

        else:

            dev_acc = evaluator.calc_score(ner_model, dev_dataset_loader)

            if dev_acc > best_acc:
                patience_count = 0
                best_acc = dev_acc
                
                test_acc = evaluator.calc_score(ner_model, test_dataset_loader)

                track_list.append(
                    {'loss': epoch_loss, 'dev_acc': dev_acc, 'test_acc': test_acc})

                print(
                    '(loss: %.4f, epoch: %d, dev acc = %.4f, acc on test= %.4f), saving...' %
                    (epoch_loss,
                     start_epoch,
                     dev_acc,
                     test_acc))

                try:
                    utils.save_checkpoint({
                        'epoch': start_epoch,
                        'state_dict': ner_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'l_map': l_map,
                        'c_map': c_map,
                        'in_doc_words': in_doc_words
                    }, {'track_list': track_list,
                        'args': vars(args)
                        }, checkpoint + 'cwlm_lstm_crf')
                except Exception as inst:
                    print(inst)

            else:
                patience_count += 1
                print('(loss: %.4f, epoch: %d, dev acc = %.4f)' %
                      (epoch_loss,
                       start_epoch,
                       dev_acc))
                track_list.append({'loss': epoch_loss, 'dev_acc': dev_acc})

        print('epoch: ' + str(start_epoch) + '\t in ' + str(epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        if patience_count >= patience and start_epoch >= least_iters:
            break

    #print best
    if 'f' in eva_matrix:
        eprint(checkpoint + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc))
    else:
        eprint(checkpoint + ' dev_acc: %.4f test_acc: %.4f\n' % (dev_acc, test_acc))

    # printing summary
    print('setting:')
    print(args)

if __name__ == "__main__":
    fire.Fire(train)