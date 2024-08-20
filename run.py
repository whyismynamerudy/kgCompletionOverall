import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    # parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    # parser.add_argument('--regions', type=int, nargs='+', default=None, 
    #                     help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_arguments('--weight_decay', default=0.00001, type=float, help='weight decay')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=4, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change in MRR to qualify as an improvement')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs to train')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def plot_mrr(mrr_values, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(mrr_values)
    plt.title('MRR over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MRR')
    plt.grid(True)
    
    plt.savefig(os.path.join(save_path, 'mrr_plot.png'))
    plt.close()


def save_model(model, optimizer, save_variable_list, args):

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((
                entity2id[h],
                relation2id[r],
                entity2id[t]
            ))
    return triples

def set_logger(args):
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be chosen.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be chosen.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    # if args.countries:
    #     regions = list()
    #     with open(os.path.join(args.data_path, 'regions.list')) as fin:
    #         for line in fin:
    #             region = line.strip()
    #             regions.append(entity2id[region])
    #     args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    all_true_triples = train_triples + valid_triples + test_triples

    print(args)
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # set both head and tial prediction
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # set steps per epoch to get proper validation per epoch
        steps_per_epoch = len(train_triples) // args.batch_size
        
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate,
            weight_decay=args.weight_decay
        )

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_epochs * steps_per_epoch // 2

    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []

        mrr_values = []

        best_mrr = -1
        best_epoch = 0
        patience_counter = 0

        for epoch in range(args.max_epochs):
            for step in range(steps_per_epoch):
                log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
                training_logs.append(log)

                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                    log_metrics(f'Training average (Epoch {epoch}, Step {step})', step, metrics)
                    training_logs = []

            # validate at the end of each epoch
            logging.info(f"Evaluting on Valid Dataset... (Epoch {epoch})")
            metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
            log_metrics(f"Valid (Epoch {epoch})", epoch * steps_per_epoch, metrics)

            mrr_values.append(metrics['MRR'])

            # check for improvement
            current_mrr = metrics['MRR']
            if current_mrr > best_mrr + args.min_delta:
                best_mrr = current_mrr
                best_epoch = epoch
                patience_counter = 0
                logging.info(f'New best MRR: {best_mrr:.6f} at epoch {best_epoch}')

                save_variable_list = {
                    'epoch': epoch,
                    'step': epoch * steps_per_epoch, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
            else:
                patience_counter += 1
                logging.info(f'No improvement for {patience_counter} epochs')

            # check for early stoppage
            if patience_counter >= args.patience:
                logging.info(f'Early stopping triggered. Best MRR: {best_mrr:.6f} at epoch {best_epoch}')
                break

            # lr decay
            if epoch >= warm_up_steps // steps_per_epoch:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at epoch %d' % (current_learning_rate, epoch))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
        
        #Training Loop
        # for step in range(init_step, args.max_steps):
            
        #     log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
        #     training_logs.append(log)
            
        #     if step >= warm_up_steps:
        #         current_learning_rate = current_learning_rate / 10
        #         logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
        #         optimizer = torch.optim.Adam(
        #             filter(lambda p: p.requires_grad, kge_model.parameters()), 
        #             lr=current_learning_rate
        #         )
        #         warm_up_steps = warm_up_steps * 3
            
        #     if step % args.save_checkpoint_steps == 0:
        #         save_variable_list = {
        #             'step': step, 
        #             'current_learning_rate': current_learning_rate,
        #             'warm_up_steps': warm_up_steps
        #         }
        #         save_model(kge_model, optimizer, save_variable_list, args)
                
        #     if step % args.log_steps == 0:
        #         metrics = {}
        #         for metric in training_logs[0].keys():
        #             metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
        #         log_metrics('Training average', step, metrics)
        #         training_logs = []
                
        #     if args.do_valid and step % args.valid_steps == 0:
        #         logging.info('Evaluating on Valid Dataset...')
        #         metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        #         log_metrics('Valid', step, metrics)
        
        plot_mrr(mrr_values, args.save_path)
        
        save_variable_list = {
            'epoch': epoch,
            'step': (epoch + 1) * steps_per_epoch, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Train', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())