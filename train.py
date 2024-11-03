'''
Copyright 2024: Haochen Zhang
Parts adapted from: butd_detr, MVT-3DVG repositories
https://github.com/nickgkan/butd_detr
https://github.com/sega-hsj/MVT-3DVG
'''

import torch
import tqdm
import time
import torch.nn as nn
import numpy as np
from torch import optim
import os
import csv
import json
import wandb
import random

from utils import *
from data.dataloader import VLADataset
from data.referit3d_loader import ReferIt3DDataset
from arg_parser import parse_args

os.environ["WANDB__SERVICE_WAIT"] = "300"

class BaseTrainer():
    '''
    Base training infrastructure
    '''
    def __init__(self, args):
        self.data_path = args.data_path
    

    def prepare_data(self, args):
        '''
        Load dataset
        :return: Dataloaders
        '''
        print("Loading dataset...")
        train_loader = None
        # only get training data is necessary
        if not args.eval:            
            if args.dataset == "r3d":
                train_dataset = ReferIt3DDataset(args.data_path, args.points_per_obj, split=args.train_split, prune_sparse=args.prune_sparse, 
                                                 sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, 
                                                 context_size=args.context_size, include_raw_labels=args.include_raw_labels,
                                                 random_seed=args.random_seed)
                
            elif args.dataset == "vla":
                train_dataset = VLADataset(args.data_path, args.points_per_obj, split=args.train_split, prune_sparse=args.prune_sparse, 
                                        sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, context_size=args.context_size,
                                        load_false_statements=args.load_false_statements, balance_false=args.balance_false, 
                                        include_raw_labels=args.include_raw_labels, random_seed=args.random_seed)
            elif args.dataset == "both":
                vla_train_dataset = VLADataset(args.data_path, args.points_per_obj, split=args.train_split, prune_sparse=args.prune_sparse, 
                                        sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, context_size=args.context_size,
                                        load_false_statements=args.load_false_statements, balance_false=args.balance_false, 
                                        include_raw_labels=args.include_raw_labels, random_seed=args.random_seed)
                r3d_train_dataset = ReferIt3DDataset(args.data_path, args.points_per_obj, split=args.train_split, prune_sparse=args.prune_sparse, 
                                                 sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, 
                                                 context_size=args.context_size, include_raw_labels=args.include_raw_labels,
                                                 random_seed=args.random_seed)
                
                train_dataset = vla_train_dataset + r3d_train_dataset
            else:
                raise ValueError(f"Dataset {args.dataset} not supported, please choose from 'r3d' or 'vla'")

            train_loader = dataset_to_dataloader(train_dataset, args)
        
        # get dataset and dataloader
        if args.dataset == "r3d":
            test_dataset = ReferIt3DDataset(args.data_path, args.points_per_obj, split=args.test_split, prune_sparse=args.prune_sparse, 
                                            sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, 
                                            context_size=args.context_size, include_raw_labels=args.include_raw_labels, 
                                            random_seed=args.random_seed)
        elif args.dataset == "vla":
            test_dataset = VLADataset(args.data_path, args.points_per_obj, split=args.test_split, prune_sparse=args.prune_sparse, 
                                    sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, context_size=args.context_size,
                                    load_false_statements=args.load_false_statements, balance_false=args.balance_false, 
                                    include_raw_labels=args.include_raw_labels, random_seed=args.random_seed)
        elif args.dataset == "both":
            vla_test_dataset = VLADataset(args.data_path, args.points_per_obj, split=args.test_split, prune_sparse=args.prune_sparse, 
                                    sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, context_size=args.context_size,
                                    load_false_statements=args.load_false_statements, balance_false=args.balance_false, 
                                    include_raw_labels=args.include_raw_labels, random_seed=args.random_seed)
            r3d_test_dataset = ReferIt3DDataset(args.data_path, args.points_per_obj, split=args.test_split, prune_sparse=args.prune_sparse, 
                                            sparsity_thresh=args.sparsity_thresh, use_context=args.use_context, 
                                            context_size=args.context_size, include_raw_labels=args.include_raw_labels, 
                                            random_seed=args.random_seed)
            
            test_dataset = vla_test_dataset + r3d_test_dataset
        else:
            raise ValueError(f"Dataset {args.dataset} not supported, please choose from 'r3d' or 'vla'")

        test_loader = dataset_to_dataloader(test_dataset, args, False)
        print("Dataset loaded...")
        
        return train_loader, test_loader


    def load_checkpoint(self, args, model, optimizer, scheduler):
        '''
        Load pretrained model checkpoint from disk
        '''
        device = torch.device('cuda')
        print("Loading from pretrained checkpoint: '{}'".format(args.resume_path))

        checkpoint = torch.load(args.resume_path, map_location=device)
        try:
            args.start_epoch = int(checkpoint['epoch']) + 1
        except Exception:
            print("Could not load existing checkpoint")
            args.start_epoch = 0

        model.load_state_dict(checkpoint['model'], strict=False)

        # load optimizer and scheduler
        if not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            elif "lr_scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print("Successfully loaded checkpoint from epoch {}".format(checkpoint['epoch']))

        del checkpoint
        torch.cuda.empty_cache()


    def save_checkpoint(self, args, epoch, model, optimizer, scheduler, name=None):
        '''
        Save model checkpoint to disk
        '''
        state = {
            'config': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        
        # default name for checkpoint
        if name == None:
            name = f'ckpt_epoch_{epoch}.pth'

        ckpt_path = os.path.join(args.log_dir, name)
        state['save_path'] = ckpt_path
        torch.save(state, ckpt_path)
        print("Checkpoint saved in {}".format(ckpt_path))


    def load_model(self, args, max_context_len):
        return None


    def preprocess_batch(self, batch):
        '''
        Preprocess batch of data before passing to model
        :return: preprocessed batch
        '''
        return batch
    

    def get_optimizer(self, params, args):
        '''
        Get optimizer based on model params
        :return: optimizer
        '''
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        return optimizer


    def get_criterion(self, args):
        '''
        Get criterion based on model eval
        :return: criterion
        '''
        return None


    def get_scheduler(self, optimizer, args):
        '''
        Get scheduler used
        :return: scheduler
        '''
        return None


    def run_one_epoch(self, args, run, epoch, train_loader, test_loader, model, optimizer, criterion, scheduler):
        '''
        Run one training epoch
        :return: metrics
        '''

        metrics = dict()  # holding the losses/accuracies

        # Set the model in training mode
        model.train()
        np.random.seed()  # call this to change the sampling of the point-clouds
        
        for batch in tqdm.tqdm(train_loader):
            batch = self.preprocess_batch(batch)

            # Forward pass
            loss = model(batch, epoch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Get statistics and print out

        # add logging here
        metrics.update({epoch:loss})

        return metrics


    def eval_one_epoch(self, model, test_loader, criterion, args, randomize):
        '''
        Run evaluation on one epoch
        :return: metrics
        '''
        metrics = {}
        return metrics
    
    
    def train(self, args):
        '''
        Main training/eval function
        '''
        
        start_time = time.time()
                
        train_loader, test_loader = self.prepare_data(args)
        if train_loader is not None and test_loader is not None: # CHANGE
            print("Num train: {}, Num test: {}".format(len(train_loader.dataset), len(test_loader.dataset)))
        
        end_time = time.time()
        print("Dataset loaded in: ", end_time-start_time)

        if train_loader is not None:
            max_context_len = train_loader.dataset.max_context_len
        else:
            max_context_len = test_loader.dataset.max_context_len

        model = self.load_model(args, max_context_len=max_context_len)

        # get criterion (depends on model eval)
        criterion = self.get_criterion(args)

        # get optimizer
        optimizer = self.get_optimizer(model, args)

        # get scheduler
        scheduler = self.get_scheduler(optimizer, args)

        log_file = os.path.join(args.log_dir, 'results.csv')

        # check if loading from pretrained checkpoint
        if args.resume_path:
            assert os.path.isfile(args.resume_path)
            self.load_checkpoint(args, model, optimizer, scheduler)
            
        # evaluate model
        if args.eval:
            print("Starting evaluation...")
            eval_metrics = self.eval_one_epoch(model, test_loader, criterion, args, randomize=False)
            print("Evaluation completed.")
            eval_loss = eval_metrics['test_total_loss']
            eval_acc = eval_metrics['test_referential_acc']
            print("Test loss: {}, Test accuracy: {}".format(eval_loss, eval_acc))
            print(eval_metrics)
            return
    
        if not os.path.isdir(args.log_dir):
            os.mkdir(args.log_dir)
        
        # save args
        arg_file = os.path.join(args.log_dir, 'args.json')
        with open(arg_file, 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        # set up wandb logging
        run_name = args.run_name
        if run_name == None:
            # generate random run name
            min = 10**(5-1)
            max = 9*min + (min-1)
            run_id = random.randint(min, max)
            run_name = 'train_'+str(run_id)
        
        run = wandb.init(project=args.proj_name, name=run_name)
        
        with open(log_file, 'a') as csvFile:
            csvWriter = csv.writer(csvFile)

            best_test_acc = 0
            eval_acc = 0
            with tqdm.trange(args.start_epoch, args.max_epoch + 1, desc='Epochs') as tot_epochs:
                timings = dict()
                if not args.resume_path:
                    csvWriter.writerow(['Epoch', 'Train loss', 'Train acc', 'Test loss', 'Test acc'])
                
                for epoch in tot_epochs:
                    # Train
                    epoch_start = time.time()

                    train_metrics = self.run_one_epoch(args, run, epoch, train_loader, test_loader, model, optimizer, criterion, scheduler)

                    epoch_end = time.time()
                    timings['train'] = (epoch_end - epoch_start) / 60
                    train_loss = train_metrics['train_total_loss']
                    train_acc = train_metrics['train_referential_acc']

                    print("Epoch: {}, Train time: {}, Train loss: {}, Train accuracy: {}".format(epoch, epoch_end-epoch_start, train_loss, train_acc))

                    self.save_checkpoint(args, epoch, model, optimizer, scheduler, 'last.pth')

                    run.log({"epoch":epoch}, commit=False)

                    # evaluate
                    if (epoch % args.test_freq) == 0:
                        tic = time.time()
                        print("Test evaluation.......")
                        eval_metrics = self.eval_one_epoch(model, test_loader, criterion, args, randomize=False)
                        self.save_checkpoint(args, epoch, model, optimizer, scheduler)

                        toc = time.time()
                        timings['test'] = (toc - tic) / 60

                        eval_loss = eval_metrics['test_total_loss']
                        eval_acc = eval_metrics['test_referential_acc']
                        print("Test loss: {}, Test accuracy: {}".format(eval_loss, eval_acc))

                        # save best model
                        if eval_acc > best_test_acc:
                            print('Test accuracy improved at epoch {}'.format(epoch))
                            best_test_acc = eval_acc
                            best_test_epoch = epoch

                            self.save_checkpoint(args, epoch, model, optimizer, scheduler, 'best_epoch.pth')

                        csvWriter.writerow([epoch, train_loss, train_acc, eval_loss, eval_acc])

                        run.log({"train_acc_epoch":train_acc, "train_loss_epoch":train_loss, "eval_acc_epoch":eval_acc, "eval_loss_epoch":eval_loss})

                    if (epoch % args.save_freq) == 0:
                        self.save_checkpoint(args, epoch, model, optimizer, scheduler, f"epoch_{epoch}.pth")

                    csvFile.flush()
                    tot_epochs.refresh()

        print('Training completed')
