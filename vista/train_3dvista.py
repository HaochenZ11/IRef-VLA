"""
Copyright 2024: Nader Zantout. 
References functions and model architecture from https://github.com/3d-vista/3D-VisTA. 
"""

import argparse
import time
import numpy as np
import yaml
import os
import json
import wandb
import torch.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim
from tqdm import tqdm

from train import BaseTrainer
from vista.pipeline.registry import registry
from vista.pipeline.pipeline_mixin import *
from vista.optimization.loss_function import get_refer_loss_v1, get_pretrain_loss_v1
from data.dataloader import VLADataset, dataset_to_dataloader
from utils import *
from arg_parser import parse_args

from vista.vla.dataset_wrappers import VLADatasetWrapper, ReferIt3DDatasetWrapper
from vista.vla.model import Grounding3DVistaModel


class VistaTrainer(
    BaseTrainer, 
    NormalDataloaderMixin, 
    ModelOptimizationMixin, 
    ModelEvaluationMixin, 
    ModelMetricMixin, 
    ModelLossMixin
    ):
    def __init__(self, args):
        super().__init__(args)
        self.device = torch.device('cuda')

        self.use_r3d = args.use_r3d
        self.eval_task = args.eval
        self.use_context = args.use_context
        self.context_size = args.context_size
        self.loss_type = args.loss_type
        self.include_binary_classifier = False

        # Parameters
        self.batch_size = args.cfg['batch_size']
        self.learning_rate = args.cfg['learning_rate']
        self.grad_norm = args.cfg['grad_norm']
        self.epochs = args.cfg['epochs']
        self.warmup_steps = args.cfg['warmup_steps']


    def load_model(self, args):
        model = Grounding3DVistaModel(args.cfg).to(self.device)
        return model
    
    def load_checkpoint(self, args, model: Grounding3DVistaModel, optimizer, scheduler):

        print("Loading from pretrained checkpoint: '{}'".format(args.resume_path))

        state_dict = torch.load(args.resume_path, map_location=self.device)
        try:
            args.start_epoch = int(state_dict['epoch']) + 1
        except Exception:
            print("Could not find epoch number in existing checkpoint")
            args.start_epoch = 0
        
        if 'model' in state_dict:
            model_dict = state_dict['model']
        else:
            model_dict = state_dict
        model.lang_encoder.load_state_dict(model_dict['lang_encoder'])
        model.point_encoder.load_state_dict(model_dict['point_encoder'])
        model.unified_encoder.load_state_dict(model_dict['unified_encoder'])
        model.ground_head.load_state_dict(model_dict['ground_head'])
        model.pretrain_head.load_state_dict(model_dict['pretrain_head'])

        # load optimizer and scheduler
        if not args.eval:
            try:
                optimizer.load_state_dict(state_dict['optimizer'])
                scheduler.load_state_dict(state_dict['scheduler'])
            except KeyError as e:
                print("Optimizer and scheduler state dicts not found, skipping loading.")

            print("Successfully loaded checkpoint")

        del state_dict
        torch.cuda.empty_cache()

    def save_checkpoint(self, args, epoch, model: Grounding3DVistaModel, optimizer, scheduler, name=None):
        '''
        Save model checkpoint to disk
        '''
        state = {
            'config': args,
            'model': {
                'lang_encoder': model.lang_encoder.state_dict(),
                'point_encoder': model.point_encoder.state_dict(),
                'unified_encoder': model.unified_encoder.state_dict(),
                'ground_head': model.ground_head.state_dict(),
                'pretrain_head': model.pretrain_head.state_dict(),
            },
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        
        # default name for checkpoint
        if name == None:
            name = f'epoch_{epoch}.pth'

        ckpt_path = os.path.join(args.log_dir, 'weights', name)
        state['save_path'] = ckpt_path
        torch.save(state, ckpt_path)
        print("Checkpoint saved in {}".format(ckpt_path))
    
    def prepare_data(self, args):
        '''
        Load dataset
        :return: Dataloaders
        '''
        print("Loading train dataset...")
        # get datasets
        wrapper_args = args.cfg['refer_dataset']['args']

        if not self.eval_task:
            if not self.use_r3d:
                train_dataset = VLADatasetWrapper(
                    tokenizer=wrapper_args['tokenizer'],
                    max_seq_length=wrapper_args['txt_seq_length'],
                    data_path=args.data_path, 
                    num_obj_pts=args.points_per_obj, 
                    split=args.train_split, 
                    prune_sparse=args.prune_sparse,
                    context_size=args.context_size,
                    use_context=args.use_context,
                    save_scene_pcs=False,
                    save_region_pcs=False
                    )
            else:
                train_dataset = ReferIt3DDatasetWrapper(
                    tokenizer=wrapper_args['tokenizer'],
                    max_seq_length=wrapper_args['txt_seq_length'],
                    data_path=args.data_path, 
                    num_obj_pts=args.points_per_obj, 
                    split=args.train_split, 
                    prune_sparse=args.prune_sparse,
                    context_size=args.context_size,
                    use_context=args.use_context,
                    save_scene_pcs=False,
                    save_region_pcs=False
                    )    
        
        print("Loading test dataset...")

        if not self.use_r3d:
            test_dataset = VLADatasetWrapper(
                tokenizer=wrapper_args['tokenizer'],
                max_seq_length=wrapper_args['txt_seq_length'],
                data_path=args.data_path, 
                num_obj_pts=args.points_per_obj, 
                split=args.test_split, 
                prune_sparse=args.prune_sparse,
                context_size=args.context_size,
                use_context=args.use_context,
                save_scene_pcs=False,
                save_region_pcs=False
                )
        else:
            test_dataset = ReferIt3DDatasetWrapper(
                tokenizer=wrapper_args['tokenizer'],
                max_seq_length=wrapper_args['txt_seq_length'],
                data_path=args.data_path, 
                num_obj_pts=args.points_per_obj, 
                split=args.test_split, 
                prune_sparse=args.prune_sparse,
                context_size=args.context_size,
                use_context=args.use_context,
                save_scene_pcs=False,
                save_region_pcs=False
                )
            
        print("Datasets loaded.")

        # get dataloaders
        if not self.eval_task:
            train_loader = dataset_to_dataloader(train_dataset, args)
        else:
            train_loader = None
        test_loader = dataset_to_dataloader(test_dataset, args, False)

        return train_loader, test_loader
    
    def preprocess_batch(self, batch: dict):
        # Move data to gpu
        for k in batch.keys():
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(self.device)
        
        return batch
    
    def get_loss(self, data_dict):
        if self.loss_type == 'refer':
            total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss = get_refer_loss_v1(
                data_dict['txt_cls_logits'], 
                data_dict['obj_cls_post_logits'], 
                data_dict['obj_cls_pre_logits'], 
                data_dict['obj_cls_raw_logits'], 
                data_dict['og3d_logits'], 
                data_dict['tgt_object_label'], 
                data_dict['tgt_object_id'], 
                data_dict['obj_labels'], 
                data_dict['obj_masks']
                )

            if self.include_binary_classifier:
                true_inds = data_dict['real']
            else:
                true_inds = torch.tensor([True]*len(data_dict['og3d_logits']))

            # total_loss *= true_inds

            # data_dict['object_existence_loss'] = F.binary_cross_entropy_with_logits(data_dict['object_existence_logit'], data_dict['real'])

            data_dict['total_loss'] = total_loss
            data_dict['og3d_loss'] = og3d_loss
            data_dict['txt_cls_loss'] = txt_cls_loss
            data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
            data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
            data_dict['obj_cls_post_loss'] = obj_cls_post_loss

            return data_dict
        elif self.loss_type == 'pretrain':
            (
                total_loss, 
                lm_cls_loss, 
                match_loss, 
                obj_cls_raw_loss, 
                obj_cls_pre_loss, 
                obj_cls_post_loss, 
                obj_cls_pre_loss_mask, 
                obj_cls_pre_loss_unmask, 
                obj_cls_post_loss_mask, 
                obj_cls_post_loss_unmask
                ) = get_pretrain_loss_v1()


    def get_optimizer(self, model: Grounding3DVistaModel, args):
        '''
        Get optimizer based on model params
        :return: optimizer
        '''
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += self.no_decay_param_group(model.lang_encoder.named_parameters(), self.learning_rate * args.cfg['lang_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(model.point_encoder.named_parameters(), self.learning_rate * args.cfg['point_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(model.unified_encoder.named_parameters(), self.learning_rate * args.cfg['unified_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(model.ground_head.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(model.pretrain_head.named_parameters(), self.learning_rate)
        
        # build optimizer
        optimizer = optim.AdamW(optimizer_grouped_parameters, betas=[args.cfg['beta1'], args.cfg['beta2']])
        self.parameters = []
        for p in optimizer_grouped_parameters:
            self.parameters.extend(p['params'])

        return optimizer


    def get_criterion(self, args):
        '''
        Get criterion based on model eval
        :return: criterion
        '''
        return None


    def get_scheduler(self, optimizer, train_loader, args):
        '''
        Get scheduler used
        :return: scheduler
        '''
        if train_loader is None:
            return None
        # build scheduler
        total_steps = self.epochs * len(train_loader)
        self.total_steps = total_steps
        print("total_steps {}".format(total_steps))
        lambda_warmup_cosine = lambda step: self.warmup_cosine(step, self.warmup_steps, total_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_warmup_cosine)
        return scheduler


    def get_metrics(self, data_dict):
        data_dict['og_acc'] = (torch.argmax(data_dict['og3d_logits'], dim=1) == data_dict['tgt_object_id'].squeeze(1)).sum().item() / float(len(data_dict['tgt_object_id']))
        data_dict['count'] = len(data_dict['tgt_object_id'])
        # get other
        data_dict['txt_acc'] = torch.sum(torch.argmax(data_dict['txt_cls_logits'], dim=1) == data_dict["tgt_object_label"].squeeze(1)).item() / float(len(data_dict['tgt_object_label']))
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)

        data_dict['target_metric'] = data_dict['og_acc']
        return data_dict
    

    def record_train_step(self, data_dict, scheduler: optim.lr_scheduler.LambdaLR, step):
        log_dict = {
            # basic info
            'step': step,
            'lr': scheduler.get_last_lr()[0],
            'grad_norm': data_dict['grad_norm'].item(),
            # shared loss
            'total_loss': data_dict['total_loss'].item(),
            'obj_cls_raw_loss': data_dict['obj_cls_raw_loss'].item(),
            'obj_cls_pre_loss': data_dict['obj_cls_pre_loss'].item(),
            'obj_cls_post_loss': data_dict['obj_cls_post_loss'].item(),
            # shared acc
            'obj_cls_raw_acc': data_dict['obj_cls_raw_acc'],
            'obj_cls_pre_acc': data_dict['obj_cls_pre_acc'],
            'obj_cls_post_acc': data_dict['obj_cls_post_acc'],

            # loss
            'og3d_loss': data_dict['og3d_loss'].item(),
            'txt_cls_loss': data_dict['txt_cls_loss'].item(),
            # acc
            'og_acc': data_dict['og_acc'],
            'txt_acc': data_dict['txt_acc'],
        }
        for k in list(log_dict.keys()):
            log_dict['train/' + k] = log_dict.pop(k)
        wandb.log(log_dict, step=step)
            
    def record_eval_step(self, train_loader, eval_dict, epoch):
        for key in eval_dict.keys():
            if self.eval_task:
                print('test_' + key, eval_dict[key])
            else:
                print('test_' + key, eval_dict[key])
                wandb.log({'test/' + key: eval_dict[key]}, step = (epoch + 1) * len(train_loader))

    def record_eval_batch_step(self, eval_dict, step):
        for key in eval_dict.keys():
            wandb.log({'test_batch/' + key: eval_dict[key]}, step=step)

    def run_one_epoch(
            self, 
            args, 
            epoch, 
            train_loader,
            test_loader,
            model: Grounding3DVistaModel, 
            optimizer: optim.AdamW, 
            criterion, 
            scheduler: optim.lr_scheduler.LambdaLR
            ):

        # Set the model in training mode
        model.train()
        np.random.seed()  # call this to change the sampling of the point-clouds

        val_freq = 100
        num_test_batches = 10
        test_data_iter = iter(test_loader)

        for i, data_dict in enumerate(tqdm(train_loader)):
            self.preprocess_batch(data_dict)
            # Forward pass
            data_dict = model(data_dict)

            data_dict = self.get_loss(data_dict)

            data_dict = self.get_metrics(data_dict)

            loss = data_dict['total_loss']
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_norm)
            data_dict['grad_norm'] = grad_norm
            # Backward
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Get statistics and print out
            step = epoch * len(train_loader) + i 
            self.record_train_step(data_dict, scheduler, step)

            # Evaluate batch

            if i%val_freq == 0:
                model.eval()
                eval_dict = {
                    'target_metric': [], 
                    'og_acc': [], 
                    'txt_acc': [], 
                    'obj_cls_raw_acc': [], 
                    'obj_cls_pre_acc': [], 
                    'obj_cls_post_acc': [],
                    'total_loss': []
                    }
                total_count = 0

                for i in range(num_test_batches):
                    try:
                        test_data_dict = next(test_data_iter)
                    except StopIteration:
                        print("Looping again over test batches...")
                        test_data_iter = iter(test_loader)
                        test_data_dict = next(test_data_iter)

                    self.preprocess_batch(test_data_dict)
                    # forward
                    test_data_dict = model(test_data_dict)

                    test_data_dict = self.get_loss(test_data_dict)
                    # get metrics
                    test_data_dict = self.get_metrics(test_data_dict)

                    # print(data_dict['og_acc'])
                    # get count
                    count = test_data_dict['obj_fts'].shape[0]
                    total_count += count
                    for key in eval_dict.keys():
                        eval_dict[key].append(float(test_data_dict[key]) * count)
                    
                # record
                for k, v in eval_dict.items():
                    eval_dict[k] = np.sum(v) / total_count
                self.record_eval_batch_step(eval_dict, step)

                model.train()

    def eval_one_epoch(self, epoch, model: Grounding3DVistaModel, train_loader, test_loader, criterion, args, randomize):
        model.eval()
        eval_dict = {
            'target_metric': [], 
            'og_acc': [], 
            'txt_acc': [], 
            'obj_cls_raw_acc': [], 
            'obj_cls_pre_acc': [], 
            'obj_cls_post_acc': [],
            'total_loss': []
            }
         # run
        total_count = 0
        if self.eval_task:
            eval_results = []
        for i, data_dict in enumerate(tqdm(test_loader)):
            self.preprocess_batch(data_dict)
            # forward
            data_dict = model(data_dict)

            data_dict = self.get_loss(data_dict)
            # get metrics
            data_dict = self.get_metrics(data_dict)

            # print(data_dict['og_acc'])
            # get count
            count = data_dict['obj_fts'].shape[0]
            total_count += count


            failure = torch.argmax(data_dict['og3d_logits'], dim=1) != data_dict['tgt_object_id'].squeeze(1)

            # print([u for u, f in zip(data_dict['utterance'], failure) if f], torch.argmax(data_dict['og3d_logits'], dim=1)[failure])
            # save object info
            if False:
                og3d_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
                item_ids = data_dict['data_idx']
                for i in range(len(item_ids)):
                    eval_results.append({
                        "scene_id": item_ids[i],
                        "bbox": data_dict['obj_boxes'][i][og3d_pred[i]].cpu().numpy().tolist(), 
                        "correct": og3d_pred[i].item() == data_dict['tgt_object_id'][i].item()
                    })
            #  save eval dict
            if i%100 == 0:
                print("Accuracy:", np.sum(eval_dict['target_metric']) / total_count)
            for key in eval_dict.keys():
                eval_dict[key].append(float(data_dict[key]) * count)
        # record
        for k, v in eval_dict.items():
            eval_dict[k] = np.sum(v) / total_count
        self.record_eval_step(train_loader, eval_dict, epoch)
        # save results
        if False:
            with open('referit3d_result.json', 'w') as fp:
                json.dump(eval_results, fp)
        return eval_dict

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
        
        model = self.load_model(args)

        # get criterion (depends on model eval)
        criterion = self.get_criterion(args)

        # get optimizer
        optimizer = self.get_optimizer(model, args)

        # get scheduler
        scheduler = self.get_scheduler(optimizer, train_loader, args)

        # check if loading from pretrained checkpoint
        if args.resume_path:
            assert os.path.isfile(args.resume_path)
            self.load_checkpoint(args, model, optimizer, scheduler)
            
        # evaluate model
        if self.eval_task:
            print("Starting evaluation...")
            eval_metrics = self.eval_one_epoch(0, model, train_loader, test_loader, criterion, args, randomize=False)
            print("Evaluation completed.")
            eval_loss = eval_metrics['total_loss']
            eval_acc = eval_metrics['target_metric']
            print("Test loss: {}, Test accuracy: {}".format(eval_loss, eval_acc))
            return
    
        if not os.path.isdir(args.log_dir):
            os.mkdir(args.log_dir)
        
        # save args
        arg_file = os.path.join(args.log_dir, 'args.json')
        with open(arg_file, 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        

        best_test_acc = 0
        eval_acc = 0
        with tqdm(range(args.start_epoch, args.max_epoch + 1), desc='Epochs') as tot_epochs:
            timings = dict()
            
            for epoch in tot_epochs:
                # Train
                self.run_one_epoch(args, epoch, train_loader, test_loader, model, optimizer, criterion, scheduler)

                self.save_checkpoint(args, epoch, model, optimizer, scheduler, 'last.pth')

                # Evaluate
                if (epoch % args.val_freq) == 0:
                    tic = time.time()
                    print("Test evaluation.......")
                    eval_metrics = self.eval_one_epoch(epoch, model, train_loader, test_loader, criterion, args, randomize=False)
                    self.save_checkpoint(args, epoch, model, optimizer, scheduler)

                    toc = time.time()
                    timings['test'] = (toc - tic) / 60

                    eval_loss = eval_metrics['total_loss']
                    eval_acc = eval_metrics['target_metric']
                    print("Test loss: {}, Test accuracy: {}".format(eval_loss, eval_acc))

                    # save best model
                    if eval_acc > best_test_acc:
                        print('Test accuracy improved at epoch {}'.format(epoch))
                        best_test_acc = eval_acc
                        best_test_epoch = epoch

                        self.save_checkpoint(args, epoch, model, optimizer, scheduler, 'best.pth')

                tot_epochs.refresh()

        print('Training completed.')

    def eval(self, args):

        train_loader, test_loader = self.prepare_data(args)

        model = self.load_model(args)

        self.load_checkpoint(args, model, None, None)

        accuracy = self.eval_one_epoch(model, train_loader, test_loader, None, args, None)

        print(accuracy)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    with open('vista/project/vista/vla_config.yml', "r") as f:
        try:
            yml_file = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(exc)
            
    args.cfg = yml_file['pipeline']

    args.log_dir = os.path.join(args.log_dir, args.proj_name, args.run_name)


    if not os.path.exists(os.path.join(args.log_dir, 'weights')):
        os.makedirs(os.path.join(args.log_dir, 'weights'))

    wandb.init(
        project=args.proj_name, 
        name=args.run_name, 
        group=args.run_name.split('-')[0], 
        dir=args.log_dir)

    trainer = VistaTrainer(args)

    trainer.train(args)

    wandb.finish()
