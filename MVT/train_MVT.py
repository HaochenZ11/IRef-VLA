'''
References functions from MVT repo: https://github.com/sega-hsj/MVT-3DVG/tree/f680e274488c46b2bbc53e562c19e5f054576300
'''
import torch
import tqdm
import time
import torch.nn as nn
import numpy as np
from torch import optim
import os
from transformers import BertTokenizer, BertModel
from train import BaseTrainer
import json
import wandb 
import argparse
import math

from referit3d_net import ReferIt3DNet_transformer
from utils import *
from arg_parser import parse_args

# import baseline models here
from MVT import *

# TOTAL_CLASSES = 524
TOTAL_CLASSES = 607

class AverageMeter(object):
    '''
    Computes and stores the average and current value of evaluation metrics
    '''

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MVTTrainer(BaseTrainer):
    def __init__(self, args):
        self.device = torch.device('cuda')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_idx = 900
        self.args = args

    
    def __initalize_average_meters(self, split):
        avg_meters = {}
        avg_meters[f'{split}_total_loss'] = AverageMeter()
        avg_meters[f'{split}_referential_acc'] = AverageMeter()
        avg_meters[f'{split}_object_cls_acc'] = AverageMeter()
        avg_meters[f'{split}_txt_cls_acc'] = AverageMeter()
        if args.include_binary_classifier:
            avg_meters[f'{split}_exist_acc'] = AverageMeter()
            avg_meters[f'{split}_true_pos'] = AverageMeter()
            avg_meters[f'{split}_false_pos'] = AverageMeter()
            avg_meters[f'{split}_true_neg'] = AverageMeter()
            avg_meters[f'{split}_false_neg'] = AverageMeter()

        return avg_meters


    def __update_average_meters(self, avg_meters, batch, res, split):
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        avg_meters[f"{split}_total_loss"].update(res["loss"].mean().item(), batch_size)

        class_predictions = torch.argmax(res['logits'], dim=1)
        class_guessed_correctly = torch.mean((class_predictions == target).double()).item()
        avg_meters[f"{split}_referential_acc"].update(class_guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = self._cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=self.pad_idx)
            avg_meters[f'{split}_object_cls_acc'].update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double()).item()
            avg_meters[f'{split}_txt_cls_acc'].update(cls_b_acc, batch_size)
        
        if args.include_binary_classifier:
            existence_predictions = torch.argmax(res['existence_logits'], dim=1)
            existence_guessed_correctly = torch.mean((existence_predictions == batch['real'].cuda()).double()).item()
            avg_meters[f'{split}_exist_acc'].update(existence_guessed_correctly, batch_size)

            true_pos = (torch.sum((existence_predictions == 1) & (batch['real'].cuda() == 1)) / torch.sum(batch['real'].cuda() == 1)).item()
            false_pos = (torch.sum((existence_predictions == 1) & (batch['real'].cuda() == 0)) / torch.sum(batch['real'].cuda() == 0)).item()
            true_neg = (torch.sum((existence_predictions == 0) & (batch['real'].cuda() == 0)) / torch.sum(batch['real'].cuda() == 0)).item()
            false_neg = (torch.sum((existence_predictions == 0) & (batch['real'].cuda() == 1)) / torch.sum(batch['real'].cuda() == 1)).item()

            if not math.isnan(true_pos):
                avg_meters[f'{split}_true_pos'].update(true_pos, batch_size)
            if not math.isnan(false_pos):
                avg_meters[f'{split}_true_pos'].update(false_pos, batch_size)
            if not math.isnan(true_neg):
                avg_meters[f'{split}_true_neg'].update(true_neg, batch_size)
            if not math.isnan(false_neg):
                avg_meters[f'{split}_false_neg'].update(false_neg, batch_size)
    

    def __log_metrics_to_wandb(self, avg_meters):
        logs = {}
        for metric, avg_meter in avg_meters.items():
            logs[metric] = avg_meter.avg
                
        wandb.log(logs)


    # from MVT utils
    def set_gpu_to_zero_position(self, real_gpu_loc):
        '''
        Sets gpu to zero position for PointNet functionality
        '''
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_loc)


    def _make_batch_keys(self, args, extras=None):
        '''
        Determines what data info is used depending on input args
        '''
        batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
        if extras is not None:
            batch_keys += extras

        if args.obj_cls_alpha > 0:
            batch_keys.append('class_labels')

        if args.lang_cls_alpha > 0:
            batch_keys.append('target_class')

        return batch_keys


    def _remap_classes(self, all_classes):
        '''
        Remap class labels since fewer classes used to match MVT
        :return: Dict mapping NYU class id to new index
        '''
        print("Remapping classes")
        nyu_to_new_idx = {}
        for i in range(len(all_classes)):
            nyu_to_new_idx.update({all_classes[i]:i})

        return nyu_to_new_idx


    def _relabel_batch(self, batch, nyu_to_new_idx):
        '''
        Relabels class ids in data to be within range of 0 to max_classes
        '''
        class_labels = batch['class_labels']
        target_class = batch['target_class']

        for i in range(class_labels.shape[0]):
            for j in range(class_labels.shape[1]):
                orig = int(class_labels[i][j])
                class_labels[i][j] = nyu_to_new_idx[orig]
        
        for i in range(len(target_class)):
            target_class[i] = nyu_to_new_idx[int(target_class[i])]


    def preprocess_batch(self, args, batch):
        self._relabel_batch(batch, self.nyu_to_new_idx)

        batch_keys = self._make_batch_keys(args)
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(self.device)

        # TODO: move this to inside the model
        lang_tokens = self.tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch['lang_tokens'] = lang_tokens

        return batch


    def load_model(self, args, max_context_len):
        metadata_file = os.path.join(args.data_path, 'metadata.json')

        with open(metadata_file) as f:
            metadata = json.load(f)
            all_classes = metadata['class_ids']
            class_name_list = metadata['class_names']

        n_classes = len(all_classes)
        print("num classes:", n_classes)
        if n_classes < TOTAL_CLASSES: # number used by MVT (hardcoded)
            num_class_diff = TOTAL_CLASSES - n_classes
            all_classes += [self.pad_idx for _ in range(num_class_diff)]
            class_name_list += ['PAD' for _ in range(num_class_diff)]

        self.nyu_to_new_idx = self._remap_classes(all_classes)
        self.pad_idx = self.nyu_to_new_idx[self.pad_idx]

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        class_name_tokens = tokenizer(class_name_list, return_tensors='pt', padding=True)
        for name in class_name_tokens.data:
            class_name_tokens.data[name] = class_name_tokens.data[name].to(self.device)
        
        model = ReferIt3DNet_transformer(
            args=args, 
            n_obj_classes=TOTAL_CLASSES,
            max_context_len=max_context_len, 
            class_name_tokens=class_name_tokens, 
            ignore_index=self.pad_idx, 
            log_freq=args.log_freq,
        )

        model = model.to(self.device)
        
        return model


    def get_optimizer(self, model, args):
        params = [
            {'params': model.language_encoder.parameters(), 'lr': args.lr * 0.1},
            {'params': model.refer_encoder.parameters(), 'lr': args.lr * 0.1},
            {'params': model.object_encoder.parameters(), 'lr': args.lr},
            {'params': model.obj_feature_mapping.parameters(), 'lr': args.lr},
            {'params': model.box_feature_mapping.parameters(), 'lr': args.lr},
            {'params': model.language_clf.parameters(), 'lr': args.lr},
            {'params': model.object_language_clf.parameters(), 'lr': args.lr},
        ]
        optimizer = optim.Adam(params, lr=args.lr)

        return optimizer


    def get_scheduler(self, optimizer, args):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_decay_rate)

        return scheduler
    

    @torch.no_grad()
    def _cls_pred_stats(self, logits, gt_labels, ignore_label):
        '''
        Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
        :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
        :param gt_labels: The ground truth labels of size: B x N_Objects
        :param ignore_label: The label of the padding class (to be ignored)
        :return: The mean accuracy and lists of correct and wrong predictions
        '''
        predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
        valid_indices = gt_labels != ignore_label

        predictions = predictions[valid_indices]
        gt_labels = gt_labels[valid_indices]

        correct_guessed = gt_labels == predictions
        assert (type(correct_guessed) == torch.Tensor)

        found_samples = gt_labels[correct_guessed]
        mean_accuracy = torch.mean(correct_guessed.double()).item()
        
        return mean_accuracy, found_samples
    

    def run_one_epoch(self, args, run, epoch, train_loader, test_loader, model, optimizer, criterion, scheduler):
        avg_meters = self.__initalize_average_meters('train')

        # Set the model in training mode
        model.train()
        np.random.seed()  # call this to change the sampling of the point-clouds
        
        best_test_acc = 0
        eval_acc = 0
        for step, batch in enumerate(tqdm.tqdm(train_loader)):

            batch = self.preprocess_batch(args, batch)

            # Forward pass
            res = model(batch, step)
            LOSS = res["loss"].mean()
            
            # Backward
            optimizer.zero_grad()
            LOSS.backward()
            optimizer.step()
            scheduler.step()

            # Update the loss and accuracy meters
            self.__update_average_meters(avg_meters, batch, res, 'train')

            # Log metrics to wandb every log_freq steps
            if step % args.log_freq == 0:
                print(args.log_freq)
                self.__log_metrics_to_wandb(avg_meters)
            
            # Evaluate on validation set every val_freq steps
            if step % args.val_freq == 0:
                val_metrics = self.eval_one_batch(model, test_loader, criterion, args, randomize=False)                
                wandb.log(val_metrics)
            
            # Evaluate on test set every test_freq epochs
            if epoch % args.test_freq == 0:
                test_metrics = self.eval_one_epoch(model, test_loader, criterion, args, randomize=False)
                self.save_checkpoint(args, epoch, model, optimizer, scheduler, f"epoch_{epoch}.pth")

                test_loss = test_metrics['test_total_loss']
                test_acc = test_metrics['test_referential_acc']
                print("Test loss: {}, Test accuracy: {}".format(test_loss, test_acc))

                # save best model
                if eval_acc > best_test_acc:
                    print('Test accuracy improved at epoch {}'.format(epoch))
                    best_test_acc = eval_acc

                    self.save_checkpoint(args, epoch, model, optimizer, scheduler, 'best_epoch.pth')

                wandb.log(test_metrics)
            
            # Save model checkpoint every save_freq epochs
            if epoch % args.save_freq == 0:
                self.save_checkpoint(args, step, model, optimizer, scheduler, f'epoch_{epoch}.pth')

        # generate average metrics
        metrics = {}
        for metric, avg_meter in avg_meters.items():
            metrics[metric] = avg_meter.avg

        return metrics
    

    def inference(self, args, model, batch, randomize=False):
        '''
        Run inference on individual data samples
        :return: return prediction on data sample
        '''
        # Set the model in training mode
        model.eval()

        if randomize:
            np.random.seed()
        else:
            np.random.seed(args.random_seed)

        batch = self.preprocess_batch(args, batch)

        # Forward pass
        res = model(batch)

        predictions = torch.argmax(res['logits'], dim=1)

        #correct = torch.mean((predictions == batch['target_pos']).double()).item()
        
        return predictions


    def eval_one_batch(self, model, test_loader, criterion, args, randomize=False):
        avg_meters = self.__initalize_average_meters('val')

        # Set the model in training mode
        model.eval()
        if randomize:
            np.random.seed()
        else:
            np.random.seed(args.random_seed)
        #rand_inds = np.random.randint(len(test_loader), size=5)
        #sample_loader = test_loader[rand_inds]
        num_batch = 5
        print("Running eval on {} batches".format(num_batch))
        start_idx = np.random.randint(0, len(test_loader)-num_batch)
       
        for step, batch in enumerate(tqdm.tqdm(test_loader)):
            if step == num_batch:
                break
            batch = self.preprocess_batch(args, batch)
            # Forward pass
            res = model(batch)
            LOSS = res["loss"].mean()
            
            # Update the loss and accuracy meters
            self.__update_average_meters(avg_meters, batch, res, 'val')
        
        metrics = {}
        for metric, avg_meter in avg_meters.items():
            metrics[metric] = avg_meter.avg
        
        return metrics


    def eval_one_epoch(self, model, test_loader, criterion, args, randomize=False):
        print("Test evaluation.......")
        avg_meters = self.__initalize_average_meters('test')

        # Set the model in training mode
        model.eval()

        if randomize:
            np.random.seed()
        else:
            np.random.seed(args.random_seed)

        for step, batch in enumerate(tqdm.tqdm(test_loader)):
            batch = self.preprocess_batch(args, batch)

            # Forward pass
            res = model(batch)
            LOSS = res["loss"].mean()

            # Update the loss and accuracy meters
            self.__update_average_meters(avg_meters, batch, res, 'test')

            if wandb.run is not None and step % args.log_freq == 0:
                self.__log_metrics_to_wandb(avg_meters)
        
        metrics = {}
        for metric, avg_meter in avg_meters.items():
            metrics[metric] = avg_meter.avg

        return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model arguments from MVT parser
    parser.add_argument('--view_number', type=int, default=4)
    parser.add_argument('--rotate_number', type=int, default=4)

    parser.add_argument('--label-lang-sup', type=str2bool, default=True)
    parser.add_argument('--aggregate-type', type=str, default='avg')

    parser.add_argument('--encoder-layer-num', type=int, default=3)
    parser.add_argument('--decoder-layer-num', type=int, default=4)
    parser.add_argument('--decoder-nhead-num', type=int, default=8)

    parser.add_argument('--object-latent-dim', type=int, default=768)
    parser.add_argument('--inner-dim', type=int, default=768)

    parser.add_argument('--dropout-rate', type=float, default=0.15)
    parser.add_argument('--lang-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing the target via '
                                                                          'language only is added.')
    parser.add_argument('--obj-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing for each segmented'
                                                                         ' object its class type is added.')
    # standard training arguments
    args = parse_args(parser)

    device = torch.device('cuda')

    trainer = MVTTrainer(args)
    trainer.set_gpu_to_zero_position("0")
    trainer.train(args)
