#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset
from torch_uncertainty.metrics import AdaptiveCalibrationError, CalibrationError, CategoricalNLL
from calutils import *
import pickle


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.to(torch.device(args.cuda_device))
            negative_sample = negative_sample.to(torch.device(args.cuda_device))
            subsampling_weight = subsampling_weight.to(torch.device(args.cuda_device))

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3)**3 +
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, calibration_models_list, calibrate=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        fname1, fname2, fname3 = (args.save_path + '/metrics_dict_valid', args.save_path + '/all_model_score_valid',
                                  args.save_path + '/all_positive_arg_valid')

        if not checkFile(fname1) or not checkFile(fname2) or not checkFile(fname3):
            print('Generate metrics_dict, all_model_score, all_positive_arg')
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            metrics_dict = {
                'MRR': [], 'MR': [], 'HITS@1': [], 'HITS@3': [], 'HITS@10': []
            }
            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            all_model_score = torch.zeros(2 * len(test_triples), args.nentity)
            all_positive_arg = torch.zeros_like(all_model_score)
            batch_idx = 0  # Initialize batch index
            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in tqdm(test_dataset):
                        if args.cuda:
                            positive_sample = positive_sample.to(torch.device(args.cuda_device))
                            negative_sample = negative_sample.to(torch.device(args.cuda_device))
                            filter_bias = filter_bias.to(torch.device(args.cuda_device))

                        batch_size = positive_sample.size(0)

                        model_score = model((positive_sample, negative_sample), mode)
                        model_score += filter_bias
                        argsort = torch.argsort(model_score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        # Calculate start and end indices for assignment
                        start_idx = batch_idx * args.test_batch_size
                        end_idx = start_idx + batch_size

                        # Dynamically calculate remaining rows for the last batch
                        if end_idx > all_model_score.size(0):
                            end_idx = all_model_score.size(0)  # Adjust for the last batch
                            batch_size = end_idx - start_idx  # Update batch size for the last batch

                        all_model_score[start_idx:end_idx, :] = model_score[:batch_size, :]

                        for i in range(batch_size):
                            all_positive_arg[start_idx + i, positive_arg[i]] = 1

                        batch_idx += 1

                        for batch_index in range(batch_size):
                            ranking = (argsort[batch_index, :] == positive_arg[batch_index]).nonzero()
                            assert ranking.size(0) == 1

                            ranking = 1 + ranking.item()

                            metrics_dict['MRR'].append(1.0 / ranking)
                            metrics_dict['MR'].append(float(ranking))
                            metrics_dict['HITS@1'].append(1.0 if ranking <= 1 else 0.0)
                            metrics_dict['HITS@3'].append(1.0 if ranking <= 3 else 0.0)
                            metrics_dict['HITS@10'].append(1.0 if ranking <= 10 else 0.0)

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1
            pickle.dump(metrics_dict, open(fname1, 'wb'))
            pickle.dump(all_model_score, open(fname2, 'wb'))
            pickle.dump(all_positive_arg, open(fname3, 'wb'))
        else:
            print('Load metrics_dict, all_model_score, all_positive_arg')
            metrics_dict = pickle.load(open(fname1, 'rb'))
            all_model_score = pickle.load(open(fname2, 'rb'))
            all_positive_arg = pickle.load(open(fname3, 'rb'))

        gc.collect()
        torch.cuda.empty_cache()

        from scipy.stats import entropy
        # Compute softmax and sort scores
        all_model_score = all_model_score.softmax(-1).cpu()
        sorted_all_model_score = torch.sort(all_model_score, dim=1, descending=True)[0]

        # Vectorized KL divergence computation
        kl_divergences = np.array([
            entropy(sorted_all_model_score[:, i - 1].numpy(), sorted_all_model_score[:, i].numpy())
            for i in range(1, sorted_all_model_score.shape[1])
        ])

        # Find maximum jump information
        arg_max_jump = kl_divergences.argmax()
        print("Arg Max Jump:", arg_max_jump)

        if calibrate:
            for calibration_model in calibration_models_list:
                calibration_model.fit(all_model_score, all_positive_arg)

        metrics = {metric: sum(values) / len(values) for metric, values in metrics_dict.items()}

        return metrics

    @staticmethod
    def calibration_predict(model, test_triples, all_true_triples, args, calibration_models_list):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        # Initialize metrics dictionaries
        original_metrics_dict = {k: [] for k in ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']}
        metrics_dict = {calibration_model.name: {} for calibration_model in calibration_models_list}

        # Define metrics
        metric_calibration = ['ECE', 'AECE', 'NLL']
        metric_kge = ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']

        # Initialize dictionaries for each calibration model
        for calibration_model_name in metrics_dict:
            metrics_dict[calibration_model_name] = {metric: [] for metric in metric_kge + metric_calibration}

        metrics_after = {calibration_model.name: {
            'ECE': CalibrationError(task="multiclass", num_classes=args.nentity),
            'AECE': AdaptiveCalibrationError(task="multiclass", num_classes=args.nentity),
            'NLL': CategoricalNLL()
        } for calibration_model in calibration_models_list}

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with (torch.no_grad()):
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in tqdm(test_dataset):
                    if args.cuda:
                        positive_sample, negative_sample, filter_bias = (
                            positive_sample.to(args.cuda_device),
                            negative_sample.to(args.cuda_device),
                            filter_bias.to(args.cuda_device)
                        )

                    batch_size = positive_sample.size(0)
                    original_model_score = model((positive_sample, negative_sample), mode) + filter_bias

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError(f"mode {mode} not supported")

                    target = positive_arg.cpu()
                    original_argsort = torch.argsort(original_model_score, dim=1, descending=True)

                    for i in range(batch_size):
                        original_ranking = (original_argsort[i] ==
                                            positive_arg[i].cpu()).nonzero(as_tuple=True)[0].item() + 1
                        original_metrics_dict['MRR'].append(1.0 / original_ranking)
                        original_metrics_dict['MR'].append(float(original_ranking))
                        for k in [1, 3, 10]:
                            original_metrics_dict[f'HITS@{k}'].append(1.0 if original_ranking <= k else 0.0)

                    for calibration_model in calibration_models_list:
                        probs_after = calibration_model.predict(original_model_score).cpu()
                        new_argsort = torch.argsort(probs_after, dim=1, descending=True)

                        # Update post-calibration metrics
                        for metric_name, metric_fn in metrics_after[calibration_model.name].items():
                            if metric_name == 'Entropy':
                                metric_fn.update(probs_after)  # Only pass `probs_before` for Entropy
                            else:
                                metric_fn.update(probs_after,
                                                 target)  # Pass both `probs_before` and `target` for other metrics

                        # do not print metric_kge
                        for i in range(batch_size):
                            new_ranking = (new_argsort[i] == positive_arg[i].cpu()).nonzero(as_tuple=True)[0].item() + 1
                            metrics_dict[calibration_model.name]['MRR'].append(1.0 / new_ranking)
                            metrics_dict[calibration_model.name]['MR'].append(float(new_ranking))
                            for k in [1, 3, 10]:
                                metrics_dict[calibration_model.name][f'HITS@{k}'].append(
                                    1.0 if new_ranking <= k else 0.0)

                    if step % args.test_log_steps == 0:
                        logging.info(f'Evaluating the model... ({step}/{total_steps})')

                    step += 1

        # Aggregate results
        original_metrics = {metric: sum(values) / len(values) for metric, values in original_metrics_dict.items()}
        calibrate_metrics = {
            f'{model}_{metric}': sum(values) / len(values) if values else 0
            for model, metrics in metrics_dict.items() for metric, values in metrics.items()
        }

        # Inject calibration error metrics after scaling
        for model_name in metrics_after:
            calibrate_metrics[f'{model_name}_ECE'] = metrics_after[model_name]['ECE'].compute().item()
            calibrate_metrics[f'{model_name}_AECE'] = metrics_after[model_name]['AECE'].compute().item()
            calibrate_metrics[f'{model_name}_NLL'] = metrics_after[model_name]['NLL'].compute().item()

        return original_metrics, calibrate_metrics