#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import entropy
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

        logging.info('KGE Calibrator training start.')
        calibration_scores, arg_max_jump = _prepare_scores_for_calibration(torch.as_tensor(all_model_score))

        if calibrate:
            _fit_calibrators(calibration_models_list, calibration_scores, torch.as_tensor(all_positive_arg), arg_max_jump)
        logging.info('KGE Calibrator training finished.')

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

        collected_scores: List[torch.Tensor] = []
        collected_targets: List[torch.Tensor] = []

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

                    collected_scores.append(original_model_score.detach().cpu())
                    collected_targets.append(positive_arg.detach().cpu())

                    if step % args.test_log_steps == 0:
                        logging.info(f'Evaluating the model... ({step}/{total_steps})')

                    step += 1

        if collected_scores:
            stacked_scores = torch.cat(collected_scores, dim=0)
            stacked_targets = torch.cat(collected_targets, dim=0)
        else:
            stacked_scores = torch.empty((0, args.nentity), dtype=torch.float32)
            stacked_targets = torch.empty((0,), dtype=torch.long)

        return _compute_metrics_from_scores(stacked_scores, stacked_targets, calibration_models_list, args.nentity)


def _prepare_scores_for_calibration(all_model_score: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Convert logits to probabilities and determine the jump index.

    Parameters
    ----------
    all_model_score: torch.Tensor
        Logits or scores output by the KGE model.

    Returns
    -------
    Tuple[torch.Tensor, int]
        A tuple containing probability scores suitable for calibration and the
        selected jump index following the original jump selection strategy.
    """

    if all_model_score.numel() == 0:
        return all_model_score.clone(), 0

    probabilities = all_model_score.softmax(-1).cpu()
    if probabilities.size(1) <= 1:
        return probabilities, 0

    sorted_probs = torch.sort(probabilities, dim=1, descending=True)[0]
    kl_divergences = np.array([
        entropy(sorted_probs[:, i - 1].numpy(), sorted_probs[:, i].numpy())
        for i in range(1, sorted_probs.shape[1])
    ])

    arg_max_jump = int(kl_divergences.argmax()) if kl_divergences.size else 0
    return probabilities, arg_max_jump


def _fit_calibrators(calibration_models_list, calibration_scores: torch.Tensor, all_positive_arg: torch.Tensor, jump_index: int) -> None:
    """Train calibration models using the provided probability scores."""

    if calibration_scores.numel() == 0:
        return

    for calibration_model in calibration_models_list:
        if calibration_model.name == "KGEC":
            calibration_model.fit(calibration_scores, all_positive_arg, jump_index=jump_index)
        else:
            calibration_model.fit(calibration_scores, all_positive_arg)


def _prepare_positive_indices(labels: torch.Tensor) -> torch.Tensor:
    """Convert one-hot encoded labels into index form when required."""

    labels = labels.to(torch.float32)
    if labels.dim() == 2:
        return torch.argmax(labels, dim=1).to(torch.long)
    return labels.view(-1).to(torch.long)


def _compute_metrics_from_scores(
    all_model_score: torch.Tensor,
    positive_indices: torch.Tensor,
    calibration_models_list,
    nentity: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute ranking and calibration metrics from pre-computed scores."""

    logits = all_model_score.to(torch.float32)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    positive_indices = _prepare_positive_indices(positive_indices)

    if logits.size(0) == 0:
        metric_names = ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']
        original_metrics = {name: 0.0 for name in metric_names}
        calibrate_metrics: Dict[str, float] = {}
        calibration_metric_names = metric_names + ['ECE', 'ACE', 'NLL']
        for calibration_model in calibration_models_list:
            for metric_name in calibration_metric_names:
                calibrate_metrics[f'{calibration_model.name}_{metric_name}'] = 0.0
        return original_metrics, calibrate_metrics

    if positive_indices.shape[0] != logits.shape[0]:
        raise ValueError('Number of labels does not match number of score rows.')

    argsort = torch.argsort(logits, dim=1, descending=True)
    matches = argsort.eq(positive_indices.view(-1, 1))
    ranking_positions = matches.nonzero(as_tuple=False)
    if ranking_positions.numel() == 0:
        raise ValueError('Unable to locate positive targets in the provided scores.')

    ranking = torch.zeros(logits.size(0), dtype=torch.long)
    ranking[ranking_positions[:, 0]] = ranking_positions[:, 1] + 1
    ranking = ranking.to(torch.float32)

    original_metrics = {
        'MRR': torch.mean(1.0 / ranking).item(),
        'MR': torch.mean(ranking).item(),
        'HITS@1': torch.mean((ranking <= 1).to(torch.float32)).item(),
        'HITS@3': torch.mean((ranking <= 3).to(torch.float32)).item(),
        'HITS@10': torch.mean((ranking <= 10).to(torch.float32)).item(),
    }

    calibrate_metrics: Dict[str, float] = {}

    for calibration_model in calibration_models_list:
        probs_after = calibration_model.predict(logits).cpu()
        new_argsort = torch.argsort(probs_after, dim=1, descending=True)
        new_matches = new_argsort.eq(positive_indices.view(-1, 1))
        new_positions = new_matches.nonzero(as_tuple=False)
        new_ranking = torch.zeros_like(ranking, dtype=torch.long)
        new_ranking[new_positions[:, 0]] = new_positions[:, 1] + 1
        new_ranking = new_ranking.to(torch.float32)

        calibrate_metrics[f'{calibration_model.name}_MRR'] = torch.mean(1.0 / new_ranking).item()
        calibrate_metrics[f'{calibration_model.name}_MR'] = torch.mean(new_ranking).item()
        calibrate_metrics[f'{calibration_model.name}_HITS@1'] = torch.mean((new_ranking <= 1).to(torch.float32)).item()
        calibrate_metrics[f'{calibration_model.name}_HITS@3'] = torch.mean((new_ranking <= 3).to(torch.float32)).item()
        calibrate_metrics[f'{calibration_model.name}_HITS@10'] = torch.mean((new_ranking <= 10).to(torch.float32)).item()

        metric_objects = {
            'ECE': CalibrationError(task="multiclass", num_classes=nentity),
            'ACE': AdaptiveCalibrationError(task="multiclass", num_classes=nentity),
            'NLL': CategoricalNLL(),
        }

        for metric_name, metric_fn in metric_objects.items():
            metric_fn.update(probs_after, positive_indices)
            calibrate_metrics[f'{calibration_model.name}_{metric_name}'] = metric_fn.compute().item()

    return original_metrics, calibrate_metrics


def calibrate_and_evaluate_from_scores(
    valid_scores: torch.Tensor,
    valid_labels: torch.Tensor,
    test_scores: torch.Tensor,
    test_labels: torch.Tensor,
    calibration_models_list,
    nentity: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Utility to reproduce script metrics using pre-computed scores."""

    valid_scores_tensor = torch.as_tensor(valid_scores, dtype=torch.float32)
    calibration_scores, jump_index = _prepare_scores_for_calibration(valid_scores_tensor)
    _fit_calibrators(calibration_models_list, calibration_scores, torch.as_tensor(valid_labels), jump_index)

    test_scores_tensor = torch.as_tensor(test_scores, dtype=torch.float32)
    positive_indices = _prepare_positive_indices(torch.as_tensor(test_labels))

    return _compute_metrics_from_scores(test_scores_tensor, positive_indices, calibration_models_list, nentity)
