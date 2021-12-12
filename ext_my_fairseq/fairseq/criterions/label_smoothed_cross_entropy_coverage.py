# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_coverage')
class LabelSmoothedCrossEntropyCriterionCoverage(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, alignment_lambda, coverage_step):
        super().__init__(task, sentence_avg, label_smoothing)
        self.alignment_lambda = alignment_lambda
        self.coverage_step = coverage_step

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--alignment-lambda', default=0.05, type=float, metavar='D',
                            help='weight for the alignment loss')
        parser.add_argument('--coverage_step', default=0, type=int, metavar='D',
                            help='num of steps using the coverage loss')

    def forward(self, model, sample, reduce=True, update_num=None):    # For Coverage
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        alignment_loss = None
        
        #attn_prob = net_output[1]['attn'][0]
        #print("Attn")
        #print(attn_prob)
        #print(attn_prob.shape)
        #print("Outputs")
        #print(net_output)
        #print(net_output[1]['attn'])
        #print(sample)
        
        # Compute alignment loss only for training set and non dummy batches.
        #if 'alignments' in sample and sample['alignments'] is not None:
        coverage_loss = self.compute_coverage_loss(sample, net_output)
        
        # coverage_step  : coverage step
        # update_num     : current step
        #apply_step = update_num - coverage_step
        if update_num is None:
            update_num = 0
        
        if coverage_loss is not None and update_num >= self.coverage_step:
        #if coverage_loss is not None:
            logging_output['coverage_loss'] = utils.item(coverage_loss.data)
            loss += self.alignment_lambda * coverage_loss
        
        return loss, sample_size, logging_output

    def compute_coverage_loss(self, sample, net_output):
        attn = net_output[1]['attn'][0]
        
        cov = torch.cumsum(attn, dim=1)
        cov_loss = torch.where(attn > cov, cov, attn)
        
        return cov_loss.sum()

    def compute_alignment_loss(self, sample, net_output):
        attn_prob = net_output[1]['attn'][0]
        
        bsz, tgt_sz, src_sz = attn_prob.shape
        attn = attn_prob.view(bsz * tgt_sz, src_sz)

        align = sample['alignments']
        align_weights = sample['align_weights'].float()

        if len(align) > 0:
            # Alignment loss computation. align (shape [:, 2]) contains the src-tgt index pairs corresponding to
            # the alignments. align_weights (shape [:]) contains the 1 / frequency of a tgt index for normalizing.
            loss = -((attn[align[:, 1][:, None], align[:, 0][:, None]]).log() * align_weights[:, None]).sum()
        else:
            return None

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        #alignment_loss_sum = utils.item(sum(log.get('alignment_loss', 0) for log in logging_outputs))
        coverage_loss_sum = utils.item(sum(log.get('coverage_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        #metrics.log_scalar('alignment_loss', alignment_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('coverage_loss', coverage_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
