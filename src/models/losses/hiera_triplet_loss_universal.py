import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from src.models.losses.tree_triplet_loss import TreeTripletLoss


def prepare_targets(targets):
    """
    prepare level 1 (super class) ground truth and super classes's sub classes indices
    :param targets:
    :return: original ground truth, level 1 ground truth, level 1 super class's subclasses indices
    """
    pass


def losses_hiera(predictions, targets, targets_top, num_classes, indices_high, eps=1e-8, gamma=2):

    # predictions to 0-1 range
    # replace 255 in targets with 0
    # convert target into one hot vector
    # replace 255 in targets top by 0
    # convert target_top into one hot vector
    # -----------



    pass