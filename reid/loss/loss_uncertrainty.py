# ETF 分类器

import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

# 无正样本或数量不一致版
def hard_example_mining_set(dist_mat, labels, return_inds=False):

    assert len(dist_mat.size()) == 2
    M = dist_mat.size(0)

    # print("dist_mat shape:", dist_mat.shape)
    # print("labels shape:", labels.shape)

    # shape [N, N]
    is_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    is_neg = ~is_pos
    is_pos.fill_diagonal_(False)

    pos_dist = dist_mat.clone()
    pos_dist[~is_pos] = float('-inf')
    neg_dist = dist_mat.clone()
    neg_dist[~is_neg] = float('inf')

    dist_ap, relative_p_inds = torch.max(pos_dist, 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(neg_dist, 1, keepdim=True)

    if return_inds:
        return dist_ap, dist_an, relative_p_inds, relative_n_inds

    return dist_ap, dist_an

class TripletLoss_set(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, s_features, labels, normalize_feature=False):
        '''
        s_features:
            batch_size*feat_dim
        labels:
            batch_size
        sample_feat:
            (n_sampling x batch_size)*feat_dim
        mean_feat:
            batch_size*feat_dim
        '''
        BS = s_features.size(0)  # batch_size / 2 * batch_size
        if normalize_feature:
            s_features = normalize(s_features, axis=-1)

        # 使用余弦相似度构造"角度距离"
        s_features_norm = F.normalize(s_features, p=2, dim=1)
        # sample_feat_norm = F.normalize(sample_feat, p=2, dim=1)
        # mean_feat_norm = F.normalize(mean_feat, p=2, dim=1)
        cosine_sim = torch.matmul(s_features_norm, s_features_norm.t())  # [BS, BS]
        scale = 30.0  # ETF下推荐 scale=10~64
        dist_mat = scale * (1 - cosine_sim)  # 越相似距离越小

        dist_ap, dist_an = hard_example_mining_set(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an