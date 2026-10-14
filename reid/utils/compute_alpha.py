import torch
import torch.nn.functional as F

@ torch.no_grad()
def compute_alpha(global_proto_type, new_proto_type, pic_num_total, pic_num):
    """
    计算全局原型和新原型之间的相似度，并根据温度参数调整相似度分布。
    
    参数:
    global_proto_type (torch.Tensor): 全局原型张量，形状为 (N, D)，其中 N 是原型数量，D 是特征维度。
    new_proto_type (torch.Tensor): 新原型张量，形状为 (M, D)，其中 M 是新原型数量，D 是特征维度。
    temperature (float): 温度参数，用于调整相似度分布的平滑度。
    
    返回:
    alpha(float) ,代表新旧模型之间的融合系数
    """
    # 调试
    print("当前全局原型个数:",len(global_proto_type.keys()))
    print("当前新原型个数:",len(new_proto_type.keys()))

    # 计算训练集数量
    train_ratio = pic_num / pic_num_total

    # 计算重叠率
    old_ids = list(global_proto_type.keys())
    new_ids = list(new_proto_type.keys())

    old_set = set(old_ids)
    new_set = set(new_ids)

    inter = len(old_set & new_set)
    union = len(old_set | new_set) 
    overlap_pids = list(old_set & new_set)
    new_only_pids = list(new_set - old_set)

    # 调试
    print("重叠ID数量:",inter)
    print("训练ID数量:",len(new_ids))
    print("新ID数量:",len(new_only_pids))
    print("总ID数量:",union)

    # 重叠ID的相似度(原型余弦)
    if len(overlap_pids) > 0:
        f_old = torch.stack([global_proto_type[pid]["mean_features"] for pid in overlap_pids]) # (K,D)
        f_new = torch.stack([new_proto_type[pid]["mean_features"] for pid in overlap_pids])     # (K,D)
        sim_mean = F.cosine_similarity(f_old, f_new, dim=1).mean().item()
        sim_mean = (sim_mean + 1) / 2  # normalize to 0~1
    else:
        sim_mean = 0.0

    # 计算相同id在新旧模型下的discrim_gain
    if len(overlap_pids) == 0:
        discrim_gain = 0.0
    else:
        gains = []
        f_old = torch.stack([global_proto_type[pid]["mean_features"].view(-1) for pid in old_ids]) # (K,D)
        f_new = torch.stack([new_proto_type[pid]["mean_features"].view(-1) for pid in new_ids])     # (L,D)
        old_norm = F.normalize(f_old, dim=1)
        new_norm = F.normalize(f_new, dim=1)

        for pid in overlap_pids:
            idx_old = old_ids.index(pid)
            idx_new = new_ids.index(pid)

            f_old = F.normalize(global_proto_type[pid]["mean_features"].view(1,-1),dim=1)
            sims_old = torch.mm(f_old, old_norm.t()).squeeze(0)
            sims_old[idx_old] = -1.0
            max_neg_old = sims_old.max().item()

            f_new = F.normalize(new_proto_type[pid]["mean_features"].view(1,-1), dim=1)
            sims_new = torch.mm(f_new, new_norm.t()).squeeze(0)
            sims_new[idx_new] = -1.0
            max_neg_new = sims_new.max().item()

            gains.append(max_neg_old - max_neg_new)

    discrim_gain = sum(gains) / len(gains)
    # normalize discrim_gain to 0~1 (rough)
    discrim_gain = (discrim_gain + 1)/2
    discrim_gain = max(0.0, min(1.0, discrim_gain))

    # noveity new IDs的判别力提升
    if len(new_only_pids) == 0:
        novelty_gain = 0.0
    else:
        f_old = torch.stack([global_proto_type[pid]["mean_features"].view(-1) for pid in old_ids]) # (K,D)
        old_norm = F.normalize(f_old, dim=1)

        max_sims = []
        for pid in new_only_pids:
            f_new = F.normalize(new_proto_type[pid]["mean_features"].view(1,-1), dim=1)
            sims_new = torch.mm(f_new, old_norm.t()).squeeze(0)
            max_sims.append(sims_new.max().item())
        
        novelty_gain = 1 - (sum(max_sims) / len(max_sims))
        # normalize to 0~1
        novelty_gain = max(0.0, min(1.0, novelty_gain))

    # 计算alpha
    alpha = 0.3 * sim_mean + 0.4 * discrim_gain + 0.3 * train_ratio

    # 调试
    print("相似度均值:",sim_mean)
    print("判别力提升:",discrim_gain)
    print("新ID判别力:",novelty_gain)
    print("训练比例:",train_ratio)

    alpha = max(0.0, min(1.0, alpha))  # 确保alpha在0到1之间

    return alpha
