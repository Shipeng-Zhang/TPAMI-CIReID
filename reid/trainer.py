from __future__ import print_function, absolute_import
import time
from torch.nn import functional as F
import torch
import torch.nn as nn
import tqdm
import numpy as np

from reid.loss.loss_uncertrainty import TripletLoss_set
# Linear classifier baseline (after restoring the commented class there):
# from reid.loss.loss_uncertrainty import TripletLoss_set_2
from .utils.meters import AverageMeter
from reid.metric_learning.distance import cosine_similarity
from reid.utils.efficiency_profiler import profiler

try:
    from gradient_visualizer import plot_gradient_bar_chart_2
except ImportError:
    # Gradient visualization is an optional experiment utility and is not
    # required for model training or evaluation.
    plot_gradient_bar_chart_2 = None

device = torch.device('cuda:0')
        
class Trainer(object):
    def __init__(self, args, model, writer=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.writer = writer
        self.uncertainty = True  # 是否使用不确定性损失
        if self.uncertainty:
            # ETF (ours): TripletLoss_set consumes s_features.
            self.criterion_triple = TripletLoss_set()
            self.criterion_triple_2 = TripletLoss_set() # ETF下
        self.criterion_ce = nn.CrossEntropyLoss()  # 交叉熵损失
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')  # KL散度损失
        self.feature_distillation_loss = nn.MSELoss()  # 添加特征蒸馏损失
        self.AF_weight = args.AF_weight  # 反遗忘损失的权重
        self.n_sampling = args.n_sampling  # 采样次数
        # 画图使用
        self.layer_groups = {
            'conv1' : [],
            'layer1' : [], 
            'layer2' : [], 
            'layer3' : []
        }
        
        #初始化设备
        self.device = torch.device('cuda:0')
        self.model.to(self.device)

     # ========= 梯度打印函数 =========
    def compute_grad_norm(self,model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
        
    def train(self, epoch, data_loader_train, optimizer, training_phase,
              global_proto_type=None, train_iters=200):

        self.layer_groups = {k: [] for k in self.layer_groups}
        self.model.train()  # 设置模型为训练模式
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr1 = AverageMeter()
        losses_tr2 = AverageMeter()
        losses_etf_align = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()  # 获取训练数据
            data_time.update(time.time() - end)


            s_inputs, targets, cids, domains = self._parse_data(train_inputs)
            overall_scope = profiler.measure("overall_framework")
            overall_scope.__enter__()
            # 获取模型输出特征
            # -s_features: [BS,2028] 对应每一张图片的特征向量,已经经过归一化
            # -merge_feat: [BS,n_sampling+1,2048] 对应不确定性uncentainty开启时，采样之后的均值特征
            # -cls_outputs: [BS,n_sampling+1,num_classes] 对应每张图片对应每个类别的概率
            # -out_var: [BS,2048] 对应每张图片的方差信息
            # -feat_final_layer: [B,1024,H,W] 对应经过图片主干之后的特征图
            s_features, merge_feat, cls_outputs, out_var, feat_final_layer = self.model(s_inputs)

            loss_ce, loss_tp1 = 0, 0
            loss_tp2 = 0

            # 全局tp2实现
            tp2_scope = profiler.measure("L_tri_2")
            tp2_scope.__enter__()
            if global_proto_type is not None:
                pairs = [(global_proto_type[pid.item()]['mean_features'], pid.item())
                        for pid in targets if pid.item() in global_proto_type]
                if (len(pairs) == 0):
                    loss_tp2 = torch.tensor(0.0, device=device)
                else:
                    global_features = torch.cat([pair[0] for pair in pairs],dim=0)
                    global_labels = torch.tensor([pair[1] for pair in pairs],device = device)

                    sampled_prototypes = []
                    sampled_labels = []
                    selected_indices = torch.randperm(global_features.size(0)) # 随机选择一些原型索引
                    for idx in selected_indices:
                        # 采样每个原型两次
                        sampled_proto = global_features[idx].unsqueeze(0)  # 确保每个原型是 [1, 2048]
                        sampled_prototypes.append(sampled_proto)
                        sampled_prototypes.append(sampled_proto)  # 每个原型采样两次

                        sampled_labels.extend([global_labels[idx].item()] * 2)  # 每个标签对应两次采样

                    # 将所有采样的原型拼接成一个 [batch_size, 2048] 的张量
                    sampled_prototypes = torch.cat(sampled_prototypes, dim=0)
                    sampled_prototypes = F.normalize(sampled_prototypes, p=2, dim=1)
                    sampled_labels = torch.tensor(sampled_labels,device = device)


                    tp2_features = torch.cat([s_features,sampled_prototypes],dim=0)
                    tp2_labels = torch.cat([targets,sampled_labels],dim=0)
                    loss_tp2 = torch.tensor(0.0, device=s_features.device)
                    loss_tp2 = self.criterion_triple_2(tp2_features, tp2_labels)[0]
                    loss_tp2 = loss_tp2 * 1.5
            tp2_scope.__exit__(None, None, None)

            loss_tp1 = self.criterion_triple(merge_feat, targets)[0]
            
            loss_tp1 = loss_tp1 * 1.5

            # 计算交叉熵损失
            for s_id in range(1 + self.n_sampling):
                loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets)
            loss_ce = loss_ce / (1 + self.n_sampling)
            loss_ce = loss_ce * 1

            BS = merge_feat.size(0)
            n_sample_plus1 = merge_feat.size(1)
            out_dim = merge_feat.size(2)

            #shape for ETF
            sample_feat = merge_feat[:,1:,:]
            sample_feat = sample_feat.reshape(BS * (n_sample_plus1 - 1), out_dim)
            targets_expanded = targets.view(BS,1).expand(BS,n_sample_plus1 - 1).reshape(-1)

            # ======== ETF 对齐损失 ========
            etf_scope = profiler.measure("L_etf")
            etf_scope.__enter__()
            # It remains zero for the linear-classifier baseline.
            loss_etf_align = torch.tensor(0.0, device=s_features.device)
            if hasattr(self.model.module, "classifier") and hasattr(self.model.module.classifier, "ori_M"):
                M = self.model.module.classifier.ori_M  # (d, C)
                M = F.normalize(M, p=2, dim=0)
                scale = 30.0

                s_features_norm = F.normalize(s_features, p=2, dim=1)
                unique_main, inv_main = torch.unique(targets, return_inverse=True) # inv 保存每个样本对应在unique_labels中的索引
                M_batch = M[:, unique_main].t()[inv_main]  # [BatchSize, feature_dim] 
                cos_sim1 = torch.sum(s_features_norm * M_batch, dim=1)
                loss_etf_main = torch.mean(1 - torch.sigmoid(scale * cos_sim1))

                sample_feat_norm = F.normalize(sample_feat, p=2, dim=1)
                unique_samp, inv_samp = torch.unique(targets_expanded,return_inverse=True)
                M_batch = M[:,unique_samp].t()[inv_samp]
                cos_sim2 = torch.sum(sample_feat_norm * M_batch, dim=1)
                loss_etf_sample = torch.mean(1 - torch.sigmoid(scale * cos_sim2))

                lambda_sample = 0.2
                loss_etf_align = loss_etf_main + lambda_sample * loss_etf_sample
            loss_etf_align = loss_etf_align * 1.5
            etf_scope.__exit__(None, None, None)

            loss = torch.tensor(0.0, device=s_features.device, requires_grad=True)
            if self.uncertainty:
                # ETF (ours)
                loss = loss_ce  + loss_tp1  + loss_tp2 + loss_etf_align
            else:
                loss = loss_ce

            # ETF (ours)
            losschoice = loss_ce + loss_tp1 + loss_tp2 + loss_etf_align

            if self.uncertainty:
                losses_tr2.update(loss_tp2.item())
                losses_tr1.update(loss_tp1.item())
                losses_etf_align.update(loss_etf_align.item())
            losses_ce.update(loss_ce.item())

            optimizer.zero_grad()
            losschoice.backward(retain_graph=True)

            gradient_scores = {}
            # 计算梯度的绝对值，并获取所有参数的梯度值
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:  # 确保梯度存在
                    grad_score = param.grad.abs().sum().item()  # 计算梯度的绝对值之和
                    gradient_scores[name] = grad_score

            # 按照梯度分数对参数进行排序（从大到小）
            sorted_params = sorted(gradient_scores.items(), key=lambda x: x[1], reverse=True)

            # 选择前 20% 梯度最大的参数
            top_20_percent_count = int(len(sorted_params) * 0.2)
            top_20_percent = sorted_params[:top_20_percent_count]

            # # 打印更新的参数信息
            # print("更新的参数:")
            # for name, score in top_20_percent:
            #     print(f"参数: {name}, 梯度分数: {score:.6f}")

            # 将前 20% 的梯度最大的参数设为 requires_grad = True，其余的设置为 False
            for name, param in self.model.named_parameters():
                # 只更新前 20% 的参数
                if any(name == top_name for top_name, _ in top_20_percent):
                    param.requires_grad = True  # 仅更新梯度分数前 20% 的参数
                else:
                    param.requires_grad = False  # 其余的参数冻结
                    
            loss.backward()
            # if epoch == 0 or epoch == 30 or epoch == 59:
            #     self.accumulate_gradients()
            optimizer.step()

            #恢复所有参数的训练
            for param in self.model.parameters():
                param.requires_grad = True
            overall_scope.__exit__(None, None, None)

            batch_time.update(time.time() - end)
            end = time.time()

            if self.writer is not None:
                self.writer.add_scalar(
                    tag=f"loss/Loss_ce_{training_phase}",
                    scalar_value=losses_ce.val,
                    global_step=epoch * train_iters + i
                )
                self.writer.add_scalar(
                    tag=f"loss/Loss_tr1_{training_phase}",
                    scalar_value=losses_tr1.val,
                    global_step=epoch * train_iters + i
                )
                self.writer.add_scalar(
                    tag=f"loss/Loss_tr2_{training_phase}",
                    scalar_value=losses_tr2.val,
                    global_step=epoch * train_iters + i
                )
                self.writer.add_scalar(
                    tag=f"loss/Loss_etf_align_{training_phase}",
                    scalar_value=loss_etf_align.item(),
                    global_step=epoch * train_iters + i
                )

            if (i + 1) == train_iters:
                # if epoch == 0 or epoch == 30 or epoch == 59:
                #     self.visualize_gradients(epoch)
                print(
                    f"\nEpoch: [{epoch}][{i + 1}/{train_iters}]"
                    f"\n----------------------------------------"
                    f"\nBatch Time       : {batch_time.val:.3f}s (avg: {batch_time.avg:.3f}s)"
                    f"\nLoad Time        : {data_time.val:.3f}s (avg: {data_time.avg:.3f}s)"
                    f"\nLoss_ce          : {losses_ce.val:.3f} (avg: {losses_ce.avg:.3f})"
                    f"\nLoss_tp1         : {losses_tr1.val:.3f} (avg: {losses_tr1.avg:.3f})"
                    f"\nLoss_tp2         : {losses_tr2.val:.3f} (avg: {losses_tr2.avg:.3f})"
                    f"\nLoss_etf         : {losses_etf_align.val:.3f} (avg: {losses_etf_align.avg:.3f})"
                    f"\n----------------------------------------"
                    f"\nTotal Loss       : {loss.item():.3f}"
                )

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.to(device)  # 将输入张量移动到主设备
        targets = pids.to(device)  # 将目标张量移动到主设备
        #inputs = imgs.cuda()
        #targets = pids.cuda()
        return inputs, targets, cids, domains
    
    def gaussian_sample(self,proto_features, n_samples):
        """
        从 proto_features 中进行高斯采样
        :param proto_features: 原型特征 (C, feature_dim)
        :param n_samples: 每个原型要采样的数量
        :return: 采样的特征
        """ 
        # 获取原型特征的维度
        C, feature_dim = proto_features.size()
        # 生成随机高斯噪声
        noise = torch.randn(C * n_samples, feature_dim).to(proto_features.device) * 0.2   # 0.1 是噪声的标准差
        # 对每个原型特征添加高斯噪声
        sampled_prototypes = proto_features.repeat(n_samples, 1) + noise
        return sampled_prototypes
    # 在 Trainer 类中添加这个新方法
    def gaussian_noise(self, features, noise_level=0.1):
        """
        给输入的特征张量添加高斯噪声。
        :param features: 特征张量 (N, feature_dim)
        :param noise_level: 噪声的标准差
        :return: 添加了噪声的特征张量
        """
        noise = torch.randn_like(features) * noise_level
        return features + noise
     
    def triplet_loss_no_positive(self, anchor, negative, margin=0.3):
        """修改后的三元组损失函数，没有正样本，仅推远负样本"""
        # 计算负样本与锚点之间的欧氏距离
        neg_distance = F.pairwise_distance(anchor, negative, 2)  
        loss = F.relu(neg_distance + margin)  # 只需要推远负样本
        return loss.mean()  # 返回标量 Tensor

    def accumulate_gradients(self):
        for name,param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.detach().norm().item()

            # ⭐ 根据名字归类（关键）
            if 'conv1' in name:
                self.layer_groups['conv1'].append(grad_norm)
            elif 'layer1' in name:
                self.layer_groups['layer1'].append(grad_norm)
            elif 'layer2' in name:
                self.layer_groups['layer2'].append(grad_norm)
            elif 'layer3' in name:
                self.layer_groups['layer3'].append(grad_norm)

    def visualize_gradients(self,epoch):
        if plot_gradient_bar_chart_2 is None:
            return
        # 聚合（取平均）
        layer_names = []
        gradients = []

        for layer, grads in self.layer_groups.items():
            if len(grads) > 0:
                layer_names.append(layer)
                gradients.append(np.mean(grads))
            else:
                layer_names.append(layer)
                gradients.append(0)

        save_path = f"./gradient_etf_epoch_{epoch}.png"

        plot_gradient_bar_chart_2(layer_names, gradients, save_path)

