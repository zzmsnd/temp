import math
from copy import deepcopy
from itertools import cycle
import argparse
import numpy as np
import torch
import torch.optim as optim
from collections import Counter
from loader import get_dataloaders
import metric
from utils import AverageMeter, set_model, DrawCluster, visualization, result_display, display_predictions, convert_to_color, DrawResult
import utils
import torch.nn.functional as F
#我加的 忽略警告
import warnings
warnings.filterwarnings("ignore")
import os
import time

#待改-------------------------
def ortrain(model,
          optim,
          lr_schdlr,
          args,
          selected_label,
          classwise_acc,
          labeled_train_loader,
          unlabeled_train_loader,
          ):
    cls_losses = AverageMeter()
    self_training_losses = AverageMeter()
    # define loss function
    cls_criterion = utils.CrossEntropyLoss()
    selected_label = selected_label.cuda()

    model.train()
    for batch_idx, data in enumerate(zip(cycle(labeled_train_loader), unlabeled_train_loader)):
        x_l, labels_l = data[0][0], data[0][1]
        x_u, x_u_strong, labels_u = data[1][0], data[1][1], data[1][2]
        x_l = x_l.cuda()
        x_u = x_u.cuda()
        x_u_strong = x_u_strong.cuda()
        labels_l = labels_l.cuda()
        batch_size = x_l.size(0)
        t = list(range(batch_size*batch_idx, batch_size*(batch_idx+1), 1))
        t = (torch.from_numpy(np.array(t))).cuda()

        # --------------------------------------
        x = torch.cat((x_l, x_u, x_u_strong), dim=0)
        y, y_pseudo = model(x)
        # cls loss on labeled data
        y_l = y[:args.batch_size]
        cls_loss = cls_criterion(y_l, labels_l)
        # self training loss on unlabeled data
        y_u, _ = y[args.batch_size:].chunk(2, dim=0)
        _, y_u_strong = y_pseudo[args.batch_size:].chunk(2, dim=0)

        #
        confidence, pseudo_labels = torch.softmax(y_u.detach(), dim=1).max(dim=1)
        mask = confidence.ge(0.95 * ((-0.3) * (torch.pow((classwise_acc[pseudo_labels] - 1), 2)) + 1)).float()
        #     self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask).mean()
        # else:
        self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask).mean()
        # if batch_idx == 100:
        #     print(t_p)
        #     print(confidence.mean())
        if t[mask == 1].nelement() != 0:
            selected_label[t[mask == 1]] = pseudo_labels[mask == 1]

        loss = cls_loss + self_training_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        cls_losses.update(cls_loss.item())
        self_training_losses.update(self_training_loss.item())
    return cls_losses.avg, self_training_losses.avg, selected_label


# 模型预热
def warmup_model(model, train_loader, optimizer, criterion, warmup_epochs):
    model.train()
    for epoch in range(warmup_epochs):
        total_loss = 0
        for batch_idx, (x_l, labels_l) in enumerate(train_loader):
            x_l, labels_l = x_l.cuda(), labels_l.cuda()
            optimizer.zero_grad()
            outputs, _ = model(x_l)
            loss = criterion(outputs, labels_l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Warmup Epoch {epoch+1}/{warmup_epochs}, Loss: {total_loss/(batch_idx+1):.4f}")
    return model

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
#-------------------待改
def train(model,
          optim,
          lr_schdlr,
          args,
          selected_label,
          classwise_acc,
          labeled_train_loader,
          unlabeled_train_loader,
          ema_gmm,
          epoch,
          total_epochs,
          combined_losses
          ):
    cls_losses = AverageMeter()
    self_training_losses = AverageMeter()
    cls_criterion = utils.CrossEntropyLoss()
    selected_label = selected_label.cuda()

    # 初始化类别计数器 (在函数内)
    num_classes = args.num_classes
    reliable_class_count = torch.zeros(num_classes, dtype=torch.long, device='cuda')
    unreliable_class_count = torch.zeros(num_classes, dtype=torch.long, device='cuda')

    model.train()
    for batch_idx, data in enumerate(zip(cycle(labeled_train_loader), unlabeled_train_loader)):
        x_l, labels_l = data[0][0], data[0][1]
        x_u, x_u_strong, labels_u = data[1][0], data[1][1], data[1][2]
        x_l = x_l.cuda()
        x_u = x_u.cuda()
        x_u_strong = x_u_strong.cuda()
        labels_l = labels_l.cuda()
        batch_size = x_l.size(0)
        t = list(range(batch_size*batch_idx, batch_size*(batch_idx+1), 1))
        t = (torch.from_numpy(np.array(t))).cuda()

        x = torch.cat((x_l, x_u, x_u_strong), dim=0)
        y, y_pseudo = model(x)
        # cls loss on labeled data
        y_l = y[:args.batch_size]
        cls_loss = cls_criterion(y_l, labels_l)
        # self training loss on unlabeled data
        y_u, _ = y[args.batch_size:].chunk(2, dim=0)
        _, y_u_strong = y_pseudo[args.batch_size:].chunk(2, dim=0)


        # 计算伪标签和损失
        pseudo_probs = torch.softmax(y_u.detach(), dim=1)
        confidence, pseudo_labels = pseudo_probs.max(dim=1)
        loss_w = F.cross_entropy(y_u, pseudo_labels, reduction='none')
        loss_w = (loss_w - loss_w.min()) / (loss_w.max() - loss_w.min() + 1e-8)
        loss_w = loss_w.view(-1, 1)
        loss_w_ori = loss_w
        loss_w_np = loss_w.cpu().detach().numpy()
        loss_w_np = loss_w_np.squeeze()
        combined_losses = np.concatenate([combined_losses,loss_w_np])
        if len(combined_losses) > 0:
            min_value = combined_losses.min()
            max_value = combined_losses.max()
            # 设置最小标准差阈值
            min_std = 1e-4
            if max_value - min_value > 1e-8:
                loss_w = (loss_w - min_value) / (
                        max_value - min_value)
            else:
                loss_w = (loss_w - min_value) / (
                        min_std * 2)
                # 确保结果在[0,1]范围内（处理可能的数值误差）
                loss_w = np.clip(loss_w, 0, 1)

        else:
            loss_w = np.random.rand(1) * 0.9 + 0.1

        # # 使用传入的GMM预测结果筛选伪标签
        # prob = ema_gmm.predict_proba(loss_w.cpu().detach().numpy())
        # prob = torch.from_numpy(prob[:, ema_gmm.means_.argmin()]).cuda()
        #mask = prob > current_threshold  # 根据GMM预测结果筛选
        # 定义阈值衰减参数

        # #改动尝试
        # initial_threshold = 0.85
        # final_threshold = 0.95
        # # 计算当前epoch的动态阈值（线性递增）
        # current_threshold = initial_threshold + ( final_threshold - initial_threshold) * (epoch - 1) / total_epochs
        #计算各个类别的阈值 困难类阈值相对更低
        # 现在最好结果
        initial_threshold = 0.95  # 初始高阈值
        final_threshold = 0.9  # 最终低阈值
        # 计算当前epoch的动态阈值（线性衰减）
        current_threshold = initial_threshold - (initial_threshold - final_threshold) * (epoch - 1) / total_epochs
        # 使用传入的GMM预测结果筛选伪标签
        # 选择均值最小的高斯分量对应的概率（即可靠样本的概率）
        #双重筛选 同时要置信度confidence > 0.9
        prob = ema_gmm.predict_proba(loss_w.cpu().detach().numpy())
        mask = ((torch.from_numpy(prob[:, ema_gmm.means_.argmin()]).cuda() > current_threshold).bool()
                & (confidence > 0.9).bool())
        #mask = torch.from_numpy(prob[:, ema_gmm.means_.argmin()]).cuda() > 0.9
        # 选择均值最大的高斯分量对应的概率（即不可靠样本的概率）
        unco_mask = ((torch.from_numpy(prob[:, ema_gmm.means_.argmax()]).cuda() > current_threshold).bool())
        #unco_mask = torch.from_numpy(prob[:, ema_gmm.means_.argmin()]).cuda() < 0.9
        # 不可靠伪标签筛选：GMM预测结果 < 0.5 且 置信度 < 0.5
        # unco_mask = (prob < 0.5) & (confidence < 0.5)
        #unco_mask = prob < 0.5  # 错误分类样本掩码
        unco_indices = t[unco_mask].cpu().numpy().tolist()  # 错误样本索引
        unco_labels = pseudo_labels[unco_mask].cpu().numpy().tolist()  # 错误样本伪标签

        # 计算错误样本的损失（用于后续高斯拟合）
        unco_losses = loss_w_ori[unco_mask].detach().cpu().numpy().flatten().tolist()

        # === 新增：统计伪标签类别数量 ===
        # 可靠伪标签类别统计
        if mask.sum() > 0:
            reliable_labels = pseudo_labels[mask]
            reliable_class_count += torch.bincount(reliable_labels, minlength=num_classes)
        # 不可靠伪标签类别统计
        if unco_mask.sum() > 0:
            unreliable_labels = pseudo_labels[unco_mask]
            unreliable_class_count += torch.bincount(unreliable_labels, minlength=num_classes)

        # ===============================

        # 应用筛选后的掩码计算自训练损失
        self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask.float()).mean()


        # 更新选中的可靠伪标签
        if t[mask == 1].nelement() != 0:
            selected_label[t[mask == 1]] = pseudo_labels[mask == 1]

        loss = cls_loss + self_training_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        cls_losses.update(cls_loss.item())
        self_training_losses.update(self_training_loss.item())
        

    reliable_counts = reliable_class_count.cpu().numpy().tolist()
    unreliable_counts = unreliable_class_count.cpu().numpy().tolist()
    print("reliable_class_count:", reliable_counts)
    print("unreliable_counts:", unreliable_counts)
    return cls_losses.avg, self_training_losses.avg, selected_label, unco_indices, unco_labels, unco_losses

# test for one epoch
def test(model, test_loader):
    model.eval()
    total_accuracy,  total_num = 0.0, 0.0
    prediction = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            logits, _ = model(data)
            out = torch.softmax(logits, dim=1)
            pred_labels = out.argsort(dim=-1, descending=True)
            total_num += data.size(0)
            total_accuracy += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            for num in range(len(logits)):
                prediction.append(np.array(logits[num].cpu().detach().numpy()))

    return total_accuracy / total_num * 100, prediction


def plot_gmm_fit(correct_losses, incorrect_losses, ema_gmm, epoch):
    plt.figure(figsize=(8, 6))

    # 绘制分类正确/错误的损失直方图
    plt.hist(correct_losses, bins=30, density=True, alpha=0.5, label='correct', color='cyan')
    plt.hist(incorrect_losses, bins=30, density=True, alpha=0.5, label='incorrect', color='salmon')

    # 生成用于绘图的连续横轴数据（归一化范围 [0, 1]）
    x = np.linspace(0, 1, 1000).reshape(-1, 1)

    # 计算混合高斯模型的PDF
    pdf_mix = ema_gmm.score_samples(x)

    # 绘制混合高斯模型的曲线
    plt.plot(x, np.exp(pdf_mix), 'r--', label='Fitted Gaussian Mixture')

    plt.xlabel('Normalized Loss')
    plt.ylabel('Empirical pdf')
    plt.title(f'Epoch {epoch} - Gaussian Mixture Fitting for Correct/Incorrect Losses')
    plt.legend()
    # 构建保存图片的文件名和路径
    save_path = os.path.join('gmm_plots', f'epoch_{epoch}_gmm_fit.png')
    plt.savefig(save_path)
    plt.close()  # 关闭当前图形，释放内存
    #plt.show()


class GaussianMixtureWithHistory(GaussianMixture):
    """继承GaussianMixture并添加对数似然值历史记录功能"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_likelihood_history = []  # 存储每次迭代的对数似然值

    def _e_step(self, X):
        """重写E步骤，捕获对数似然值"""
        log_prob_norm, log_resp = super()._e_step(X)
        self.log_likelihood_history.append(log_prob_norm.sum())  # 保存完整对数似然值
        return log_prob_norm, log_resp

def main():
    dataset_names = ['KSC' 'PU', 'Houston','Botswana','PaviaU']
    parser = argparse.ArgumentParser(description='Pseudo label for HSIC')
    parser.add_argument('--dataset', type=str, default='PaviaU', choices=dataset_names)
    parser.add_argument('--model', default='TRNetwork', type=str, choices='TRNetwork')
    parser.add_argument('--feature_dim', default=256, type=int, help='Feature dim for last conv')
    parser.add_argument('--batch_size', default=30, type=int, help='Number of data in each mini-batch')
    parser.add_argument('--epoches', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--runs', type=int, default=10, help='number of training times')
    #添加模型预测runs次数
    parser.add_argument('--warmup_epochs', type=int, default=20, help='number of warmup_epochs')
    args = parser.parse_args()
    print("set_model args",args)
    batch_size, epochs= args.batch_size, args.epoches
    # data prepare
    # 在main()函数外部初始化用于存储所有实验结果的列表
    all_aa = []
    all_oa = []
    all_kappa = []
    all_each_acc = []  # 存储每类精度的二维列表
    start = time.time()
    for n in range(0, args.runs):
        print(f"----Now begin the {format(n)} run----")
        labeled_train_loader, _, unlabeled_train_loader, _, _, _, unlabeled_dataset = get_dataloaders(batchsize=batch_size, n=n,dataset=args.dataset)
        _, test_loader, _, TestLabel, TestPatch, pad_width_data, _ = get_dataloaders(batchsize=batch_size, n=n,dataset=args.dataset)

        args.bands = int(TestPatch.shape[1])
        args.num_classes = len(np.unique(TestLabel))
        model = set_model(args)
        #print("model:",model)
        model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], 0.2, last_epoch=-1)
        args.patchsize = 24
        args.threshold = 0.95

        # ------------------- 模型预热 -------------------
        criterion = utils.CrossEntropyLoss()  # 假设损失函数为交叉熵（需与train函数一致）
        model = warmup_model(
            model=model,
            train_loader=labeled_train_loader,
            optimizer=optimizer,
            criterion=criterion,
            warmup_epochs=args.warmup_epochs
        )
        model.cuda()
        print("模型预热完成")
        # 初始化可靠伪标签存储
        reliable_pseudo_indices = []
        reliable_pseudo_labels = []

        # 初始高斯拟合（仅使用有标签数据）
        model.eval()
        correct_losses, incorrect_losses = [], []
        all_losses = []
        all_correct_masks = []  # 新增：存储所有批次的正确掩码

        with torch.no_grad():
            for batch_idx, (x_l, labels_l) in enumerate(labeled_train_loader):
                x_l, labels_l = x_l.cuda(), labels_l.cuda()
                y, _ = model(x_l)
                y_l = y[:args.batch_size]
                cls_loss = criterion(y_l, labels_l)

                if cls_loss.dim() == 0:
                    log_softmax_output = torch.log_softmax(y_l, dim=1)
                    cls_loss = -torch.gather(log_softmax_output, 1, labels_l.unsqueeze(1)).squeeze(1)
                # 处理可能的标量损失
                if cls_loss.dim() == 0:  # 再次检查，以防上面的代码没有正确处理
                    cls_loss = cls_loss.unsqueeze(0)  # 转换为一维张量

                _, predicted = torch.max(y_l.data, 1)
                correct_mask = predicted == labels_l

                # 存储当前批次的损失和掩码
                all_losses.append(cls_loss.cpu().numpy())
                all_correct_masks.append(correct_mask.cpu().numpy())  # 保存掩码

            all_correct_masks = np.concatenate(all_correct_masks)
            all_losses = np.concatenate(all_losses)
            correct_losses = all_losses[np.array(all_correct_masks)].tolist()
            incorrect_losses = all_losses[~np.array(all_correct_masks)].tolist()
        # 打印结果（可选）
        print(f"正确损失数量: {len(correct_losses)}")
        print(f"错误损失数量: {len(incorrect_losses)}")
        # 合并所有批次的损失值并进行全局归一化
        #all_losses = np.concatenate(all_losses)
        if len(all_losses) > 0:
            # 计算统计量
            min_val = np.min(all_losses)
            max_val = np.max(all_losses)

            # 设置最小标准差阈值
            min_std = 1e-4

            if max_val - min_val > 1e-8:
                # 正常归一化
                normalized_losses = (all_losses - min_val) / (max_val - min_val)
            else:
                # 当差异过小时，使用最小标准差进行缩放
                normalized_losses = (all_losses - min_val) / (min_std * 2)
                # 确保结果在[0,1]范围内
                normalized_losses = np.clip(normalized_losses, 0, 1)
        else:
            normalized_losses = np.array([])

        # 合并所有批次的掩码
        if len(all_correct_masks)>0:
            #all_correct_masks = np.concatenate(all_correct_masks)

            # 直接使用全局掩码分割归一化后的损失
            guiyi_correct_losses = normalized_losses[np.array(all_correct_masks)].tolist()
            guiyi_incorrect_losses = normalized_losses[~np.array(all_correct_masks)].tolist()
        else:
            guiyi_correct_losses = []
            guiyi_incorrect_losses = []

        # 初始高斯拟合
        guiyi_all_losses = np.concatenate([
            np.array(guiyi_correct_losses).reshape(-1, 1),
            np.array(guiyi_incorrect_losses).reshape(-1, 1)
        ], axis=0)
        # ema_gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=1e-6)
        # ema_gmm.fit(all_losses)
        ema_gmm = GaussianMixtureWithHistory(
            n_components=2,
            max_iter=50,
            tol=1e-4,
            reg_covar=1e-6,
            random_state=42,
        )
        ema_gmm.fit(guiyi_all_losses)
        # plot_gmm_fit(guiyi_correct_losses, guiyi_incorrect_losses, ema_gmm, 1000)

        # 打印对数似然值变化
        print("迭代过程中的对数似然值变化:")
        for i, ll in enumerate(ema_gmm.log_likelihood_history):
            print(f"迭代 {i + 1}: {ll:.4f}")

        # training loop
        best_acc, best_epoch = 0.0, 0
        selected_label = torch.ones((len(unlabeled_dataset),), dtype=torch.long, ) * -1
        classwise_acc = torch.zeros((args.num_classes,)).cuda()

        label_incorrect_losses = incorrect_losses
        combined_losses = np.array([])
        for epoch in range(1, epochs + 1):
            print('len(correct_losses)',len(correct_losses))
            print('len(incorrect_losses)',len(incorrect_losses))
            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(unlabeled_dataset):
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
            # if epoch==1:
            #     plt.hist(correct_losses, bins=30, density=True, alpha=0.5, label='correct', color='cyan')
            #     plt.hist(incorrect_losses, bins=30, density=True, alpha=0.5, label='incorrect', color='salmon')
            #     plt.xlabel('Loss Value')
            #     plt.ylabel('Frequency')
            #     plt.title('Distribution of Correct vs Incorrect Losses')
            #     plt.legend()
            #     plt.grid(True, linestyle='--', alpha=0.5)
            #     plt.show()
            # 每个epoch更新高斯拟合（加入新的可靠伪标签样本）
            if epoch > 1 and len(reliable_pseudo_indices) > 0:
                # 计算可靠伪标签样本的损失
                pseudo_correct_losses = []
                model.eval()
                with torch.no_grad():
                    for idx, pseudo_label in zip(reliable_pseudo_indices, reliable_pseudo_labels):
                        x_u = unlabeled_dataset[idx][0].unsqueeze(0).cuda()
                        y_u, _ = model(x_u)
                        loss = criterion(y_u, torch.tensor([pseudo_label]).cuda())
                        if loss.dim() == 0:
                            log_softmax_output = torch.log_softmax(y_u, dim=1)
                            loss = -torch.gather(log_softmax_output, 1,
                                                 torch.tensor([pseudo_label]).cuda().unsqueeze(1)).squeeze(1)
                        pseudo_correct_losses.append(loss.item())

                # 更新高斯拟合（合并原有标签损失和新的伪标签损失）
                updated_correct_losses = np.array(correct_losses + pseudo_correct_losses)
                print('len(updated_correct_losses)', len(updated_correct_losses))
                updated_incorrect_losses = np.array(incorrect_losses)
                # if epoch%5==0:
                #     plt.hist(updated_correct_losses, bins=30, density=True, alpha=0.5, label='correct', color='cyan')
                #     plt.hist(updated_incorrect_losses, bins=30, density=True, alpha=0.5, label='incorrect', color='salmon')
                #     plt.xlabel('Loss Value')
                #     plt.ylabel('Frequency')
                #     plt.title('Distribution of Correct vs Incorrect Losses')
                #     plt.legend()
                #     plt.grid(True, linestyle='--', alpha=0.5)
                #     plt.show()

                # 归一化损失
                # 检查数组是否为空，避免除以零
                combined_losses = np.concatenate([updated_correct_losses, updated_incorrect_losses])
                if len(combined_losses) > 0:
                    min_value = combined_losses.min()
                    max_value = combined_losses.max()
                    # 设置最小标准差阈值
                    min_std = 1e-4
                    if max_value - min_value > 1e-8:
                        guiyi_updated_correct_losses = (updated_correct_losses - min_value) / (
                                    max_value - min_value)
                        guiyi_updated_incorrect_losses = (updated_incorrect_losses - min_value) / (
                                    max_value - min_value)
                    else:
                        guiyi_updated_correct_losses = (updated_correct_losses - min_value) / (
                                min_std*2)
                        guiyi_updated_incorrect_losses = (updated_incorrect_losses - min_value) / (
                                min_std*2)
                        # 确保结果在[0,1]范围内（处理可能的数值误差）
                        guiyi_updated_correct_losses = np.clip(guiyi_updated_correct_losses, 0, 1)
                        guiyi_updated_incorrect_losses = np.clip(guiyi_updated_incorrect_losses, 0, 1)

                else:
                    guiyi_updated_correct_losses = np.random.rand(1) * 0.1
                    guiyi_updated_incorrect_losses = np.random.rand(1) * 0.9 + 0.1

                # 重新拟合高斯分布
                updated_all_losses = np.concatenate([
                    guiyi_updated_correct_losses.reshape(-1, 1),
                    guiyi_updated_incorrect_losses.reshape(-1, 1)
                ], axis=0)

                #可视化更新后的高斯拟合 每5个epoch更新一次高斯模型 并显示
                # if epoch % 5 == 0:
                #     #ema_gmm.fit(updated_all_losses)
                #     plot_gmm_fit(guiyi_updated_correct_losses, guiyi_updated_incorrect_losses, ema_gmm, epoch)


            # 调用train()函数，使用当前的ema_gmm筛选伪标签
            model.train()
            loss_x, loss_u, selected_label,  unco_indices, unco_labels, unco_losses= train(
                model=model,
                optim=optimizer,
                lr_schdlr=lr_scheduler,
                args=args,
                selected_label=selected_label,
                classwise_acc=classwise_acc,
                labeled_train_loader=labeled_train_loader,
                unlabeled_train_loader=unlabeled_train_loader,
                ema_gmm=ema_gmm,
                epoch = epoch,
                total_epochs = epochs,
                combined_losses = combined_losses
            )
            # 更新错误样本损失列表
            incorrect_losses = label_incorrect_losses + unco_losses
            # 收集本轮新的可靠伪标签
            new_reliable_mask = selected_label != -1
            new_reliable_indices = torch.where(new_reliable_mask)[0].cpu().numpy().tolist()
            new_reliable_labels = selected_label[new_reliable_mask].cpu().numpy().tolist()


            # 更新可靠伪标签列表
            reliable_pseudo_indices = new_reliable_indices
            reliable_pseudo_labels = new_reliable_labels



            # 测试模型
            test_acc, predictions = test(model, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(model, './results/best_acc_result1.pth')

            print(
                f'Epoch: [{epoch}/{epochs}] | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Test Acc: {test_acc:.2f} | Reliable Pseudo: {len(reliable_pseudo_indices)}')

        print(f'Best test_acc: {best_acc:.2f} at epoch {best_epoch}')

        model = torch.load('./results/best_acc_result1.pth')
        model.eval()

        pred_y = np.empty((len(TestLabel)), dtype='float32')
        number = len(TestLabel) // 100

        for i in range(number):
            temp = TestPatch[i * 100:(i + 1) * 100, :, :, :]
            temp = temp.cuda()
            temp2, _ = model(temp)
            #  _, temp2, _, _, _, = model(temp, temp, temp, temp)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
            del temp, temp2, temp3, _
        # 不足100个的情况
        if (i + 1) * 100 < len(TestLabel):
            temp = TestPatch[(i + 1) * 100:len(TestLabel), :, :, :]
            temp = temp.cuda()
            temp2, _ = model(temp)
            # _, _, _,  _, temp2, _, _, _ = model(temp, temp, temp, temp)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
            del temp, temp2, temp3, _

        # 评价指标
        pred_y = torch.from_numpy(pred_y).long()
        Classes = np.unique(TestLabel)
        EachAcc = np.empty(len(Classes))
        AA = 0.0
        for i in range(len(Classes)):
            cla = Classes[i]
            right = 0
            sum = 0
            for j in range(len(TestLabel)):
                if TestLabel[j] == cla:
                    sum += 1
                if TestLabel[j] == cla and pred_y[j] == cla:
                    right += 1
            EachAcc[i] = right.__float__() / sum.__float__()
            AA += EachAcc[i]
        # 计算并保存当前实验结果
        AA *= 100 / len(Classes)
        results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
        OA = results["Accuracy"]
        Kappa = results["Kappa"]

        # 保存当前实验结果到全局列表
        all_aa.append(AA)
        all_oa.append(OA)
        all_kappa.append(Kappa)
        all_each_acc.append(EachAcc * 100)  # 保存每类精度
        # 打开文件准备写入
        with open('./实验结果1.txt', 'a', encoding='utf-8') as f:
            print('-------------------')
            # 写入每类精度
            f.write('----Now begin the %d run----\n' %(n))
            for i in range(len(EachAcc)):
                f.write('|第%d类精度：' % (i + 1))
                f.write('%.2f|\n' % (EachAcc[i] * 100))
                f.write('-------------------\n')
                print('|第%d类精度：' % (i + 1), '%.2f|' % (EachAcc[i] * 100))
                print('-------------------')


            # 计算并写入其他指标
            results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
            f.write('test accuracy（OA）: %.2f ' % results["Accuracy"])
            # 计算并写入AA
            # AA *= 100 / len(Classes)
            f.write(f'AA : %.2f \n' % AA)
            f.write('Kappa : %.2f \n' % results["Kappa"])
            print('test accuracy（OA）: %.2f ' % results["Accuracy"], 'AA : %.2f ' % AA,
                  'Kappa : %.2f ' % results["Kappa"])

        print("结果已成功写入 实验结果.txt")


        # print('-------------------')
        # for i in range(len(EachAcc)):
        #     print('|第%d类精度：' % (i + 1), '%.2f|' % (EachAcc[i] * 100))
        #     print('-------------------')
        # AA *= 100 / len(Classes)
        #
        # results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
        # print('test accuracy（OA）: %.2f ' % results["Accuracy"], 'AA : %.2f ' % AA, 'Kappa : %.2f ' % results["Kappa"])
        #print('confusion matrix :')
        #print(results["Confusion matrix"])
    end = time.time()
    with open('./实验结果1.txt', 'a', encoding='utf-8') as f:
        f.write('\n===== 10次实验汇总统计 =====\n')

        # 计算总体统计数据
        f.write(f'AA 平均: {np.mean(all_aa):.2f}%, 标准差: {np.std(all_aa):.2f}%\n')
        f.write(f'OA 平均: {np.mean(all_oa):.2f}%, 标准差: {np.std(all_oa):.2f}%\n')
        f.write(f'Kappa 平均: {np.mean(all_kappa):.2f}%, 标准差: {np.std(all_kappa):.2f}%\n')

        # 计算每类精度的统计数据
        all_each_acc = np.array(all_each_acc)
        f.write('\n每类精度统计:\n')
        for i in range(all_each_acc.shape[1]):
            class_mean = np.mean(all_each_acc[:, i])
            class_std = np.std(all_each_acc[:, i])
            f.write(f'| 第{i + 1}类 | 平均: {class_mean:.2f}% | 标准差: {class_std:.2f}% |\n')
        f.write(f'运行时长：{end - start}秒')
        f.write('===========================\n')

    print("所有实验结果及统计数据已成功写入 实验结果.txt")

if __name__ == '__main__':
    # 创建保存图片的文件夹
    save_folder = 'gmm_plots'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    main()



