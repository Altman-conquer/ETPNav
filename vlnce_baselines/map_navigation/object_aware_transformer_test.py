import os
import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

classes = ['appliances', 'bathtub', 'beam', 'bed', 'blinds', 'board_panel', 'cabinet', 'chair', 'chest_of_drawers',
           'clothes', 'column', 'counter', 'curtain', 'cushion', 'door', 'fireplace', 'furniture', 'gym_equipment',
           'lighting', 'mirror', 'misc', 'objects', 'picture', 'plant', 'railing', 'seating', 'shelving', 'shower',
           'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'towel', 'tv_monitor', 'void', 'wall', 'window']


class ObjectAwareTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6, max_seq_len=100, num_classes=39):
        super(ObjectAwareTransformer, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len + 1  # +1 for target_cls token
        self.num_classes = num_classes

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # target_cls token投影层
        self.target_cls_projection = nn.Linear(1, d_model)  # 将target_cls_int映射到d_model

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(self.max_seq_len, d_model))

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        self.output_projection = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target_cls_int, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        # target_cls_int shape: (batch_size,)
        batch_size, seq_len, _ = x.shape

        # 投影输入序列
        x = self.input_projection(x)  # [B, L, d_model]

        # 创建target_cls token
        target_cls_token = self.target_cls_projection(target_cls_int.unsqueeze(-1).float())  # [B, d_model]
        target_cls_token = target_cls_token.unsqueeze(1)  # [B, 1, d_model]

        # 拼接target_cls token到序列开头
        x = torch.cat([target_cls_token, x], dim=1)  # [B, L+1, d_model]

        # 添加位置编码
        actual_seq_len = seq_len + 1
        x = x + self.pos_encoding[:actual_seq_len].unsqueeze(0)

        # 处理mask：为target_cls token添加False（不mask），其他位置保持原样
        if mask is not None:
            # 在mask前面添加False（target_cls token不被mask）
            target_cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([target_cls_mask, mask], dim=1)

        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 只对原始序列部分（不包括target_cls token）进行输出投影
        x = x[:, 1:, :]  # 去掉第一个target_cls token
        output = self.output_projection(x)
        output = self.sigmoid(output).squeeze(-1)

        return output


class ObjectDataset(Dataset):
    def __init__(self, dataset, max_seq_len=100):
        self.dataset = dataset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 现在x_data不再包含target_cls_int，因为它会作为单独的token
        x_data = item['x']
        y_data = item['y']

        # 处理向后兼容性 - 检查是否存在target_cls_int字段
        if 'target_cls_int' in item:
            target_cls_int = item['target_cls_int']
        else:
            # 如果是旧格式数据，从y中推断target_cls_int
            # 找到y中为1的位置对应的类别
            target_cls_int = 0  # 默认值
            for i, label in enumerate(y_data):
                if label == 1:
                    # 从x_data中获取对应的类别
                    if i < len(x_data):
                        for cls_idx in range(len(classes)):
                            if x_data[i][cls_idx] == 1:
                                target_cls_int = cls_idx
                                break
                    break

        # 获取原始序列长度
        original_len = len(y_data)

        # 填充到固定长度
        while len(x_data) < self.max_seq_len:
            # 创建padding特征：不再包含target_cls_int
            padding_feature = [0] * len(x_data[0]) if x_data else [0] * (len(classes) + 1 + 4)
            x_data.append(padding_feature)

        while len(y_data) < self.max_seq_len:
            y_data.append(0)

        # 截断到最大长度
        x_data = x_data[:self.max_seq_len]
        y_data = y_data[:self.max_seq_len]

        x = torch.tensor(x_data, dtype=torch.float32)
        y = torch.tensor(y_data, dtype=torch.float32)
        target_cls = torch.tensor(target_cls_int, dtype=torch.long)

        # 创建padding mask - True表示需要mask的位置
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        if original_len < self.max_seq_len:
            mask[original_len:] = True

        return x, y, target_cls, mask


def collate_fn(batch):
    """自定义collate函数，处理变长序列"""
    batch_x, batch_y, batch_target_cls, batch_mask = zip(*batch)

    # 所有张量现在应该有相同的形状
    batch_x = torch.stack(batch_x, dim=0)
    batch_y = torch.stack(batch_y, dim=0)
    batch_target_cls = torch.stack(batch_target_cls, dim=0)
    batch_mask = torch.stack(batch_mask, dim=0)

    return batch_x, batch_y, batch_target_cls, batch_mask


def setup_distributed(rank, world_size):
    """设置分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练"""
    dist.destroy_process_group()

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, use_multi_gpu=True, distributed=False, rank=0):
    # 设备配置
    if distributed:
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
    elif use_multi_gpu and torch.cuda.device_count() > 1:
        device = torch.device('cuda')
        model = model.to(device)
        model = nn.DataParallel(model)
        print(f"使用 {torch.cuda.device_count()} 张GPU进行训练")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"使用设备: {device}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 分布式训练需要设置epoch
        if distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # 训练
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_x, batch_y, batch_target_cls, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_target_cls = batch_target_cls.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, batch_target_cls, batch_mask)

            # 只计算非padding位置的损失
            loss_mask = ~batch_mask
            loss = criterion(outputs[loss_mask], batch_y[loss_mask])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_target_cls, batch_mask in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_target_cls = batch_target_cls.to(device)
                batch_mask = batch_mask.to(device)

                outputs = model(batch_x, batch_target_cls, batch_mask)
                loss_mask = ~batch_mask
                loss = criterion(outputs[loss_mask], batch_y[loss_mask])

                val_loss += loss.item()
                val_batches += 1

        train_loss /= train_batches
        val_loss /= val_batches

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 只在主进程或单GPU时保存模型
            if not distributed or rank == 0:
                # 如果是DataParallel，需要保存module.state_dict()
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, 'tmp/best_model.pth')

        if (not distributed or rank == 0) and epoch % 1 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, use_multi_gpu=True, distributed=False, rank=0):
    # 设备配置
    if distributed:
        device = torch.device(f'cuda:{rank}')
    elif use_multi_gpu and torch.cuda.device_count() > 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, batch_target_cls, batch_mask in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_target_cls = batch_target_cls.to(device)
            batch_mask = batch_mask.to(device)

            outputs = model(batch_x, batch_target_cls, batch_mask)
            predictions = (outputs > 0.5).float()

            # 只保留非padding位置的预测和目标
            loss_mask = ~batch_mask
            predictions_valid = predictions[loss_mask].cpu().numpy()
            targets_valid = batch_y[loss_mask].cpu().numpy()

            all_predictions.extend(predictions_valid)
            all_targets.extend(targets_valid)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('tmp/training_curves.png')
    plt.show()


def random_data():
    cls = random.choice(classes)
    result = {'conf': random.random(), 'cls_int': classes.index(cls),
              'xyxy': [random.uniform(0, 100) for _ in range(4)]}
    return result


def data_to_feature(data):
    """将数据转换为特征，不再包含target_cls_int"""
    features = []
    for item in data:
        feature = [0] * len(classes)
        feature[item['cls_int']] = 1  # one-hot编码
        feature.append(item['conf'])  # 置信度
        feature.extend(item['xyxy'])  # 边界框坐标
        features.append(feature)

    return features


def generate_dataset(num_samples):
    dataset_path = 'tmp/random_dataset.pkl'

    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    os.makedirs('tmp', exist_ok=True)
    dataset = []

    for _ in range(num_samples):
        seq = []
        seq_len = random.randint(5, 50)  # 限制序列长度范围
        for _ in range(seq_len):
            data = random_data()
            seq.append(data)

        target_cls = random.choice(classes)
        target_cls_int = classes.index(target_cls)

        # target_cls_int现在单独存储，不作为特征
        feature = data_to_feature(seq)

        dataset.append({
            'x': feature,
            'y': [1 if i['cls_int'] == target_cls_int else 0 for i in seq],
            'target_cls_int': target_cls_int
        })

    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


def run_distributed_training(rank, world_size, train_dataset, val_dataset, test_dataset):
    """分布式训练函数"""
    setup_distributed(rank, world_size)

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # 创建数据加载器
    train_loader = DataLoader(
        ObjectDataset(train_dataset),
        batch_size=64,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        ObjectDataset(val_dataset),
        batch_size=64,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # 创建模型
    input_dim = len(classes) + 1 + 4
    model = ObjectAwareTransformer(input_dim=input_dim, num_classes=len(classes))

    if rank == 0:
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print("开始分布式训练...")

    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=10000, distributed=True, rank=rank
    )

    # 只在主进程进行评估和绘图
    if rank == 0:
        # 绘制训练曲线
        plot_training_curves(train_losses, val_losses)

        # 加载最佳模型
        model.load_state_dict(torch.load('tmp/best_model.pth'))

        # 评估模型
        test_loader = DataLoader(
            ObjectDataset(test_dataset),
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn
        )

        print("评估模型...")
        metrics = evaluate_model(model, test_loader, distributed=True, rank=rank)

        print("测试集结果:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")

    cleanup_distributed()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 删除旧的数据集文件以确保重新生成
    # dataset_path = 'tmp/random_dataset.pkl'
    # if os.path.exists(dataset_path):
    #     print("删除旧数据集文件...")
    #     os.remove(dataset_path)

    # 生成数据集
    print("生成数据集...")
    dataset = generate_dataset(num_samples=10000)

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    # 选择训练模式
    use_distributed = torch.cuda.device_count() > 1 and torch.cuda.is_available()
    use_multi_gpu = torch.cuda.device_count() > 1 and torch.cuda.is_available()

    print(f"可用GPU数量: {torch.cuda.device_count()}")

    if use_distributed and True:  # 设置为True启用分布式训练
        print("使用分布式训练...")
        world_size = torch.cuda.device_count()
        mp.spawn(
            run_distributed_training,
            args=(world_size, train_dataset, val_dataset, test_dataset),
            nprocs=world_size,
            join=True
        )
    else:
        # 使用DataParallel或单GPU训练
        # 创建数据加载器
        train_loader = DataLoader(
            ObjectDataset(train_dataset),
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            ObjectDataset(val_dataset),
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            ObjectDataset(test_dataset),
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        # 创建模型
        input_dim = len(classes) + 1 + 4
        model = ObjectAwareTransformer(input_dim=input_dim, num_classes=len(classes))

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 训练模型
        print("开始训练...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            num_epochs=10000, use_multi_gpu=use_multi_gpu
        )

        # 绘制训练曲线
        plot_training_curves(train_losses, val_losses)

        # 加载最佳模型
        model.load_state_dict(torch.load('tmp/best_model.pth'))

        # 评估模型
        print("评估模型...")
        metrics = evaluate_model(model, test_loader, use_multi_gpu=use_multi_gpu)

        print("测试集结果:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")


if __name__ == '__main__':
    main()