import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR, LambdaLR, SequentialLR
from torchmetrics import Accuracy
from config import Config

'''
这是ResNet网络的基础，残差块类，其结构大致为
[Conv1 -> BN1 -> Relu1 -> dropout -> Conv2 -> BN2 -> Relu2]
[->                 Downsample(or not)                  ->]
使用的是预激活结构
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, isChange = False):
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if isChange else 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(p=0.3)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)    
        
        # 下采样层（如果需要）
        self.downsample = nn.Identity()
        if isChange:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        

    # 前向传递
    def forward(self, x):
        ind = self.downsample(x)
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.dropout(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out += ind
        
        return out

'''
这是主类，定义了WRNWRN-40-K模型，具体的K值可以在config中更换
'''
class WRN(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # 捕获模型的全部属性
        self.save_hyperparameters()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 三个阶段，每个阶段6个残差块
        self.stage1 = self._make_stage(16, 16*config.K, num_blocks=6)
        self.stage2 = self._make_stage(16*config.K, 32*config.K, num_blocks=6)
        self.stage3 = self._make_stage(32*config.K, 64*config.K, num_blocks=6)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(64*config.K, config.num_classes)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 评估指标
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
    '''
    这个函数用于创建阶段，一共三个阶段，每个阶段中有6个残差块，其中第一个残差块会进行下采样
    '''
    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        
        # 第一个块可能进行下采样
        if in_channels != out_channels:
            layers.append(BasicBlock(in_channels, out_channels, isChange=True))
        else:
            layers.append(BasicBlock(in_channels, out_channels))
        
        # 剩余的块
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 三个阶段，每个阶段6个基础块
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # 平均池化，展平，全连接
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    '''
    实现了父类中提供的三个计算损失的函数
    '''
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):

        optimizer = SGD(
            self.parameters(), 
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Warmup调度器
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / self.config.warmup_epochs
            if epoch < self.config.warmup_epochs else 1.0
        )
        # 余弦退火调度器（在warmup后开始）
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.T_max,
        )
        # 组合调度器：先warmup，后余弦退火
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs]
        )
        return [optimizer], [scheduler]

    
def on_train_epoch_end(self: pl.LightningModule):
    # 每个epoch结束后打印学习率
    optimizer = self.optimizers()
    
    if isinstance(optimizer, list):
        optimizer = optimizer[0]  # 提取列表中的优化器
    current_lr = optimizer.param_groups[0]['lr']
    
    self.log("learning_rate", current_lr, prog_bar=True, on_step=False, on_epoch=True)