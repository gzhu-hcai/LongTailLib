"""
ResNet for FedETF - Direct implementation based on FedETF-main/models_dict/resnet.py
Includes ETF classifier components for neural collapse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class BasicBlock(nn.Module):
    """Basic ResNet block - copied from FedETF source"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Proto_Classifier(nn.Module):
    """
    ETF Classifier - copied from FedETF source models_dict/proto_classifiers.py
    Generates fixed simplex ETF structure for neural collapse
    """
    def __init__(self, feat_in, num_classes, device='cuda'):
        super(Proto_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - ((1 / num_classes) * one))
        
        self.register_buffer('proto', M.to(device))

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        return P

    def load_proto(self, proto):
        self.proto = copy.deepcopy(proto)

    def forward(self, label):
        target = self.proto[:, label].T
        return target


class ResNet_cifar_FedETF(nn.Module):
    """
    ResNet for CIFAR with FedETF structure - copied from FedETF source
    Returns (feature, logit, out) for ETF-based learning
    """
    def __init__(self, block, num_blocks, num_classes=10, device='cuda'):
        super(ResNet_cifar_FedETF, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # FedETF components
        self.linear_proto = nn.Linear(64 * block.expansion, num_classes)
        self.proto_classifier = Proto_Classifier(num_classes, num_classes, device=device)
        self.scaling_train = nn.Parameter(torch.tensor(1.0))
        
        # Linear classifier (for baseline methods)
        self.linear_head = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        
        # Proto classifier: generate normalized features
        feature = self.linear_proto(out)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)
        
        # Linear classifier
        logit = self.linear_head(out)
        
        return feature, logit, out


def ResNet20_FedETF(num_classes=10, device='cuda'):
    """
    ResNet20 for FedETF on CIFAR
    Architecture: [3, 3, 3] blocks = 6*3+2 = 20 layers
    Feature dim: 64
    """
    return ResNet_cifar_FedETF(BasicBlock, [3, 3, 3], num_classes=num_classes, device=device)


def ResNet56_FedETF(num_classes=10, device='cuda'):
    """
    ResNet56 for FedETF on CIFAR
    Architecture: [9, 9, 9] blocks = 6*9+2 = 56 layers
    Feature dim: 64
    """
    return ResNet_cifar_FedETF(BasicBlock, [9, 9, 9], num_classes=num_classes, device=device)


# ============================================================================
# ResNet8 and ResNet18 for FedETF
# ============================================================================

class ResNet8_cifar_FedETF(nn.Module):
    """
    ResNet8 for CIFAR with FedETF structure
    Architecture: [1, 1, 1] blocks = 6*1+2 = 8 layers
    Feature dim: 64 (same as ResNet20 for ETF compatibility)
    """
    def __init__(self, num_classes=10, device='cuda', scaling=4):
        super(ResNet8_cifar_FedETF, self).__init__()
        self.in_planes = 16 * scaling
        self.num_classes = num_classes
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, 16 * scaling, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * scaling)
        self.layer1 = self._make_layer(block, 16 * scaling, 1, stride=1)
        self.layer2 = self._make_layer(block, 32 * scaling, 1, stride=2)
        self.layer3 = self._make_layer(block, 64 * scaling, 1, stride=2)

        # FedETF components - feature_dim = 64 * scaling = 256
        feature_dim = 64 * scaling * block.expansion
        self.linear_proto = nn.Linear(feature_dim, num_classes)
        self.proto_classifier = Proto_Classifier(num_classes, num_classes, device=device)
        self.scaling_train = nn.Parameter(torch.tensor(1.0))

        # Linear classifier (for baseline methods)
        self.linear_head = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        # Proto classifier: generate normalized features
        feature = self.linear_proto(out)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)

        # Linear classifier
        logit = self.linear_head(out)

        return feature, logit, out


def ResNet8_FedETF(num_classes=10, device='cuda'):
    """
    ResNet8 for FedETF on CIFAR
    Architecture: [1, 1, 1] blocks = 6*1+2 = 8 layers
    Feature dim: 256 (with scaling=4)
    """
    return ResNet8_cifar_FedETF(num_classes=num_classes, device=device, scaling=4)


class ResNet18_cifar_FedETF(nn.Module):
    """
    ResNet18 for CIFAR with FedETF structure
    Architecture: [2, 2, 2, 2] blocks = 8*2+2 = 18 layers
    Feature dim: 512
    """
    def __init__(self, num_classes=10, device='cuda'):
        super(ResNet18_cifar_FedETF, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        # FedETF components - feature_dim = 512
        feature_dim = 512 * block.expansion
        self.linear_proto = nn.Linear(feature_dim, num_classes)
        self.proto_classifier = Proto_Classifier(num_classes, num_classes, device=device)
        self.scaling_train = nn.Parameter(torch.tensor(1.0))

        # Linear classifier (for baseline methods)
        self.linear_head = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        # Proto classifier: generate normalized features
        feature = self.linear_proto(out)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)

        # Linear classifier
        logit = self.linear_head(out)

        return feature, logit, out


def ResNet18_FedETF(num_classes=10, device='cuda'):
    """
    ResNet18 for FedETF on CIFAR
    Architecture: [2, 2, 2, 2] blocks = 8*2+2 = 18 layers
    Feature dim: 512
    """
    return ResNet18_cifar_FedETF(num_classes=num_classes, device=device)
