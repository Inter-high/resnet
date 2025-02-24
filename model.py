import torch
import torch.nn as nn


class PlainBlock(nn.Module):
    """
    CIFAR10 실험을 위한 plain block:
    두 개의 3x3 conv → BatchNorm → ReLU (순차적으로 적용)
    residual shortcut 없이 단순히 순차 연산만 수행합니다.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out


class PlainNet(nn.Module):
    """
    CIFAR10용 plain network.
    - 첫 번째 층: 3x3 conv (16 filters)
    - 이후 세 그룹(layer)을 구성:
        * layer1: 2n blocks, 16 filters, feature map 크기 32x32
        * layer2: 2n blocks, 32 filters, 첫 블록에서 stride=2 적용 (16x16)
        * layer3: 2n blocks, 64 filters, 첫 블록에서 stride=2 적용 (8x8)
    - 전역 평균 풀링 및 10-way fc layer로 분류.
    총 weighted layer 수: 6n+2.
    """
    def __init__(self, block, n, num_classes=10):
        super(PlainNet, self).__init__()
        self.in_planes = 16
        
        # 첫 번째 층: 3x3 conv, 16 filters
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 세 그룹: 각 그룹에 대해 2n개의 block을 쌓음.
        self.layer1 = self._make_layer(block, 16, 2 * n, stride=1)   # 32x32
        self.layer2 = self._make_layer(block, 32, 2 * n, stride=2)   # 16x16
        self.layer3 = self._make_layer(block, 64, 2 * n, stride=2)   # 8x8
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        # 첫 블록은 stride를 이용해 다운샘플링 (필요한 경우)
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        # 나머지 block은 stride=1
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 입력: 32x32, 픽셀별 평균(mean) 제거되어 있다고 가정
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class ResidualBlock(nn.Module):
    """
    CIFAR10 실험을 위한 Residual block.
    두 개의 3×3 conv → BatchNorm → ReLU를 적용한 후, shortcut 연결을 통해 입력을 더합니다.
    
    shortcut_type 옵션:
        - "A": 차원 증가 시 zero-padding 숏컷을 사용하며, 모든 숏컷은 파라미터 없는 항등 매핑.
        - "B": 차원 증가 시에만 projection 숏컷(1×1 conv + BN)을 사용하고, 나머지는 항등 매핑.
        - "C": 모든 숏컷을 projection 방식으로 설정.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut_type="B"):
        super(ResidualBlock, self).__init__()
        self.shortcut_type = shortcut_type
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = planes * self.expansion

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if shortcut_type == "C":
            # 옵션 C: 모든 숏컷을 projection 방식으로.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_planes)
            )
        elif shortcut_type == "B":
            # 옵션 B: 차원 증가 시에만 projection, 그렇지 않으면 항등 매핑.
            if stride != 1 or in_planes != self.out_planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.out_planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.out_planes)
                )
            else:
                self.shortcut = nn.Identity()
        elif shortcut_type == "A":
            # 옵션 A: 파라미터 없는 zero-padding 숏컷.
            # 차원 증가(공간적 downsampling 혹은 채널 수 차이)가 필요하면 forward에서 처리.
            if stride != 1 or in_planes != self.out_planes:
                self.need_pad = True
            else:
                self.need_pad = False
                self.shortcut = nn.Identity()
        else:
            raise ValueError("Invalid shortcut_type. Choose from 'A', 'B', or 'C'.")

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut_type in ["B", "C"]:
            shortcut = self.shortcut(x)
        elif self.shortcut_type == "A":
            if self.need_pad:
                # 만약 stride가 1이 아니라면, 공간 downsampling: x의 각 채널을 stride 간격으로 sampling.
                if self.stride != 1:
                    x = x[:, :, ::self.stride, ::self.stride]
                # 채널 차이가 있다면, 부족한 채널만큼 0으로 pad (채널 dimension에서 뒤쪽에 추가).
                ch_pad = self.out_planes - self.in_planes
                padding = torch.zeros(x.size(0), ch_pad, x.size(2), x.size(3),
                                      device=x.device, dtype=x.dtype)
                shortcut = torch.cat([x, padding], dim=1)
            else:
                shortcut = x
        else:
            shortcut = x  # 기본 항등 매핑
        
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    CIFAR10용 ResNet.
    - 첫 번째 층: 3×3 conv (16 filters)
    - 세 그룹(layer): 각 그룹에 2n개의 block을 쌓음
        * layer1: 16 filters, 32×32 feature map
        * layer2: 32 filters, 첫 블록에서 stride=2 적용 (16×16)
        * layer3: 64 filters, 첫 블록에서 stride=2 적용 (8×8)
    - 전역 평균 풀링 후 10-way fc layer.
    총 weighted layer 수: 6n+2.
    
    shortcut_type: "A", "B", or "C" (default는 "B")
    """
    def __init__(self, block, n, num_classes=10, shortcut_type="B"):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, 2 * n, stride=1, shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(block, 32, 2 * n, stride=2, shortcut_type=shortcut_type)
        self.layer3 = self._make_layer(block, 64, 2 * n, stride=2, shortcut_type=shortcut_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, shortcut_type):
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut_type))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut_type=shortcut_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

def get_plain_network(n, num_classes=10):
    """
    CIFAR10 plain network 생성 함수.
    입력 n에 따라 총 층 수는 6n+2가 됩니다.
    예) n=3 -> 20-layer, n=9 -> 56-layer network
    """
    return PlainNet(PlainBlock, n, num_classes=num_classes)


def get_resnet(n, num_classes=10, shortcut_type="B"):
    """
    CIFAR10용 ResNet 생성 함수.
    입력 n에 따라 총 층 수는 6n+2가 됩니다.
    예: n=3 -> 20-layer, n=9 -> 56-layer network.
    
    shortcut_type: "A", "B", or "C"
    """
    return ResNet(ResidualBlock, n, num_classes=num_classes, shortcut_type=shortcut_type)
