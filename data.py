import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split


def get_transforms():
    # CIFAR10의 통상적인 평균과 표준편차 값
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2023, 0.1994, 0.2010)

    # 학습 시 적용할 transform: 4픽셀 패딩, 32x32 랜덤 크롭, 수평 뒤집기, 텐서 변환, 정규화(픽셀평균 제거)
    train_transform = transforms.Compose([
        transforms.Pad(4),                # 모든 측면에 4픽셀 패딩
        transforms.RandomCrop(32),        # 32x32 랜덤 크롭
        transforms.RandomHorizontalFlip(),# 수평 뒤집기
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    # 검증 및 테스트 시 적용할 transform: augmentation 없이 단순 ToTensor 및 정규화
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    return train_transform, test_transform

def create_train_valid_split(seed, train_dataset, valid_dataset):
    num_samples = len(train_dataset)  # 일반적으로 50000
    train_size = int(0.9 * num_samples)
    valid_size = num_samples - train_size

    # torch.utils.data.random_split을 사용해 인덱스 분할을 위한 더미 리스트를 분할
    dummy_dataset = list(range(num_samples))
    generator = torch.Generator().manual_seed(seed)  # 재현성을 위한 시드 설정
    train_idx_dataset, valid_idx_dataset = random_split(dummy_dataset, [train_size, valid_size], generator=generator)

    # random_split은 Subset 객체를 반환하므로, 내부 인덱스 리스트를 추출합니다.
    train_indices = list(train_idx_dataset)
    valid_indices = list(valid_idx_dataset)

    # 각각의 인덱스를 이용해 Subset 객체 생성 (각각 다른 transform 적용)
    train_dataset = Subset(train_dataset, train_indices)
    valid_dataset = Subset(valid_dataset, valid_indices)
    
    return train_dataset, valid_dataset

def get_datasets(seed, data_dir):
    train_transform, test_transform = get_transforms()

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    valid_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
    train_dataset, valid_dataset = create_train_valid_split(seed, train_dataset, valid_dataset)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader
    