import random
import torch
import matplotlib.pyplot as plt
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """
    Set seed for reproducibility across different modules.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    random.seed(seed)  # Python built-in random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed

    torch.backends.cudnn.deterministic = True  # Ensures deterministic execution
    torch.backends.cudnn.benchmark = False  # Disable if model structure is not fixed


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params


def plot_compare_loss(data, labels, save_path, title='Loss Comparison'):
    """
    여러 실험의 Loss 데이터를 비교하는 함수.
    
    Args:
        data (list): 각 실험의 Loss 데이터 (리스트 또는 배열)의 리스트.
        labels (list): 각 실험에 대한 라벨 문자열의 리스트.
        save_path (str): 저장할 파일 경로 (예: 'loss_comparison.png').
        title (str): 그래프 제목 (기본값: 'Loss Comparison').
    
    Raises:
        ValueError: data와 labels의 길이가 다를 경우.
    """
    if len(data) != len(labels):
        raise ValueError("data와 labels의 길이는 같아야 합니다.")
        
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(data):
        epochs = range(1, len(d) + 1)
        # 자동 기본 색상 사용 (예: 'C0', 'C1', 'C2', ...)
        plt.plot(epochs, d, label=labels[i], linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


def plot_compare_acc(data, labels, save_path, title='Accuracy Comparison'):
    """
    여러 실험의 Accuracy 데이터를 비교하는 함수.
    
    Args:
        data (list): 각 실험의 Accuracy 데이터 (리스트 또는 배열)의 리스트.
        labels (list): 각 실험에 대한 라벨 문자열의 리스트.
        save_path (str): 저장할 파일 경로 (예: 'accuracy_comparison.png').
        title (str): 그래프 제목 (기본값: 'Accuracy Comparison').
    
    Raises:
        ValueError: data와 labels의 길이가 다를 경우.
    """
    if len(data) != len(labels):
        raise ValueError("data와 labels의 길이는 같아야 합니다.")
    
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(data):
        epochs = range(1, len(d) + 1)
        plt.plot(epochs, d, label=labels[i], linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


def plot_compare_error(errors, labels, save_path, title='Error Comparison'):
    """
    여러 실험의 단일 Error 값을 비교하는 함수 (막대그래프).
    
    Args:
        errors (list): 각 실험의 Error 값 (단일 값)의 리스트.
        labels (list): 각 실험에 대한 라벨 문자열의 리스트.
        save_path (str): 저장할 파일 경로 (예: 'error_comparison.png').
        title (str): 그래프 제목 (기본값: 'Error Comparison').
    
    Raises:
        ValueError: errors와 labels의 길이가 다를 경우.
    """
    if len(errors) != len(labels):
        raise ValueError("errors와 labels의 길이는 같아야 합니다.")
    
    n = len(errors)
    x = np.arange(n)
    width = 0.5
    # 기본 색상: 'C0', 'C1', ... 자동 지정
    colors = [f'C{i}' for i in range(n)]
    
    plt.figure(figsize=(8, 6))
    plt.bar(x, errors, width, color=colors)
    plt.xticks(x, labels)
    plt.ylabel('Error')
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()
