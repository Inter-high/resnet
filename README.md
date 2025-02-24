# ResNet

This repository is designed to verify and evaluate the research on residual connections presented in the "Deep Residual Learning for Image Recognition" paper.

## Key Summary about ResNet

- Plain networks (without residual connections) tend to exhibit increasing error as the network depth increases.
- Various solutions had been proposed to address this issue; this work investigates a method that resolves the problem without increasing the number of parameters.
- The effectiveness of residual learning is demonstrated by achieving first place not only in ILSVRC but also in the COCO 2015 challenge, proving its general applicability.
- Even with the same number of parameters, networks with residual connections converge faster than their plain counterparts.

## Environment

### Software
- **Host OS**: Windows 11
- **CUDA**: 12.4
- **Docker**: Ubuntu 22.04
- **Python**: 3.10.12
- **Libraries**: See `requirements.txt`

### Hardware
- **CPU**: AMD Ryzen 5 7500F 6-Core Processor
- **GPU**: RTX 4070 Ti Super
- **RAM**: 32GB

## Dataset & Augmentation

The CIFAR-10 dataset is used with data augmentation applied exactly as described in the original paper:
- **Training:**
  - Images are of size 32×32.
  - Mean pixel subtraction is performed.
  - A 4-pixel padding is applied on all sides of each image.
  - A random 32×32 crop is sampled from the padded image or a horizontal flip is applied.
- **Testing:**
  - The original 32×32 image is evaluated in a single view without any augmentation.

## Training Environment

The training process follows the settings described in the paper:
- **Training Settings:**
  - **Weight Decay**: 0.0001
  - **Momentum**: 0.9
  - **Weight Initialization**: Using the method from [13]
  - **Batch Normalization (BN)**: Applied as described in [16]
  - **Dropout**: Not used
  - **Minibatch Size**: 128
  - Training is conducted using 2 GPUs.
- **Learning Rate Schedule:**
  - Initial learning rate: 0.1
  - The learning rate is reduced by a factor of 10 at 32,000 and 48,000 iterations.
  - Training terminates at 64,000 iterations.
  - This schedule is based on a train/validation split of 45k/5k images.

## Verification List

1. Verify that a plain network (without residual connections) exhibits increased error as the network depth increases.
2. Verify that adding residual connections leads to reduced error with increasing network depth.
3. Verify that networks with residual connections converge faster than traditional plain networks.

## Results
- 집가서 작성

## How to Run the Experiment
- 집가서 작성(train.py 및 config.yaml 변경을 통해 plain 네트워크도 선택가능하게끔 수정필요)