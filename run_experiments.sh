#!/bin/bash

# yq 설치가 필요합니다.
# 예: brew install yq (macOS) 또는 apt-get install yq (Ubuntu)

for n in 3 5; do
  for type in A B C; do
    echo "Running experiment with n=${n}, type=${type}"
    # config.yaml 파일의 model.n과 model.type 값을 수정
    yq eval ".model.n = ${n}" -i ./conf/config.yaml
    yq eval ".model.type = \"${type}\"" -i ./conf/config.yaml

    # 학습 스크립트 실행 (예: train.py)
    python3 train.py
  done
done
