#!/bin/bash

for n in 3 5; do
  for type in A B C; do
    echo "Running experiment with n=${n}, type=${type}"
    # Modify model.n and model.type values in the config.yaml file
    yq eval ".model.n = ${n}" -i ./conf/config.yaml
    yq eval ".model.type = \"${type}\"" -i ./conf/config.yaml
    yq eval ".model.resnet = true" -i ./conf/config.yaml

    # Run the training script (e.g., train.py)
    python3 train.py
  done
done

# Additional: For n values 3 and 5, set model.resnet to False and run
for n in 3 5; do
  echo "Running experiment with n=${n}, model.resnet=False"
  # Modify model.n and model.resnet values in the config.yaml file
  yq eval ".model.n = ${n}" -i ./conf/config.yaml
  yq eval ".model.resnet = false" -i ./conf/config.yaml

  # Run the training script (e.g., train.py)
  python3 train.py
done
