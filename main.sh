#!/bin/bash

./run.sh

echo "Test set:"
python evaluator/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv
echo "Training set:"
python evaluator/evaluate.py train_predictions.csv data/fashion_mnist_train_labels.csv
