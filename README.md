# Fashion-MNIST Solver - Deep Learning from Scratch

  - This project implements a neural network in C++ without the use of advanced libraries or frameworks.
  - The neural net solves the Fashion-MNIST dataset using a backpropagation
    algorithm.

## Compile & Run
  - The project contains a runnable script called `run.sh`, which compiles and executes the code 
    (and exports the results).
  - The executable outputs two files to the root project directory:
     - `train_predictions.csv` - network predictions for the train set.
     - `test_predictions.csv`  - network predictions for the test set.
    - Format:
       - One prediction per line.
       - Prediction for i-th input vector (ordered by the input .csv file) 
         is on the i-th line in the associated output file.
       - Each prediction is a single integer 0 - 9.

## Data Set
Fashion MNIST (https://arxiv.org/pdf/1708.07747.pdf) - a modern version of a
well-known MNIST (http://yann.lecun.com/exdb/mnist/). It is a dataset of
Zalando's article images ‒ consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes. The dataset is in CSV format. There
are four data files included:  
 - `fashion_mnist_train_vectors.csv`   - training input vectors
 - `fashion_mnist_test_vectors.csv`    - testing input vectors
 - `fashion_mnist_train_labels.csv`    - training labels
 - `fashion_mnist_test_labels.csv`     - testing labels
