# Neural Network Classification with PyTorch 

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A deep learning project demonstrating how to build, train, and evaluate a neural network from scratch to solve a binary classification problem using PyTorch.

## 📌 Project Overview

Linear models often fail when data cannot be separated by a straight line. This project tackles the classic non-linear classification problem using the `make_circles` toy dataset. By leveraging PyTorch, we construct a feedforward neural network capable of learning complex, non-linear decision boundaries to accurately separate two concentric circles of data points.



## 🚀 Features

* **Custom Data Generation:** Utilizes Scikit-Learn to generate a challenging non-linear dataset (`make_circles`).
* **Data Preprocessing:** Converts raw NumPy arrays into structured PyTorch Tensors and Pandas DataFrames for exploratory data analysis (EDA).
* **Model Architecture:** Implements a custom neural network subclassing `nn.Module` with hidden layers and non-linear activation functions (e.g., ReLU).
* **Training Loop:** Features a robust training loop utilizing Binary Cross Entropy loss (`BCEWithLogitsLoss`) and Stochastic Gradient Descent (SGD) or Adam optimization.
* **Visualization:** Uses Matplotlib to plot the dataset and visualize the model's learned decision boundaries.

## 🛠️ Tech Stack

* **Framework:** [PyTorch](https://pytorch.org/)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/neural-network-classification.git](https://github.com/yourusername/neural-network-classification.git)
   cd neural-network-classification
