

#  MNIST Classification - Open Ended Lab

The MNIST dataset consists of handwritten digits (0-9) represented as 28x28 grayscale images, flattened into 784-dimensional vectors with pixel intensities ranging from 0 to 255. The dataset is pre-split into training (mnist_train.csv, 60,000 samples) and testing (mnist_test.csv, 10,000 samples) sets. This lab leverages these preprocessed CSV files to train and evaluate classification models, exploring their performance on digit recognition in an open-ended framework.

---

## 🎯 Objective
The goal of this open-ended lab is to experiment with different classification models and compare their performance on the MNIST dataset. The tasks include:
- Understanding the dataset structure.
- Training various machine learning models.
- Evaluating and analyzing their performance.
- Documenting findings with results and improvements.

## 📂 Dataset Description
The dataset consists of handwritten digits (0-9) represented as **28×28 grayscale images**.  
It has been preprocessed as follows:
- Flattened images into a **1D vector of 784 features**.
- Split into **training (60,000 samples) and testing (10,000 samples)**.
- Stored as CSV files (`mnist_train.csv` and `mnist_test.csv`).

## 🔄 Workflow
1. **Load Data:** Read `mnist_train.csv` & `mnist_test.csv`.
2. **Preprocess Data:** Normalize pixel values (0-255) to (0-1).
3. **Train Models:** Logistic Regression, KNN, and Naïve Bayes.
4. **Evaluate Performance:** Use accuracy, confusion matrix, and classification report.
5. **Analyze & Improve:** Compare models and suggest improvements.

## 🚀 Implemented Models

### 1️⃣ Logistic Regression
- Linear model that predicts probabilities using **sigmoid function**.
- Suitable for classification problems.

### 2️⃣ K-Nearest Neighbors (KNN)
- Stores training data and classifies based on **nearest neighbors**.
- Performance depends on the **value of K**.

### 3️⃣ Naïve Bayes
- Probabilistic classifier based on **Bayes’ Theorem**.
- Assumes features are **independent**.

## 📊 Performance Evaluation

### ✅ Accuracy Scores:
| Model              | Accuracy (%) |
|-------------------|-------------|
| Logistic Regression | 92.5%       |
| KNN (K=5)         | 95.1%       |
| Naïve Bayes       | 85.3%       |

### 🔍 Confusion Matrix:
Below is an example of a confusion matrix generated during evaluation.

**Confusion Matrix:**
![alt text](<Confusion Matrix-1.png>)

## 📈 Graphical Analysis
- **Accuracy Comparison:** Bar chart of model performances.
- **Confusion Matrix Heatmaps:** Visualization of classification errors.

## 🔧 Improvements & Future Work
- **Hyperparameter Tuning:** Optimize KNN (`K` value) and Logistic Regression (regularization).
- **Feature Engineering:** Try **PCA (Principal Component Analysis)** for dimensionality reduction.
- **Deep Learning:** Use a **Convolutional Neural Network (CNN)** for better accuracy.

---

## 📜 Conclusion
- **KNN outperformed other models** with the highest accuracy.
- **Logistic Regression performed well** but lacked precision on some digits.
- **Naïve Bayes had the lowest accuracy**, likely due to its independence assumption.

🚀 **Future Work:** By implementing a **deep learning model (CNN)** we improved performance.

