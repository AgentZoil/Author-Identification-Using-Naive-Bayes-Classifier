# Author Identification Using Naive Bayes Classifier

This project implements a **Naive Bayes Classifier** to identify the author of emails. The dataset consists of email text labeled by their respective authors: **Sara** (label `0`) and **Chris** (label `1`). The classifier uses text preprocessing, feature extraction, and feature selection techniques to train and test the model.

---

## Overview

The goal of this project is to classify emails based on their authors using the **Gaussian Naive Bayes algorithm**. The project involves:
- Text preprocessing (cleaning, vectorization, feature selection)
- Training and testing the Naive Bayes model
- Evaluating the performance of the model


---

## Project Structure

```plaintext
Author Identification Using Naive Bayes Classifier
├── maildir/                # Contains email dataset files
├── doc2unix.py             # Utility script to convert line endings
├── email_authors.pkl       # Preprocessed author labels (binary)
├── email_preprocess.py     # Handles email preprocessing pipeline
├── nb_author_id.py         # Main script for training and testing the classifier
├── word_data.pkl           # Preprocessed email text data
└── word_data_unix.pkl      # Converted dataset with Unix line endings
├── README.md               # Project documentation
├── startup.py              # Optional script to initialize/run the project
```
## Installation


1. **Clone the Repository:**

   ```bash
   git clone https://github.com/AgentZoil/Author-Identification-Naive-Bayes.git
   cd Author-Identification-Naive-Bayes
   ```

2. **Install Dependencies**



3. **Run the Preprocessing Script:**

   ```bash
   python doc2unix.py
   ```
   
4. **Run the Main Script**

   ```bash
   python nb_author_id.py
   ```


## Project Outcome

- Successfully classified emails into two categories: Sara (0) and Chris (1) using the Gaussian Naive Bayes algorithm.
- Preprocessed text data using TF-IDF Vectorization and feature selection for improved performance.

## Results

### Prediction Output

```
no. of Chris training emails: 7936
no. of Sara training emails: 7884
Training Time: 0.697 s
Predicting Time: 0.113 s
Accuracy: 0.9732650739476678
```




## Technologies Used

- Python 3
- scikit-learn
- joblib
- NumPy
- TF-IDF Vectorization

##
