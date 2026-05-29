1. Team Information & Project Title
Team Number: 5
Team Members: Taehun Kim (20221121), Jun Park (20221183), Yuchan Jo (20221374), Sanghun Han (20221403)
Project Title: Lightweight Duplicate Question Detection Using Quora Question Pairs Dataset
2. Dataset Information & Baseline
Dataset Name: Quora Question Pairs
Kaggle URL: https://www.kaggle.com/competitions/quora-question-pairs/overview
Dataset Size & Type: This dataset consists of over 400,000 labeled question pairs from Quora. Each data point includes two question sentences and a binary label, is_duplicate, indicating whether the two questions share the same semantic meaning. This project addresses a natural language processing (NLP)-based supervised binary text-pair classification problem.
Key Columns: id, qid1, qid2, question1, question2, is_duplicate
Note: Since the original competition test labels are not public, we will divide the labeled training data into a training set, validation set, and test set for our experiments.
Baseline Public Codes:
https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
https://www.kaggle.com/code/nicapotato/tf-idf-and-features-logistic-regression
https://www.kaggle.com/code/vabatista/quora-question-pairs-transfer-learning-w-bert
3. Project Description and Approach
Project Overview: This project utilizes the Quora Question Pairs dataset to perform a symmetric semantic search task that determines whether two questions are semantically duplicated. Moving beyond a simple accuracy competition, we assume a realistic high-traffic service environment and conduct analysis of the trade-off between performance (Accuracy) and computational efficiency (Inference Speed and Model Size). our goal is to build an optimization pipeline that makes the model lightweight and fast while maintaining high accuracy.
Analysis and Approach:
Data Exploration (EDA) & Preprocessing: We will analyze dataset characteristics such as class distribution, question length distribution, and word overlap. Subsequently, we will perform basic NLP preprocessing, including missing value handling, lowercasing, tokenization, and punctuation removal. We will extract TF-IDF features from the text and generate similarity-based features such as cosine similarity, word overlap ratio, and length differences.
Traditional Machine Learning Baselines: Adhering to the project guidelines, we will apply traditional machine learning techniques covered in class. Using the extracted features, we will train Logistic Regression, Linear SVM, Naive Bayes, Decision Tree, Random Forest, and Gradient Boosting models. While these models may yield relatively lower accuracy, they are extremely fast and lightweight, serving as the Baseline.
Deep Learning-Based Models: To establish a middle ground in performance and efficiency between traditional ML and state-of-the-art transformer models, we will apply deep learning embeddings. We will generate Word2Vec or GloVe dense embeddings and train a Siamese LSTM network to capture contextual information, establishing a deep learning performance baseline.
Transformer-Based Model Experiments: We will train a Cross-Encoder (BERT-based) that performs attention mechanisms on both sentences simultaneously. As this model understands context most profoundly, it will set the "accuracy upper bound" for the project (expected ~90.7%). However, we will explicitly quantify its massive parameter size, high computational cost, and slow inference speed to highlight the realistic limitations of deep learning.
Optimization & Efficiency Analysis: To find the most efficient architecture compared to the heavy transformer model, we will introduce two optimization pipelines:
Structural Optimization (Bi-Encoder & Knowledge Distillation): We will adopt a Bi-Encoder (e.g., Sentence-BERT) architecture that encodes sentences independently, dramatically increasing processing speed. To minimize performance degradation, we will apply Knowledge Distillation, where the lightweight Bi-Encoder learns from the prediction probabilities of the heavy Cross-Encoder.
Dynamic Quantization & Feature Tuning: We will compress the model size to approximately 1/4 by applying dynamic quantization, which converts the model's weights from floats to integers. Concurrently, we will experiment with minimizing the complexity of traditional models by adjusting the number of TF-IDF features (5K~50K), applying L1/L2 regularization, and tuning maximum tree depths.
Final Benchmarking: For the final evaluation, we will expand our metrics beyond Log Loss, Accuracy, and F1-score to include Model Parameter Size, Training Time, Inference Time (Latency), and Memory Usage. This will quantitatively and visually demonstrate that our lightweight model achieves speeds comparable to traditional ML while approaching the accuracy of the Cross-Encoder.
4. Roles of Team Members
Taehun Kim: Overall project management and data preprocessing (TF-IDF, etc.) execution. Responsible for building the fastest traditional machine learning baselines (Logistic Regression, SVM, etc.).
Jun Park: Generating dense embeddings (Word2Vec/GloVe), training the Siamese LSTM deep learning model, and conducting K-Means clustering visual analysis.
Yuchan Jo: Training the heavy Cross-Encoder (BERT) to derive maximum accuracy, and profiling the chronic deep learning issues by collecting data on slow inference speeds and computational overhead.
Sanghun Han: Implementing the optimization pipeline (Bi-Encoder introduction and Quantization) to compress the model size by 1/4. Responsible for the comprehensive visual benchmarking of final performance (Accuracy) versus efficiency (Speed/Size/Memory).

