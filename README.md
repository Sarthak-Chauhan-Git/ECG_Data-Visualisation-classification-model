# Enhanced ECG Visualization & Arrhythmia Diagnosis using DNN and ML with test accuracy of 99.88%
This project presents an integrated framework for enhanced ECG signal visualization and accurate arrhythmia diagnosis using Deep Neural Networks and advanced Machine Learning techniques.
Leveraging well-established datasets like the MIT-BIH Arrhythmia Database and the St. Petersburg INCART 12-lead Arrhythmia Database, this framework is designed to not only improve the interpretability of ECG signals but also provide reliable real-time diagnostic capabilities.

🔍 Key Highlights:
A. Exploratory Data Analysis & Visualization
Comprehensive EDA techniques were used to understand and preprocess the ECG signals. 
This includes:
1. Scatterplots
2. Boxplots
3. Pairplots
4. Correlation Heatmaps
5. Time-Frequency Spectrograms

B. Feature Extraction & Dimensionality Reduction
Extraction of relevant features from high-dimensional ECG data.
Techniques like PCA, t-SNE, and UMAP were used to reduce complexity while preserving critical diagnostic information.

C. Machine Learning Models
A wide range of ML models were benchmarked for classification performance:
1. Random Forest
2. Extra Trees
3. Gradient Boosting
4. HistGradientBoosting
Automated model selection and optimization were facilitated using the LazyClassifier framework.

D. Deep Learning Architecture
A custom-built 1D Convolutional Neural Network (CNN) enhanced with Attention Mechanisms was developed to learn temporal patterns effectively.

E. Performance Metrics
Evaluated using:
1. Classification Accuracy
2. Mean Squared Error (MSE)
3. R² Score
Achieved an exceptional test accuracy of 99.88%, outperforming traditional methods.
