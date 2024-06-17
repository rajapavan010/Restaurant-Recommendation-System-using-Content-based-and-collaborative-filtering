# Restaurant Recommendations and Analysis

This project involves analyzing a Swiggy dataset to extract insights about restaurant ratings, popular cuisines, and geographical patterns. It also employs various machine learning techniques to predict restaurant ratings and provide personalized recommendations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Recommendation Systems](#recommendation-systems)
- [Results](#results)

## Overview
This project utilizes a dataset from Swiggy containing information about restaurants. The primary objectives are to analyze the dataset, build predictive models for restaurant ratings, and develop recommendation systems.

## Features
- **Data Preprocessing and Cleaning**: Handling missing values, feature encoding, and normalization.
- **Exploratory Data Analysis (EDA)**: Visualizing distributions, correlations, and geographic data.
- **Machine Learning Models**: Implementing Logistic Regression, Random Forest, and SVM for classification and regression tasks.
- **Recommendation Systems**: Developing content-based and collaborative filtering recommendation systems.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/restaurant-recommendations.git
    cd restaurant-recommendations
    ```

## Usage
1. Import the necessary libraries:
    ```python
    import pandas as pd
    import numpy as np
    import folium
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from fuzzywuzzy import process
    ```

2. Load the dataset:
    ```python
    df = pd.read_csv('swiggy.csv', encoding='latin1')
    ```

3. Follow the steps in the provided Jupyter notebook or Python scripts to preprocess data, perform EDA, train models, and generate recommendations.

## Data Preprocessing
- **Missing Values**: Address missing values using appropriate imputation techniques.
- **Categorical Encoding**: Encode categorical features using methods such as one-hot encoding or label encoding.
- **Normalization**: Normalize numerical features to ensure they are on a similar scale.
- **Target Transformation**: Transform target variables for use in classification and regression models.

## Exploratory Data Analysis
- **Ratings Distribution**: Visualize the distribution of restaurant ratings.
- **Cuisine Analysis**: Identify and analyze the popularity of various cuisines.
- **Feature Correlation**: Examine correlations between different features.
- **Visualization**: Use heatmaps, bar plots, and other visualization tools to uncover insights.

## Predictive Modeling
- **Model Training**: Train models including Logistic Regression, Random Forest, and Support Vector Machines (SVM).
- **Model Evaluation**: Assess model performance using metrics such as accuracy for classification and RMSE for regression.
- **Data Splitting**: Split the dataset into training and testing sets to evaluate model generalizability.

## Recommendation Systems
- **Content-Based Filtering**: Develop content-based recommendations using TF-IDF vectorization and cosine similarity.
- **Collaborative Filtering**: Implement collaborative filtering recommendations based on user-item interactions and preferences.

## Results
- **Classification Accuracy**: Achieved high accuracy (89%) in classification tasks.
- **Regression Performance**: Obtained a low Root Mean Squared Error (RMSE) of 0.40 in regression tasks.
- **Personalized Recommendations**: Successfully generated personalized restaurant recommendations based on user preferences and historical data.
