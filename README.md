# Restaurant Recommendations and Analysis

This project focuses on analyzing a dataset from Swiggy to uncover insights about restaurant ratings, popular cuisines, and geographical distributions. It also implements various machine learning models to predict restaurant ratings and provide personalized recommendations.

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
- [Contributing](#contributing)
- [License](#license)

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
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
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
- Handle missing values and encode categorical features.
- Normalize numerical features.
- Transform target variables for classification and regression tasks.

## Exploratory Data Analysis
- Visualize the distribution of aggregate ratings.
- Analyze the popularity of different cuisines.
- Explore correlations between features.
- Create heatmaps and bar plots to visualize insights.

## Machine Learning Models
- Train Logistic Regression, Random Forest, and SVM models.
- Evaluate models using accuracy and RMSE metrics.
- Split data into training and testing sets to validate models.

## Recommendation Systems
- Implement content-based recommendations using TF-IDF vectorization and cosine similarity.
- Develop collaborative filtering recommendations based on user-item interactions.

## Results
- Achieved high accuracy (89%) in classification tasks.
- Obtained low RMSE (0.40) in regression tasks.
- Generated personalized restaurant recommendations based on user preferences.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
