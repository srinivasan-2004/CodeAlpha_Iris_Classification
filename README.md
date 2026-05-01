# Iris Flower Classification 

**Task 1 completed for the CodeAlpha Data Science Internship.**

## Project Overview
This project involves building a machine learning model to classify Iris flowers into three species: *setosa*, *versicolor*, and *virginica* based on their botanical measurements. 

## Dataset
The dataset contains four specific measurements (in centimeters) for each flower:
* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

*Note: The dataset's `Id` column was explicitly dropped during preprocessing to prevent data leakage and ensure the model learns purely from the botanical features.*

## Architecture & Technologies
* **Language:** Python
* **Data Handling:** Pandas
* **Machine Learning Library:** Scikit-learn
* **Model:** Random Forest Classifier (`n_estimators=100`)

## Performance
The data was split using an 80/20 ratio for training and testing. The model achieved **100% accuracy** on the unseen test data.

## How to Run
1. Ensure Python is installed.
2. Install required libraries: `pip install pandas scikit-learn`
3. Run the script: `python iris_classifier.py`
