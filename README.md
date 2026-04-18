**#Student Academic Performance Predictor#**

A machine learning model that predicts grades.

**Overview**

This project focuses on Early Intervention Analytics. I developed a machine learning model to predict final student grades ($G3$) based on a variety of demographic, social, and academic factors. The goal is to provide educational institutions with a proactive tool to identify students who may require additional support before the end of the semester. This model serves as an early predictor to help identify students that are likely to underperform or is underperforming.

**Data Insights**
Source: Student Performance Dataset (Kaggle).

Key Features: Academic history (past failures), lifestyle factors (study time, absences, internet access), and family background (guardian status).
The data set was obtained from kaggle:

https://www.kaggle.com/code/ahmedwaelnasef/student-higher-education-ml-algorithms-with-gradio

**Strategic Objectives**
Correlation Analysis: Evaluate the relationship between previous academic results and final outcomes.

Factor Impact Study: Examine how external variables like absences and study habits influence performance.

Predictive Modeling: Construct a regression model using academic, personal, and family-related data.

Feature Importance: Identify the most influential variables to understand what truly drives student success.

Model Performance & Evaluation
**Training Score: 0.89**

**Test Score: 0.12**

Analysis: The significant gap between the training and test scores indicates model overfitting. While the model has effectively "learned" the training data, it currently struggles to generalize to new, unseen data.

Current Iteration: I am currently performing Feature Selection and Hyperparameter Tuning to reduce variance and improve the model's reliability on the test set.

**G3 Model - Train Score: 0.89, Test Score: 0.**12
