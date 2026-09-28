# Resume Screening & Candidate Classification System

A machine learning project that classifies resumes into different job-role categories using TF-IDF text representation and LinearSVC.

## Overview

This project demonstrates an end-to-end text classification workflow:

1. Load resume dataset
2. Remove duplicate resumes
3. Split data into training and testing sets
4. Convert resume text into numerical features using TF-IDF
5. Train a LinearSVC classifier
6. Predict candidate categories
7. Evaluate model performance

## Dataset

The original dataset contained:

- 962 rows
- 2 columns
- 25 candidate categories

After removing duplicate resumes:

- 166 unique resumes

The duplicate removal was performed before the train/test split to avoid the same resume appearing in both training and testing data.

## Machine Learning Workflow

```text
Resume Dataset
      ↓
Remove Duplicate Resumes
      ↓
166 Unique Resumes
      ↓
Train/Test Split
      ↓
TF-IDF Vectorization
      ↓
LinearSVC
      ↓
Predictions
      ↓
Model Evaluation