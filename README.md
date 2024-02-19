 # Algonapse

Algonapse is a Flask web application that analyzes the security of input text using multiple sentiment analysis models and a trained model B.

## Project Overview
Algonapse allows users to enter text and choose a sentiment analysis model for initial analysis. It then gathers predictions from all models and retrains Model B with the new data. Model B then predicts the sentiment and calculates the difference from the initial model, providing a security recommendation.

## Features
- Home page listing available sentiment analysis models
- Enter text and choose initial model A for analysis
- Gather predictions from all models and retrain Model B
- Model B predicts sentiment and compares to initial model
- View Model B prediction and security recommendation

## Installation
Clone the repo, install dependencies with pip, and run with Flask.

## Dependencies
- Flask
- HuggingFace Transformer API
- Scikit-learn (for Model B)
- Textattack library for text augmentation

