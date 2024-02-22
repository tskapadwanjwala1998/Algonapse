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

## Copyright

This project is © 2024 Taha Kapadwanjwala. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


