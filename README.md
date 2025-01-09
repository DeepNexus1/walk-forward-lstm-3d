# Deep Nexus Research Repository for Categorical LSTM with Walk-Forward Optimization

## Project Overview

This repository contains two LSTM Sequence-to-Sequence model examples with rolling walk-forward optimization. The walk-forward optimization is commonly used with financial time-series data to address market regime-change. 

Each model uses categorical outputs and can be highly customized to provide signals for binary, ternary, or other ranking outputs.

One of the models is adapted for a portfolio of assets and utilizes an Embedding Layer to learn relationships between assets.

## Core Components

- **Prediction Models**:
  - Implemented using TensorFlow and Keras
  - Long Short-Term Memory (LSTM) neural network architecture
  - Categorical Prediction Output

## Technical Objectives

- Ensuring 3-D arrays are properly structured for walk-forward training
- Flexible structure for a variety of categorical outputs
- Capacity for feature creation and scaling at sequence generation
- Out-of-Sample Performance Evaluation Metrics
- Accommodating multiple assets in one predictive model
- Implement Embedding Layer in Portfolio model

 ![Walk-Forward and Tensor Illustration](docs/walk_forward_and_tensor_image.png)

*Figure: Walk-Forward Training (left) and Portfolio Batch Illustration (right).* 

## Limitations and Considerations

- Research project from 2020.
- Python 3.5 was used to build this project.
- Uses basic input features and output categories intended for testing purposes only.
- Must be customized with suitable features and prediction targets.
- Import your own data to train the model, as data is not provided.
- Final production models are proprietary.

## Disclaimer
This is a research project and should not be considered financial advice. Trading involves significant financial risk.

## License
http://www.apache.org/licenses/LICENSE-2.0

## Contact
web@deepnexus.com

_Repository initialized in 2025, based on research conducted in 2020._
