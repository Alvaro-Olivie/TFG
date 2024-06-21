### README

# Bond Price Prediction Using Machine Learning
### Author: Álvaro Olivié Molina

## Description
This project aims to develop and validate models that can predict bond price changes over various horizons (1, 3, 6, and 12 months) using machine learning methods. The main file (`main.py`) orchestrates the execution of all models as well as a simple portfolio simulation.

## Directory Overview
- **Data:** Contains raw and processed datasets in Excel format.
- **DataPreprocessing:** Script for data cleaning and preprocessing.
- **Ensemble:** Implementation of ensemble models.
- **GradientBoosting:** Contains models and script related to Gradient Boosting.
- **NeuralNetwork:** Implements neural network models.
- **Portfolio:** Script for portfolio construction based on model predictions.
- **RandomForest:** Implements Random Forest models.
- **Regression:** Implements various regression models.
- **SVM:** Implements Support Vector Machine models.

### Model Directories
  - `.py`: Main script for neural network training.
  -  Saved models for different horizons.
  - `loss_*.png`, `r2_*.png`, `scatter_*.png`: Performance visualizations.
  - `y_pred_*.csv`: Predicted values for different horizons.
  - `results.csv`: Consolidated results.

## Usage
1. **Install Requirements:**
   ```
   pip install -r requirements.txt
   ```

2. **Run Main Script:**
   ```
   python main.py
   ```

## Data
- **bonds_not_clean.csv:** Initial raw dataset.
- **bonds.csv:** Cleaned dataset used for training and testing.
- **train_bonds.csv:** Training dataset.
- **test_bonds.csv:** Testing dataset.

These are have not been uploaded as they are not publically available but are referenced in the code. The same has been done for the original Excel files

## License
This project is licensed under the license referenced in the LICENSE file.
