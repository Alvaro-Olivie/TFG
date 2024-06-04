from Regression import regression
from SVM import svm
from RandomForest import randomForest
from GradientBoosting import xg_boost
from NeuralNetwork import nn
from Ensemble import ensemble

#regression.main()
#svm.main()
try:
    randomForest.main()
except Exception as e:
    print(f"Error in randomForest: {e}")

try:
    xg_boost.main()
except Exception as e:
    print(f"Error in xg_boost: {e}")

try:
    nn.main()
except Exception as e:
    print(f"Error in nn: {e}")

try:
    ensemble.main()
except Exception as e:
    print(f"Error in ensemble: {e}")



