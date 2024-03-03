# Breast Cancer Prediction Classification
## Dataset
I used the [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code) from Kaggle, for this project. In the future, I hope to gain access to a dataset that reflects African low and middle-class groups.

The characteristics are computed from a digitized representation of a fine needle aspirate (FNA) captured from a breast mass, outlining details of the cell nuclei depicted in the image.

### Attributes include:
1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:
a) radius (mean of distances from the center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The dataset is designed for a binary classification problem where the target variable, **diagnosis**, represents the presence or absence of breast cancer. The dataset undergoes preprocessing steps, including handling missing values (dropna(axis=1)) and encoding the target variable (LabelEncoder). The dataset is partitioned into training, testing, and validation sets using train_test_split from sklearn.model_selection. Model 1 is initially created without explicit optimization techniques. Model 2 introduces optimization techniques: L1, L2 Regularization, and Adam Optimizer. Evaluation metrics for both models include accuracy, loss, confusion matrix, specificity, F1 score, and classification report.

## Model 1 (without optimization)
The neural network architecture comprises an input layer with 30 neurons (ReLU activation) and an output layer with 1 neuron (Sigmoid activation) for binary classification. During compilation, the model utilizes binary crossentropy as the loss function and measures accuracy. The training process consists of 100 epochs with a batch size of 64, and validation occurs on the test set.

## Model 1 (with optimization)
The neural network features an input layer with 30 neurons (ReLU activation) and two hidden layers. The first hidden layer has 16 neurons with ReLU activation and L1 regularization (0.01), while the second employs L2 regularization (0.01). The output layer is a single neuron with 'sigmoid' activation for binary classification. It was compiled with the Adam optimizer, binary crossentropy loss, and accuracy metric. Training uses 100 epochs with a batch size of 64, and validation occurs on the test set. Optimization techniques include L1 and L2 regularization to counter overfitting and the adaptive learning rates of the Adam optimizer.

## Libraries used
- **numpy and pandas:** for data manipulation and analysis.
- **seaborn and matplotlib.pyplot:** for data visualization.
- **tensorflow:** for building and training neural network models.
- **LabelEncoder from sklearn.preprocessing:** for encoding categorical labels.
- **MinMaxScaler from sklearn.preprocessing:** for feature scaling.
- **train_test_split from sklearn.model_selection:** for splitting data into training and test sets.
- **Sequential, Dense, Activation, l1, l2, and Adam from keras.models and keras.layers:** for defining and optimizing neural network models.
- **confusion_matrix and f1_score from sklearn.metrics:** for evaluating model performance.

## The benefits of Optimization techniques
Considering the mass of the dataset, it may contain noise or irrelevant features. To prevent overfitting or underfitting, L1 and L2 are used. These techniques encourage sparsity in the weight matrix, performing feature selection and penalizing large weights. In Model 2, L1 regularization strength is set to 0.01 (kernel_regularizer=l1(0.01)), and L2 regularization strength is set to 0.01 (kernel_regularizer=l2(0.01)).This was set through trial and error to prevent over-penalizing the model. 

In addition, the Adam optimizer used in Model 2 is used to improve convergence and generalization as it adapts its learning rates. It is used with default parameters. In general, the parameter tuning for both models was based on references from other existing models as well as observing the performance of the validation sets.

## Model Prediction
Using the test data for prediction purposes, I used a confusion matrix, f1 score, loss, and a classification report as methods for evaluation. It is found that the model using optimization techniques consistently performs much better than the first model without any optimization techniques.
