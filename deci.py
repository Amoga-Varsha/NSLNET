import os
import numpy as np
from skimage import io, color, feature, util
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import time
import matplotlib.pyplot as plt

st = time.time()

X = []  # Features
y = []  # Labels

# Distance and angle offsets for GLCM
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Iterate through every image found in the specified directory
directory = r'C:\Users\Amoga\Documents\workspace gdb\statsistics and optimization\greyscale_train\trial'
for file in os.listdir(directory):
    if file.endswith('.jpg'):
        # Convert image to grayscale
        image = io.imread(os.path.join(directory, file))
        gs = util.img_as_ubyte(color.rgb2gray(image))

        # Calculate Gray Level Co-occurrence Matrix (GLCM)
        glcm = feature.graycomatrix(gs, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

       # Calculate GLCM properties
        features = np.hstack([
        feature.graycoprops(glcm, 'contrast').ravel(),
        feature.graycoprops(glcm, 'dissimilarity').ravel(),
        feature.graycoprops(glcm, 'homogeneity').ravel(),
        feature.graycoprops(glcm, 'energy').ravel(),
        feature.graycoprops(glcm, 'correlation').ravel(),
    ])
        
        # Append features and label to lists
        X.append(features)
        y.append(file.split('_')[0])  # In line with dataset's image naming convention

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=42)

Src = StandardScaler()
dTree = DecisionTreeClassifier()

pipe = Pipeline([
    ('scaler', Src),
    ('DecisionTree', dTree)
])

param_grid = [{
    'DecisionTree__max_depth': [2, 3, 4, 5, 6],
    'DecisionTree__min_samples_split': [2, 3, 5, 10, 15]
}]

grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)

DTree = grid.fit(X, y)


print('Best hyperparameters:', grid.best_params_)

y_pred=grid.predict(X_test)
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)


cm = confusion_matrix(y_test, y_pred)
class_labels = np.unique(y_test)
class_accuracy = {}
for i in range(len(class_labels)):
    class_accuracy[class_labels[i]] = cm[i,i]/np.sum(cm[i,:])
    print(f'Accuracy for class {class_labels[i]}:', class_accuracy[class_labels[i]])

