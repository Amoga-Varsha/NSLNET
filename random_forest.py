import os
from skimage import io, color, feature, util
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Set up empty lists for features and labels
X = []
y = []

# Define distance and angle offsets for GLCM
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Loop through each image in the directory
directory = r'C:\Users\Amoga\Documents\workspace gdb\statsistics and optimization\greyscale_train\trial'
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        # Load image and convert to grayscale
        image = io.imread(os.path.join(directory, filename))
        gs = util.img_as_ubyte(color.rgb2gray(image))
        
        # Calculate GLCM
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
        y.append(filename.split('_')[0]) # Assumes filename is in the format 'class_label_filename.jpg'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred=rf.predict(X_test)
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

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrBr)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()
