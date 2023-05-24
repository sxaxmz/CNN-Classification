import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from contextlib import redirect_stdout


dataset_path = 'Dataset'#'raw-img'
class_folders = os.listdir(dataset_path) # list of folder names
image_size = (112, 112)
batch_size = 32
images = []
labels = []
img_count = 0

def preprocess_img(image_path, image_size):
    img = Image.open(image_path).convert('RGB').resize(image_size, Image.ANTIALIAS)
    image_arr = np.array(img) / 255.0
    return image_arr

for folder in class_folders:
    folder_path = os.path.join(dataset_path, folder)
    image_files = os.listdir(folder_path)
    img_count += len(image_files)
    print(folder, ' : ', len(image_files))
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        
        # img = Image.open(img_path).convert('RGB').resize(image_size, Image.ANTIALIAS)
        # img_array = np.array(img) / 255.0
        
        images.append(preprocess_img(img_path, image_size))
        #images.append(img_array)
        labels.append(folder)
print('Total Image Count: ', img_count)


unique_labels = list(set(labels))
print(unique_labels)
plt.figure(figsize = (20 , 20))
plt.ion()
for i, label in enumerate(unique_labels):
    plt.subplot(5 , 5, i+1)
    plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
    plt.imshow(images[labels.index(label)])
    plt.title(f'Class: {label}')
    plt.axis("off")
    plt.pause(0.1)
plt.ioff() 
plt.show()

images = np.array(images)
labels = np.array(labels)
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
print('Splitting {} into  {} train and {} test'.format(img_count, round(img_count*0.8), round(img_count*0.2)))
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
print('aa')
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.summary()

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(20,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.figure(figsize=(20,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=-1)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: % {:.2f}".format(accuracy * 100))

report = classification_report(y_test, y_pred, target_names=encoder.classes_)
print("Classification Report:\n", report)

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    return np.array(specificity)

# Replace 'y_pred_rf' with 'y_pred'
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
specificity = specificity_score(y_test, y_pred)

# Calculate the overall specificity
class_frequencies = np.bincount(y_test)
overall_specificity = np.average(specificity, weights=class_frequencies)

print("Accuracy: % {:.2f}".format(accuracy * 100))
print("Precision: % {:.2f}".format(precision * 100))
print("Recall: % {:.2f}".format(recall * 100))
print("F1 Score: % {:.2f}".format(f1 * 100))
print("Overall Specificity: % {:.2f}".format(overall_specificity * 100))
