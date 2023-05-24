# CNN-Classification
CNN model to read, process, and classify images (i.e. 10 classes of animals)

![Data Sample](images/natural_images_sample.png)

| Class     | Image Count |
|-----------|-------------|
| car       | 968         |
| cat       | 885         |
| dog       | 702         |
| flower    | 843         |
| fruit     | 1000        |
| motorbike | 788         |
| person    | 986         |

<strong>Total Image Count:</strong>  6899

| Set Split    | Image Count |
|--------------|------|
| Training Set | 5519 |
| Testing Set  | 1380 |


#### Model Summary:

![Model Summary](images/model_summary.png)

#### Model Evaluation:
![Model Loss Plot](images/model_loss.png)
![Model Accuracy Plot](images/model_accuracy.png)

| Metric      | Score  |
|-------------|--------|
| Accuracy    | %91.67 |
| F1 Score    | %91.51 |
| Sensitivity | %91.67 |
| Specificity | %98.84 |

![Confusion Matrix](images/confusion_matrix.png)

| Classification Report |           |        |          |         |
|-----------------------|-----------|--------|----------|---------|
|                       | Percision | Recall | f1-score | support |
| airplane              | 0.97      | 0.96   | 0.97     | 145     |
| car                   | 0.95      | 0.96   | 0.96     | 194     |
| cat                   | 0.75      | 0.84   | 0.79     | 177     |
| dog                   | 0.79      | 0.61   | 0.69     | 140     |
| flower                | 0.87      | 0.92   | 0.89     | 169     |
| fruit                 | 0.99      | 1.00   | 1.00     | 200     |
| motorbike             | 0.99      | 0.98   | 0.99     | 158     |
| person                | 0.99      | 0.99   | 0.99     | 197     |
| accuracy              |           |        | 0.92     | 1380    |
| macro avg             | 0.91      | 0.91   | 0.91     | 1380    |
| weighted avg          | 0.92      | 0.92   | 0.92     | 1380    |


#### Load Model:

    from tensorflow.keras.models import load_model
    loaded_model = load_model('natural_images_model_91.h5')


#### Predict:
    def preprocess_img(image_path, image_size):
        img = Image.open(image_path).convert('RGB').resize(image_size, Image.ANTIALIAS)
        image_arr = np.array(img) / 255.0
        return image_arr
    arr_im = preprocess_img(path_, image_size)
    pred_ = x = np.expand_dims(arr_im, axis=0)
    loaded_model.predict(pred_)


#### Reference:
* Dataset: [Natural Images](https://www.kaggle.com/datasets/prasunroy/natural-images)
* Split: [Split Tool](https://github.com/jfilter/split-folders)
