from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, test_ds, class_names):

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))
