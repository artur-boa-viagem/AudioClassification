import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model
import numpy as np
from utils import getEveryFile, getSpectrogramFromImage
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_model(model_name='HOG_spectrogram_model.keras'):
    # Load the specific model
    model = load_model(model_name)

    # Running the model on the specific test set
    testFiles = getEveryFile(fileType = 'Test/', typeDir = 'HOG/' if 'HOG' in model_name else 'LBP/')

    # Load the test data and labels
    test_data, test_label = zip(*[getSpectrogramFromImage(path) for path in testFiles])
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data, test_label, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

    # Predict the test data
    predictions = model.predict(test_data)

    # Get the confusion matrix
    cm = confusion_matrix(test_label, np.argmax(predictions, axis=1))

    # Display the confusion matrix
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Bird', 'Cat', 'Dog', 'Sheep', 'Cow'])

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    cmd.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
