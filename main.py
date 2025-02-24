import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values (0 to 255 → 0 to 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 1D vector
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    keras.layers.Dense(10, activation='softmax')  # Output layer (10 classes: digits 0-9)
])

# Compile the model (Optimizer, Loss function, Metrics)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)
model.save('handwritten.keras')

'''# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Make a prediction on a single image
index = 0  # Change index to test different images
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Actual Label: {y_test[index]}")
plt.show()

prediction = model.predict(x_test[index].reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
print(f"Predicted Label: {predicted_label}")
'''