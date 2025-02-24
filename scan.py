import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from main import x_test, y_test

model=tf.keras.models.load_model('handwritten.keras')
loss , accuracy =model.evaluate(x_test, y_test)

imgno=1

while os.path.isfile(f"digits/image{imgno}.png"):
    img = cv2.imread(f"digits/image{imgno}.png")[:,:,0]
    img=np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"This digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    imgno+=1