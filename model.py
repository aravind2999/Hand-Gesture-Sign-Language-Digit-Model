#import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

#Data Augumentation
from keras.preprocessing.image import ImageDataGenerator
image_datagen = ImageDataGenerator(rescale = 1./255)
image_generator = image_datagen.flow_from_directory('Sign_Language_digits', target_size = (100,100), class_mode = 'categorical', batch_size = 10)



#Colutional Neural Network Model
import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Conv2D(64,(3,3),input_shape = (100,100,3), activation = 'relu'))
ann.add(tf.keras.layers.MaxPooling2D(2,2))
ann.add(tf.keras.layers.Conv2D(64 ,(3,3), activation = 'relu'))
ann.add(tf.keras.layers.MaxPooling2D(2,2))
ann.add(tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'))
ann.add(tf.keras.layers.MaxPooling2D(2,2))
ann.add(tf.keras.layers.Flatten())
ann.add(tf.keras.layers.Dense(512, activation = 'relu'))
ann.add(tf.keras.layers.Dense(10, activation = 'softmax'))
ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
ann.fit_generator(image_generator, steps_per_epoch = 8, epochs = 35, verbose = 1)



#VISUALIZATION
#import Libraries
import cv2
import copy
import time

#Ouput Values
numbers = ['0','1','2','3','4','5','6','7','8','9']

# start point/total width
x_start = 0.4
y_end = 0.9

#Threshold Value
threshold = 80

#Boolean Value
isBgCaptured = 0

# GaussianBlur parameter
blur_value = 50

bgSubThreshold = 50
learningRate = 0
#Open Inbuilt Camera of Pc
camera = cv2.VideoCapture(0)

#Prediction of the image captured
def predict(img):
    img = np.array(img, dtype = 'float32')
    img /=255
    pred_array = ann.predict(img)
    print(f'pred_array: {pred_array}')
    result = numbers[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

#Remove Background of the captured Image
def remove_background(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame, (int(x_start * frame.shape[1]), 0),
                  (frame.shape[1], int(y_end * frame.shape[0])), (255, 0, 0), 2)
    k = cv2.waitKey(10)
    cv2.imshow('Window', frame)

    #If 'b' is pressed
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                   int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        cv2.imshow('mask',img)

        #Convert Image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('ori', thresh)
        thresh1 = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(thresh1,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maximumArea = -1
        if length>0:
            for i in range(length):
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maximumArea:
                    maximumArea = area
                    c = i
            res = contours[c]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        cv2.imshow('output', drawing)
    #When ESC is pressed
    if k == 27:
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')

    elif k == ord('r'):
        time.sleep(1)
        bgModel = None
        triggerSwitch  = False
        isBgCaptured = 0
        print("Reset Background")

    #When SpaceBar is pressed
    elif k == 32:
        cv2.imshow('original', frame)
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (100, 100))
        target = target.reshape(1, 100, 100, 3)
        prediction, score = predict(target)
        camera.release
        cv2.destroyAllWindows()

#Visualize the intermediate Layers in Convolutoinal Neural Networks
successive_outputs =  [layer.output for layer in ann.layers[:]]
visualization_model = tf.keras.models.Model(inputs = ann.input, outputs = successive_outputs)
successive_feature_maps = visualization_model.predict(image_generator)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in ann.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 3:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
        size = feature_map.shape[2]
    # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
      # Postprocess the feature to make it visually palatable
            feature_map_x = feature_map[0, :, :, i]
            feature_map_x -= feature_map_x.mean()
            feature_map_x /= feature_map_x.std()
            feature_map_x *= 64
            feature_map_x += 128
            feature_map_x = np.clip(feature_map_x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size : (i + 1) * size] = feature_map_x
    # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

#Prediction of Manually Input Image
from keras.preprocessing import image
test_image = image.load_img('IMG.JPG',target_size = (100,100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = ann.predict(test_image)
result = np.argmax(result)
