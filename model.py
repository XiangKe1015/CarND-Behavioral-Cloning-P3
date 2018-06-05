import csv 
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg


samples=[]
with open('data/driving_log.csv') as f:
    reader=csv.reader(f)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
samples_per_epoch=len(train_samples)*4
nb_val_samples=len(validation_samples)*4

def generator(samples, batch_size=32,correction=0.229):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images=[]
            measurements=[]
            for batch_samples in batch_samples:
                for i in range(3):
                    source_path=batch_samples[i]
                    if len(source_path.split('/')[-1])<50:
                        filename=source_path.split('/')[-1]
                    else:
                        filename=source_path.split('\\')[-1]
                    current_path='data/IMG/'+filename
                    image=mpimg.imread(current_path)
                    images.append(image)
                    measurement=float(batch_samples[3])
                    if i==0:
                        measurements.append(measurement)
                        images.append(cv2.flip(image,1)) 
                        measurements.append(measurement*-1.0)
                    elif i==1:
                        measurements.append(measurement+correction)
                    elif i==2:
                        measurements.append(measurement-correction)

           
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)
#  model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D,Activation
from keras.layers import Convolution2D,Dropout,MaxPooling2D

model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

history_object = model.fit_generator(train_generator,
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=5,
                              validation_data=validation_generator,
                              nb_val_samples=nb_val_samples,
                              verbose=1)
model.save('model.h5')

from keras.models import Model
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()