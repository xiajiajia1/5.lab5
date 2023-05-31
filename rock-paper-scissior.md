# 实验5.2：TensorFlow训练石头剪刀布


## 1.下载所有石头剪刀布的数据集
![](./photo/6.png)
## 2.解压数据集


```python
import os
import zipfile

local_zip = './mldownload/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./mldownload/')
zip_ref.close()

local_zip = './mldownload/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./mldownload/')
zip_ref.close()

```

## 3.检测数据集的解压结果，打印相关信息


```python
rock_dir = os.path.join('./mldownload/rps/rock')
paper_dir = os.path.join('./mldownload/rps/paper')
scissors_dir = os.path.join('./mldownload/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

```

    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock01-066.png', 'rock04-069.png', 'rock05ck01-006.png', 'rock04-085.png', 'rock05ck01-089.png', 'rock06ck02-018.png', 'rock05ck01-075.png', 'rock05ck01-110.png', 'rock04-041.png', 'rock06ck02-046.png']
    ['paper04-093.png', 'paper03-075.png', 'paper04-109.png', 'paper06-036.png', 'paper01-110.png', 'paper07-005.png', 'paper04-082.png', 'paper01-106.png', 'paper05-067.png', 'paper01-068.png']
    ['testscissors02-046.png', 'scissors04-050.png', 'testscissors01-033.png', 'testscissors03-082.png', 'scissors02-099.png', 'testscissors03-038.png', 'scissors02-103.png', 'scissors01-021.png', 'testscissors02-093.png', 'scissors02-015.png']
    

## 4.各打印两张石头剪刀布训练集图片


```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

```


    
![png](output_6_0.png)
    



    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



    
![png](output_6_3.png)
    



    
![png](output_6_4.png)
    



    
![png](output_6_5.png)
    


## 5.调用TensorFlow的keras进行数据模型的训练和评估
Keras是开源人工神经网络库，TensorFlow集成了keras的调用接口，可以方便的使用。


```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./mldownload/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "./mldownload/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    

    2023-05-31 01:12:24.151149: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2023-05-31 01:12:24.151187: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2023-05-31 01:12:24.151234: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-d7c12f): /proc/driver/nvidia/version does not exist
    2023-05-31 01:12:24.151910: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
     2D)                                                             
                                                                     
     conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 6272)              0         
                                                                     
     dropout (Dropout)           (None, 6272)              0         
                                                                     
     dense (Dense)               (None, 512)               3211776   
                                                                     
     dense_1 (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________
    

    2023-05-31 01:12:25.521662: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.
    

    Epoch 1/25
    

    2023-05-31 01:12:27.969627: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.
    2023-05-31 01:12:28.004826: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.
    2023-05-31 01:12:28.804955: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 176633856 exceeds 10% of free system memory.
    2023-05-31 01:12:28.905281: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 167215104 exceeds 10% of free system memory.
    

    20/20 [==============================] - 79s 4s/step - loss: 1.6453 - accuracy: 0.3242 - val_loss: 1.0951 - val_accuracy: 0.3333
    Epoch 2/25
    20/20 [==============================] - 75s 4s/step - loss: 1.1051 - accuracy: 0.4020 - val_loss: 1.0883 - val_accuracy: 0.3683
    Epoch 3/25
    20/20 [==============================] - 76s 4s/step - loss: 1.0160 - accuracy: 0.4925 - val_loss: 0.7983 - val_accuracy: 0.8387
    Epoch 4/25
    20/20 [==============================] - 75s 4s/step - loss: 0.9120 - accuracy: 0.5444 - val_loss: 0.6206 - val_accuracy: 0.6667
    Epoch 5/25
    20/20 [==============================] - 75s 4s/step - loss: 0.8003 - accuracy: 0.6171 - val_loss: 0.3324 - val_accuracy: 1.0000
    Epoch 6/25
    20/20 [==============================] - 74s 4s/step - loss: 0.7596 - accuracy: 0.6567 - val_loss: 0.4173 - val_accuracy: 0.7366
    Epoch 7/25
    20/20 [==============================] - 75s 4s/step - loss: 0.6478 - accuracy: 0.7242 - val_loss: 0.2654 - val_accuracy: 0.9919
    Epoch 8/25
    20/20 [==============================] - 73s 4s/step - loss: 0.4636 - accuracy: 0.8123 - val_loss: 0.1094 - val_accuracy: 1.0000
    Epoch 9/25
    20/20 [==============================] - 76s 4s/step - loss: 0.4130 - accuracy: 0.8357 - val_loss: 0.0913 - val_accuracy: 0.9919
    Epoch 10/25
    20/20 [==============================] - 74s 4s/step - loss: 0.3168 - accuracy: 0.8726 - val_loss: 0.0674 - val_accuracy: 0.9892
    Epoch 11/25
    20/20 [==============================] - 71s 4s/step - loss: 0.3930 - accuracy: 0.8563 - val_loss: 0.0616 - val_accuracy: 0.9839
    Epoch 12/25
    20/20 [==============================] - 73s 4s/step - loss: 0.2877 - accuracy: 0.8960 - val_loss: 0.4448 - val_accuracy: 0.7258
    Epoch 13/25
    20/20 [==============================] - 74s 4s/step - loss: 0.1907 - accuracy: 0.9310 - val_loss: 0.0932 - val_accuracy: 0.9651
    Epoch 14/25
    20/20 [==============================] - 73s 4s/step - loss: 0.2121 - accuracy: 0.9278 - val_loss: 0.5036 - val_accuracy: 0.6694
    Epoch 15/25
    20/20 [==============================] - 71s 4s/step - loss: 0.2078 - accuracy: 0.9167 - val_loss: 0.1735 - val_accuracy: 0.9032
    Epoch 16/25
    20/20 [==============================] - 71s 3s/step - loss: 0.1354 - accuracy: 0.9492 - val_loss: 0.1382 - val_accuracy: 0.9570
    Epoch 17/25
    20/20 [==============================] - 74s 4s/step - loss: 0.1817 - accuracy: 0.9294 - val_loss: 0.0351 - val_accuracy: 0.9839
    Epoch 18/25
    20/20 [==============================] - 71s 4s/step - loss: 0.1395 - accuracy: 0.9548 - val_loss: 0.0344 - val_accuracy: 1.0000
    Epoch 19/25
    20/20 [==============================] - 73s 4s/step - loss: 0.1272 - accuracy: 0.9571 - val_loss: 0.0699 - val_accuracy: 0.9597
    Epoch 20/25
    20/20 [==============================] - 73s 4s/step - loss: 0.1181 - accuracy: 0.9603 - val_loss: 0.0233 - val_accuracy: 0.9919
    Epoch 21/25
    20/20 [==============================] - 72s 4s/step - loss: 0.1390 - accuracy: 0.9492 - val_loss: 0.0327 - val_accuracy: 0.9839
    Epoch 22/25
    20/20 [==============================] - 71s 4s/step - loss: 0.0795 - accuracy: 0.9722 - val_loss: 0.0439 - val_accuracy: 0.9785
    Epoch 23/25
    20/20 [==============================] - 70s 3s/step - loss: 0.1289 - accuracy: 0.9567 - val_loss: 0.8433 - val_accuracy: 0.4973
    Epoch 24/25
    20/20 [==============================] - 71s 4s/step - loss: 0.1042 - accuracy: 0.9575 - val_loss: 0.0251 - val_accuracy: 0.9946
    Epoch 25/25
    20/20 [==============================] - 70s 3s/step - loss: 0.1248 - accuracy: 0.9567 - val_loss: 0.0255 - val_accuracy: 0.9866
    

## 6.绘制训练和验证结果的相关信息


```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

```


    
![png](output_10_0.png)
    



    <Figure size 640x480 with 0 Axes>

