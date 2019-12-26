import pandas as pd
import tensorflow as tf
import numpy as np

tf.test.is_gpu_available()

df_train = pd.read_csv("train.csv")
df_train.head()

df_train.shape

df_train['Id'].value_counts()

len(df_train['Id'].value_counts())


def encode_images(train, shape, path):
    
    x_train = np.zeros((shape, 100, 100, 3))
    count = 0
    
    for fig in train['Image']:
        
        #load images into images of size 100x100x3
        img = tf.keras.preprocessing.image.load_img(path+"/"+fig, target_size=(100, 100, 3))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.imagenet_utils.preprocess_input(x)

        x_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return x_train
    
    def target_encode(label_column):
    
    targets = list(set(label_column))
    id_mapping = {}
    
    
    for i in range(len(targets)):
        id_mapping[targets[i]] = i
    
    y = np.zeros((len(label_column), len(targets)))
    
    for i, label in enumerate(label_column):
        y[i][id_mapping[label]] = 1
    
    return y, id_mapping
    
    X_train = encode_images(df_train, df_train.shape[0], './train')
    
    X_train.nbytes
    
    X_train /= 255
    
    y_train, id_mapping = target_encode(df_train['Id'])
    
    X_train.shape
    y_train.shape
    
    

#number of filters


#Window Size
#KERNEL_SIZE = (5,5)
INPUT_SHAPE = (100,100,3)


tf.keras.backend.clear_session()





model = tf.keras.models.Sequential([
    
    ######CONV####
    #Conv1
    tf.keras.layers.Conv2D(32, kernel_size=(7,7), strides = (1, 1), input_shape=INPUT_SHAPE, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.1),
    
    #Conv3
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.AveragePooling2D((3, 3)),
    
    

  
        
    #####OUTPUT####
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(y_train.shape[1], activation = "softmax"),

])

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss='categorical_crossentropy',  metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train,
                    batch_size=1,
                    epochs=1,
                    verbose=2,
         )
