import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import glob
import pathlib
import os

batch_size = 16
img_height = 128
img_width = 128


TRAIN  = False

if TRAIN:
    data_dir = "/Users/rohitjain/Desktop/aniket/data/traindir" # 0 and 1
    data_dir = pathlib.Path(data_dir)


    train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.05,
            subset="training",
            seed=242,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=243,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    print(np.min(first_image), np.max(first_image))


    num_classes = len(class_names)

    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

    print(model.summary())

    epochs=25
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


    model.save('/Users/rohitjain/Desktop/aniket/classification/model_new.h5')

else:

    def classify(img_path): 
        class_names = ['0', '1']
        model_new = tf.keras.models.load_model("/Users/rohitjain/Desktop/aniket/classification/model_new.h5")
                
        # print(os.path.basename(img_path))
        img = tf.keras.utils.load_img(
            img_path, target_size=(img_height, img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model_new.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(score)
        return np.argmax(score)


# img_path = ''

# if 'True' in img_path:
#     shutil.copy(img_path, '/Users/rohitjain/Desktop/aniket/data/traindir/1')
# else:
#     shutil.copy(img_path, '/Users/rohitjain/Desktop/aniket/data/traindir/0')