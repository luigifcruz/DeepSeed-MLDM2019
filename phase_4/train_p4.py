import os
import sys
import glob
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras import backend as K
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    train_img = 'training_set_reduced2/' 
    validation_img = 'test_set_reduced2/'
    nb_train_samples = get_nb_files(train_img)
    nb_classes = len(glob.glob(train_img + "/*"))

    # Data Argumentation
    train_datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Load Data     
    train_generator = train_datagen.flow_from_directory(
		train_img,
		target_size=(299, 299),
		batch_size=64,
		class_mode='categorical'
	)

    validation_generator = validation_datagen.flow_from_directory(
		validation_img,
		target_size=(299, 299),
		batch_size=32,
		class_mode='categorical'
	)

    if(K.image_dim_ordering() == 'th'):
        input_tensor = Input(shape=(3, 299, 299))
    else:
        input_tensor = Input(shape=(299, 299, 3))
    
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="elu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='elu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Currently Training: best_phase4.model")
    print(train_generator.class_indices)

    history_tl = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=40,
        validation_data=validation_generator,
        validation_steps=64,
        shuffle=True,
        callbacks=None
    ) 

    plot_training(history_tl)

    model.save(args.output_model_file)
        
        
def plot_training(history):
    print(history.history.keys())

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig('accuracy.png')
    
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig('loss.png')


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--plot", action="store_true")
    a.add_argument("--output_model_file", default="best_phase4.model")
    args = a.parse_args()
    
    train(args)
