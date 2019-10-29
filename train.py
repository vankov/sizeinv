from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import numpy as np

from argparser import args

def test_model(model, data_gen, batch_size):
    acc = 0
    n = 0
    act = 0.0
    while n < batch_size:
        inputs, targets = next(data_gen)
        r = model.predict(inputs)
        acc += float(np.mean(np.equal(np.argmax(targets, axis=1), np.argmax(r, axis=1)) * 1.0))
        act += np.mean(np.max(r, axis=1))
        n += 1
            
    return float(acc) / n, float(act) / n
    
def vgg16_preprocess(input_image):
    x = np.expand_dims(input_image, axis=0)
    x = preprocess_input(x)
    return x[0]

img_gen = ImageDataGenerator(
        preprocessing_function=vgg16_preprocess,            
    )

train_input = img_gen.flow_from_directory(
        'data/train/{}.{}.{}'.format(args.t, args.d, args.noise),
        target_size=(args.image_width, args.image_height),
        batch_size=args.batch_size,
        class_mode='categorical')

test_input = img_gen.flow_from_directory(
            'data/test/{}.{}.{}'.format(args.t, args.d, args.noise),
            target_size=(args.image_width, args.image_height),
            batch_size=args.batch_size,
            class_mode='categorical')
    
base_model = VGG16(
        weights=('imagenet' if args.pretrained else None), 
        include_top=False, 
        input_shape = (args.image_width, args.image_height, 3)
    )
#print(base_model.summary())
x = base_model.output

if args.gap:
    x = GlobalAveragePooling2D()(x)
else:
    x = Flatten()(x)
    

predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

if args.pretrained:
    for layer in base_model.layers:
        layer.trainable = False

model.compile(
        optimizer=RMSprop(lr=args.lr), 
        loss='categorical_crossentropy', 
        metrics=["acc"])

if args.stop_min_delta > 0:
    callbacks = [
            EarlyStopping(
                    monitor="loss", 
                    min_delta = args.stop_min_delta, 
                    patience = args.stop_patience, 
                    verbose=args.v
                )
        ]
else:
    callbacks = []

model.fit_generator(
        train_input, 
        steps_per_epoch = args.train_steps, 
        epochs = args.epochs, 
        verbose = args.v, 
        validation_data = test_input, 
        validation_steps = args.test_batch_size,
        callbacks = []
    )

train_acc, train_act = test_model(model, train_input, batch_size=args.test_batch_size)        
test_acc, test_act = test_model(model, test_input, batch_size=args.test_batch_size)    

print("{}\t{}\t{}\t{}".format(train_acc, train_act, test_acc, test_act))                                

    