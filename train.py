import numpy as np
import pandas as pd
from params import TRAIN_PATH, BATCH_SIZE, EPOCHS
from utils import get_mask_with_image
from model.u_net import get_unet_model
from model.callbacks import callbacks_list, weight_path

def make_image_gen(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    masks = []
    images = []
    while True:
        np.random.shuffle(all_batches)
        for image, masks_df in all_batches:
            image, mask = get_mask_with_image(image, masks_df)
            images += image
            masks += mask
            if len(images)>=batch_size:
                yield np.stack(images, 0)/255.0, np.stack(masks, 0)
                masks, images=[], []


model = get_unet_model()
model.summary()

train_df = pd.read_csv(TRAIN_PATH+'train_df.csv')
valid_df = pd.read_csv(TRAIN_PATH+'valid_df.csv')

train_gen = make_image_gen(train_df)
valid_x, valid_y = next(make_image_gen(valid_df))

history = model.fit_generator(train_gen,
                             steps_per_epoch=len(train_df)//BATCH_SIZE,
                             epochs=EPOCHS,
                             validation_data=(valid_x, valid_y),
                             callbacks=callbacks_list)

accuracy = history.history['acc']
loss = history.history['loss']
val_accuracy = history.history['val_acc']
val_loss = history.history['val_loss']

# print(f'Training Accuracy: {np.max(accuracy)}')
# print(f'Training Loss: {np.min(loss)}')
# print(f'Validation Accuracy: {np.max(val_accuracy)}')
# print(f'Validation Loss: {np.min(val_loss)}')

model.load_weights(weight_path)
model.save('model.h5')