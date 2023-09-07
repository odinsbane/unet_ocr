from tensorflow import keras
import data
import math

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: train.py model_file data_folder")
        sys.exit(0)
    
    model_file = sys.argv[1]
    data_folder = sys.argv[2]
    
    model = keras.models.load_model(model_file)
    
    
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-5), 
        loss = keras.losses.MeanSquaredError(),
         )
    images_names, labels_names = data.get_file_names(data_folder)
    
    model.summary()
    chunk = 100
    epochs = 5
    for loops in range(epochs):
        for i in range(0, len(images_names), chunk):
            images, labels = data.get_training_data(data_folder, images_names[i:i+chunk], labels_names[i:i + chunk])
            print(images.shape, labels.shape)
            model.fit(x = images, y = labels, batch_size=4, epochs = 20)
            
            y = model(images[:2])
            loser = model.loss(labels[:2], y).numpy()
            
            print( "loss of: ", loser)
            if math.isnan(loser):
                break
            model.save("text_box_unet-5-epochs")
        
    
