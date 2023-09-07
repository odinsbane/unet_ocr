from tensorflow import keras
import data
import math

if __name__ == "__main__":

    model = keras.models.load_model("text_box_unet")
    
    
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-5), 
        loss = keras.losses.MeanSquaredError(),
         )
    images_names, labels_names = data.get_file_names("data")
    
    model.summary()
    chunk = 100
    for loops in range(5):
        for i in range(0, len(images_names), chunk):
            images, labels = data.get_training_data("data", images_names[i:i+chunk], labels_names[i:i + chunk])
            print(images.shape, labels.shape)
            model.fit(x = images, y = labels, batch_size=4, epochs = 20)
            
            y = model(images[:2])
            loser = model.loss(labels[:2], y).numpy()
            
            print( "loss of: ", loser)
            if math.isnan(loser):
                break
            model.save("text_box_unet-5-epochs")
        
    
