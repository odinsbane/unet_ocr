from tensorflow import keras
import east_unet.data as data
import math
import pathlib
import random

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: train.py model_file data_folder")
        sys.exit(0)
    
    model_file = pathlib.Path(sys.argv[1])
    data_folder = pathlib.Path(sys.argv[2])
    
    model = keras.models.load_model(model_file)
    
    
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-5), 
        loss = keras.losses.MeanSquaredError(),
        loss_weights={"boxes":0.01, "score":0.99}
         )
    images_names, labels_names = data.getFileNames(data_folder)
    a = [(img, lbl) for img, lbl in zip(images_names, labels_names)]
    n = len(a)
    validation = 100
    t = n - validation
    train = a[:t]
    random.shuffle(train)
    val = a[t:]


    vi_names = [ il[0] for il in val]
    vl_names = [ il[1] for il in val]
    val_images, val_labels = data.getTrainingData( data_folder, vi_names, vl_names )
    images_names = [il[0] for il in train]
    labels_names = [il[1] for il in train]

    model.summary()
    chunk = 500
    epochs = 5

    with open("%s-loss.txt"%model_file.name, 'w') as log:
        log.write("#epoch\tchunk\tbox-loss\tscore-loss\n")
        for loops in range(epochs):
            for i in range(0, len(images_names), chunk):
                images, labels = data.getTrainingData(data_folder, images_names[i:i+chunk], labels_names[i:i + chunk])
                model.fit(x = images, y = labels, batch_size=2, epochs = 2)
                y = model.predict(val_images, batch_size=2)
                bl = model.loss(val_labels["boxes"], y["boxes"]).numpy()
                sl = model.loss(val_labels["score"], y["score"][:,:,:,0]).numpy()
                log.write("%s\t%s\t%s\t%s\n"%(loops, i, bl, sl))
                log.flush()
                if math.isnan(bl) or math.isnan(sl):
                    break
                model.save( pathlib.Path(model_file.parent, "%s-e-%s"%(model_file.name, loops)) )
        
    
