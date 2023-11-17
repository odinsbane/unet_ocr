
import sys
from tensorflow import keras
import east_unet.data as data
import numpy

import matplotlib.pyplot as pyplot

if __name__=="__main__":
    model = keras.models.load_model(sys.argv[1])
    img = data.readImage(sys.argv[2])
    print(model.outputs[0].shape)
    tiles = [img]
    #for tile in tiles:
        #pyplot.imshow(tile[:,:,0])
        #pyplot.show()
        
    y = model(numpy.array(tiles))
    for i in range(len(tiles)):
        figure = pyplot.figure(i+1)
        zz = y["boxes"][0]
        c = 1
        figure.add_subplot(2, 3, c);
        pyplot.imshow(zz[:, :, 0])
        c += 1
        figure.add_subplot(2, 3, c);
        pyplot.imshow(zz[:, :, 1])
        c += 1
        figure.add_subplot(2, 3, c);
        pyplot.imshow(zz[:, :, 2])
        c += 1
        figure.add_subplot(2, 3, c);
        pyplot.imshow(zz[:, :, 3])
        c += 1
        figure.add_subplot(2, 3, c);
        pyplot.imshow(tiles[i][:, :])
        c += 1
        figure.add_subplot(2, 3, c);
        pyplot.imshow(y["score"][0])

        pyplot.show()
