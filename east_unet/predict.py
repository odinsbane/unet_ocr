
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
    print("predicted!")

    for i in range(len(tiles)):
        zz = y["boxes"][i]
        sc = y["score"][i]
        print(zz.shape, sc.shape)
        c = 1
        figure = pyplot.figure(1)
        figure.add_subplot(3, 2, c);
        pyplot.imshow(img)
        c += 1
        figure.add_subplot(3, 2, c);
        pyplot.imshow(sc)
        c += 1
        figure.add_subplot(3, 2, c);
        pyplot.imshow(zz[:, :, 0])
        c += 1
        figure.add_subplot(3, 2, c);
        pyplot.imshow(zz[:, :, 1])
        c += 1
        figure.add_subplot(3, 2, c);
        pyplot.imshow(zz[:, :, 2])
        c += 1
        figure.add_subplot(3, 2, c);
        pyplot.imshow(zz[:, :, 3])
        pyplot.show()
