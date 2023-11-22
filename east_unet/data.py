import os
import imageio
import pathlib
import numpy
import math, random
import re

from matplotlib import pyplot

DEBUG=False

class Line:
    def __init__(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        
        self.length = math.sqrt(dx*dx + dy*dy)
        if self.length > 0:
            self.origin = a
            self.dir = ( dx/self.length, dy/self.length)
        else:
            self.origin = a
            self.dir = ( 0, 0)
    
    def distance(self, pt):
        dx = pt[0] - self.origin[0]
        dy = pt[1] - self.origin[1]
        # a x b = |a||b|sin[theta] 
        #|b| is 1 and |a|sin[theta] is the distance to the line.
        cross_product = dx*self.dir[1] - dy * self.dir[0]
        return -cross_product
        
class BoundingBox:
    def __init__(self, pts, text=""):
        self.lines = []
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        
        self.minx = min(xs)
        self.maxx = max(xs)
        
        self.miny = min(ys)
        self.maxy = max(ys)
        
        for i in range(4):
            self.lines.append( Line(pts[i], pts[ (i+1)%4 ]))
        self.text = text
        
    def distances(self, pt):
        """
           Depends on the windings of the input bounding box. This should be fixed!

           pt: point of interest.


        """
        return [line.distance(pt) for line in self.lines]
    def crop(self, image, faulty=False):
        if faulty:
            miny = self.miny + 1 - int(3 * random.random())
            maxy = self.maxy + 1 - int(3 * random.random())
            minx = self.minx + 1 - int(3 * random.random())
            maxx = self.maxx + 1 - int(3 * random.random())
        else:
            miny = self.miny
            maxy = self.maxy
            minx = self.minx
            maxx = self.maxx
            if miny < 0:
                miny = 0
            if minx < 0:
                minx = 0
        #print(image.shape, "::", miny, minx, maxy, maxx)
        #print("to: ", self.text, image[miny:maxy, minx:maxx, :].shape)
        #print("mx: ", numpy.max(image[miny:maxy, minx:maxx, :]))
        return image[miny:maxy, minx:maxx, :]
    def __str__(self):
        return "%s : (%s, %s, %s, %s)"%(self.__class__, self.miny, self.minx, self.maxy, self.maxx)
        
        
        
def tileImg(img, shape):
    ny = int(img.shape[0]/shape[0])
    nx = int(img.shape[1]/shape[1])
    ret = numpy.zeros( (ny*nx, shape[0], shape[1], shape[2]))
    n = 0
    for i in range(ny):
        for j in range(nx):
            ret[n] = img[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], :]
            n += 1
    return ret

def label(img, rect):
    """
        Labels the provided image with the the bounding box coordinates
        and the distance transform normalized from 0 to 1.
    """
    #get bounding box
    #iterate over bounding box and find the 4 distances for each point.
    #all four must be positive.
    dmx = (rect.maxx - rect.minx)/2.0
    dmy = (rect.maxy - rect.miny)/2.0
    max_d = dmx
    if dmy < max_d:
        max_d = dmy

    bpt = [rect.miny, rect.minx, rect.maxy, rect.maxx]
    for i in range(rect.minx, rect.maxx):
        for j in range(rect.miny, rect.maxy):
            if i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]:
              continue
            pt = (i, j)
            ds = rect.distances(pt)
            if all( d > 0 for d in ds):
                img[j, i,0:4] = ds[:]
                img[j, i, 4] = min(ds)/max_d
                
def getFileNames(folder):
    all_files = os.listdir(folder)
    
    labels = [f for f in all_files if f.endswith(".txt")]
    images = [f for f in all_files if f.endswith(".jpg") or f.endswith(".png")]

    labels.sort()
    images.sort()
    
    if all( images[i][0:-4] in labels[i] for i in range(len(labels)) ):
        print("all %d training data recognized"%len(images))
    else:
        print("image and label names to not correctly correspond.")
    return images, labels
        
def readImage(img_file):
    img = imageio.read(pathlib.Path(img_file)).get_data(0)
    return img

def readBoxes(box_file):
    boxes = open(box_file, 'r').read()
    if boxes[0] == "\ufeff":
        funny += 1
        boxes = boxes[1:]
    boxes = [line for line in boxes.split("\n") if len(line) > 0]
    rectangles = []
    for box in boxes:
        rect = box.strip().split(",", 9)
        #print(box[0], "... testing")
        #print(rect)
        pts = []
        for j in range(4):
            xi = int(rect[2*j])
            yi = int(rect[2*j + 1])
            pts.append( (xi, yi) )
        box = BoundingBox(pts, text=rect[-1])
        rectangles.append(box)
    return rectangles

def loadImageLabels(img_file, lbl_file):
    img = readImage(img_file)
    boxes = readBoxes(lbl_file)
    return img, boxes

def labelImage(image, boxes):
    zz = numpy.zeros( (image.shape[0], image.shape[1], 5) )
    for rect in boxes:
        label(zz, rect)
    return zz

def getTextData(image_names, label_names):
    loaded_images = []
    loaded_outputs = []
    funny = 0
    for image, labels in zip(image_names, label_names):
        img, boxes = loadImageLabels( image, labels)
        tiles = []
        texts = []
        for rect in boxes:
            tile = rect.crop(img)
            text = rect.text
            tiles.append(tile)
            texts.append(text)
        
        for tile in tiles:
            loaded_images.append(tile)
        for text in texts:
            loaded_outputs.append(text)
    return loaded_images, loaded_outputs

def getTrainingData(folder, image_names, label_names):
    loaded_images = []
    loaded_outputs = {"boxes":[], "score":[]}
    funny = 0
    for image, labels in zip(image_names, label_names):
        img, rectangles = loadImageLabels(pathlib.Path(folder, image), pathlib.Path(folder, labels))
        zz = labelImage(img, rectangles)

        if DEBUG:
            c = 1
            figure = pyplot.figure(1)
            figure.add_subplot(3, 2, c);
            pyplot.imshow(img[:, :])
            c += 1
            figure.add_subplot(3, 2, c);
            pyplot.imshow(zz[:, :, 4])
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

        #tiles = tile_img(img, (240, 320, 3))
        #tiled_labels = tile_img(zz, (240, 320, 4))
        loaded_images.append(img)
        bb = zz[:, :, 0:4]
        dt = zz[:, :, 4]
        loaded_outputs["boxes"].append(bb)
        loaded_outputs["score"].append(dt)

    loaded_outputs["boxes"] = numpy.array(loaded_outputs["boxes"], dtype="float32")
    loaded_outputs["score"] = numpy.array(loaded_outputs["score"], dtype="float32")

    return numpy.array(loaded_images, dtype=float), loaded_outputs

import sys

if __name__ == "__main__":
    DEBUG = True
    if len(sys.argv) < 2:
        print("Provide the folder with images for loading")
        print("usage: data.py data_folder")
        sys.exit(0)
    data_folder = sys.argv[1]
    
    img_names, lbl_names = getFileNames(data_folder)
    images, labels = getTrainingData(data_folder, img_names[:8], lbl_names[:8])
    
