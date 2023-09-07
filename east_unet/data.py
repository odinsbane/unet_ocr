import os
import imageio
import pathlib
import numpy
import math

from matplotlib import pyplot
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
		return [line.distance(pt) for line in self.lines]
	def crop(self, image):
		return image[self.miny:self.maxy, self.minx:self.maxx, :]
		
		
		
		
def tile_img(img, shape):
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
	#get bounding box
	#iterate over bounding box and find the 4 distances for each point.
	#all four must be positive.
	for i in range(rect.minx, rect.maxx):
		for j in range(rect.miny, rect.maxy):
			if i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]:
			  continue
			pt = (i, j)
			ds = rect.distances(pt)
			if all( d > 0 for d in ds):
				img[j, i, :] = ds[:]
				
DEBUG = False
MODEL_SHAPE = 120, 160, 3

def get_file_names(folder):
	all_files = os.listdir(folder)
	
	labels = [f for f in all_files if f.endswith(".txt")]
	images = [f for f in all_files if f.endswith(".jpg")]
	
	labels.sort()
	images.sort()
	
	if all( images[i][0:-4] in labels[i] for i in range(len(labels)) ):
		print("all %d training data recognized"%len(images))
	else:
		print("image and label names to not correctly correspond.")
	return images, labels
		
def read_image(img_file):
	img = imageio.read(pathlib.Path(img_file)).get_data(0)
	return img
	
def get_text_data(folder, image_names, label_names):
	loaded_images = []
	loaded_outputs = []
	funny = 0
	for i, image in enumerate(image_names):
		img = imageio.read(pathlib.Path(folder, image)).get_data(0)
		boxes = open(pathlib.Path(folder, label_names[i]), 'r').read()
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
		tiles = []
		texts = []
		for rect in rectangles:
			tile = rect.crop(img)
			text = rect.text
			tiles.append(tile)
			texts.append(text)
		
		for tile in tiles:
			loaded_images.append(tile)
		for text in texts:
			loaded_outputs.append(text)
		
	return loaded_images, loaded_outputs


def get_training_data(folder, image_names, label_names):
	loaded_images = []
	loaded_outputs = []
	funny = 0
	for i, image in enumerate(image_names):
		img = imageio.read(pathlib.Path(folder, image)).get_data(0)
		print(img.shape, image)
		#if(img.shape[0] != 720):
		#	continue
		zz = numpy.zeros( (img.shape[0], img.shape[1], 4) )
		boxes = open(pathlib.Path(folder, label_names[i]), 'r').read()
		if boxes[0] == "\ufeff":
			funny += 1
			boxes = boxes[1:]
		boxes = [line for line in boxes.split("\n") if len(line) > 0]
		rectangles = []
		for box in boxes:
			rect = box.strip().split(",")
			#print(box[0], "... testing")
			#print(rect)
			pts = []
			for j in range(4):
				xi = int(rect[2*j])
				yi = int(rect[2*j + 1])
				pts.append( (xi, yi) )
			rectangles.append(BoundingBox(pts))
		for rect in rectangles:
			label(zz, rect)
		if DEBUG:
			c = 1
			figure = pyplot.figure(1)
			figure.add_subplot(2, 2, c);
			pyplot.imshow(img[:, :])
			c += 1
			figure.add_subplot(2, 2, c);
			pyplot.imshow(zz[:, :, 1])
			c += 1
			figure.add_subplot(2, 2, c);
			pyplot.imshow(zz[:, :, 2])
			c += 1
			figure.add_subplot(2, 2, c);
			pyplot.imshow(zz[:, :, 3])
			pyplot.show()
		tiles = tile_img(img, (240, 320, 3))
		tiled_labels = tile_img(zz, (240, 320, 4))
		for tile in tiles:
			loaded_images.append(tile)
		for tile in tiled_labels:
			loaded_outputs.append(tile)
		#loaded_images.append(img)
		#loaded_outputs.append(zz)
	return numpy.array(loaded_images, dtype=float), numpy.array(loaded_outputs, dtype=float)
	
if __name__ == "__main__":
	DEBUG = True
	img_names, lbl_names = get_file_names("data")
	images, labels = get_training_data("data", img_names[:8], lbl_names[:8])
	
