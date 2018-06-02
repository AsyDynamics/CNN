def direct2label():
	os.chdir('/home/asy/Documents/label')

def direct2image():
	os.chdir('/home/asy/Documents/train')

def direct2newimage():
	os.chdir('/home/asy/Documents/train-new')

def readtext(arg1):
	f = open(arg1 + '.txt')
	lines = f.readlines()
	line_num = len(lines)
	object      = []
	coord_left  = []
	coord_top   = []
	coord_right = []
	coord_down  = []
	for row in range(0, line_num):
		line = lines[row]
		word = line.split()
		if word[0] != 'DontCare':
			object.append(word[0])
			coord_left.append(int(float(word[4])))
			coord_top.append(int(float(word[5])))
			coord_right.append(int(float(word[6])))
			coord_down.append(int(float(word[7])))
			word = []
	return object, coord_left, coord_top, coord_right, coord_down

def showimage(arg1):
	my_linewidth = 2
	Im_w = 124
	
	im     = plt.imread(arg1 + '.png')
	image  = plt.gcf()
	my_dpi = image.dpi
	imnew  = Image.open(arg1+'.png')
	L, H   = imnew.size
	plt.figure(figsize=(L/my_dpi, H/my_dpi), dpi=my_dpi)

	print('The given image DPI is: ',my_dpi)
	print('The give image dimension is: ', L, H)
	plt.imshow(im)

	plt.plot([left, left], [top, down], 'r', linewidth=my_linewidth)
	plt.plot([left, right], [top, top], 'r', linewidth=my_linewidth)
	plt.plot([right, right], [top, down], 'r', linewidth=my_linewidth)
	plt.plot([left, right], [down, down], 'r', linewidth=my_linewidth)
	plt.axis([0, L, H, 0])

	direct2newimage();
	plt.savefig('Truck-'+arg1+'.png')
	plt.show()
	


def imagesize(arg1):
	im = plt.imread(arg1 + '.png')
	imnew = Image.open(arg1+'.png')
	L, H = imnew.size
	print('Image size: ',L, H, type(L))
	image = plt.gcf()
	my_dpi  = image.dpi
	my_size = image.get_size_inches()
	print('The given image DPI is: ',my_dpi)
	print('The give image dimension is: ',my_size*my_dpi)

def drawimage(arg1):
	my_dpi = 80
	imnew  = Image.open(arg1+'.png')
	L, H   = imnew.size
	im = np.array(Image.open(arg1+'.png'), dtype=np.uint8)
	fig, ax = plt.subplots(1)
	ax.imshow(im)
	rect = patches.Rectangle((left, down), right-left, top-down, linewidth=2, edgecolor='r', facecolor='none')
	ax.add_patch(rect)
	plt.figure(figsize=(L/my_dpi, H/my_dpi), dpi=my_dpi)
	plt.savefig('Truck-'+arg1+'.png')
	plt.show()

def openSave(arg1):
	im     = plt.imread(arg1 + '.png')
	height, width, nbands = im.shape
	print(height,width)
	plt.imshow(im)
	plt.savefig('openSave-' + arg1 + '.png', dpi=80)

def cropImage(name, object, coord_left, coord_top, coord_right, coord_down):
	im = Image.open(name+'.png')
	object_num = len(object)
	for i in range(0,object_num):
		aera = (coord_left[i], coord_top[i], coord_right[i], coord_down[i])
		testaera = (300,50,900,250)
		cropped = im.crop(aera)
		#cropped.show()
		Imname = object[i]+'-'+name+'-'+str(i)+'.png'
		cropped.save('/home/asy/Documents/train-cropped/'+Imname)

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


# 7481 in total, better to 0-
total = 0;
for x in range(6001,7481):
	name = str(x).zfill(6)
	text_name = name+'.txt'
	image_name = name+'.png'
	direct2label()
	object, coord_left, coord_top, coord_right, coord_down = readtext(name)
	total = total+len(object)
	direct2image()
	cropImage(name, object, coord_left, coord_top, coord_right, coord_down)
print('total object num: ', total)