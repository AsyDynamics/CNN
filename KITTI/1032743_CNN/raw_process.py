def direct2label():
	os.chdir('/home/asy/Documents/label')

def direct2image():
	os.chdir('/home/asy/Documents/train')

def direct2newimage():
	os.chdir('/home/asy/Documents/train-new')

def direct2draw():
	os.chdir('/home/asy/Documents/draw-test/')

def direct2labeltest():
	os.chdir('/home/asy/Documents/label-test/')

def readtext(filename):
	direct2labeltest()
	f = open(filename)
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

# 7481 in total, better to 0-
def crop():
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

def openANDdraw(filename):
	direct2labeltest()
	object, left, top, right, down = readtext(filename+'.txt')
	direct2draw()
	im = Image.open(filename+'.png')
	fig, ax = plt.subplots(1)
	ax.imshow(im)
	for i in range(0,len(object)):
		rect = patches.Rectangle((left[i], down[i]), right[i]-left[i], top[i]-down[i], linewidth=2, edgecolor='r', facecolor='none')
		ax.add_patch(rect)
	plt.savefig('test1.png')
	plt.show()




def main():
	print('enter main')

if __name__ == '__main__':
	import os
	from PIL import Image
	import matplotlib.pyplot as plt
	import numpy as np
	import matplotlib.patches as patches
	import split_process as SP
	labelPath = '/home/asy/Documents/label-test/'
	imagePath = '/home/asy/Documents/draw-test/'
	filename = '000005'
	openANDdraw(filename)