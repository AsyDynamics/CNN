def direct2train(path):
	os.chdir(path)

def get_file_names(path):
	file_names = [f for f in listdir(path) if isfile(join(path, f))]
	file_num = len(file_names)
	return file_num, file_names


def split_class_name(file_names):
	Car = []
	Van = []
	Truck = []
	Pedestrian = []
	Person_sitting = []
	Cyclist = []
	Tram = []
	Misc = []
	for i in range(0, len(file_names)):
		label = file_names[i].split('-')[0]
		if label == 'Car':
			Car.append(file_names[i])
		elif label == 'Van':
			Van.append(file_names[i])
		elif label == 'Truck':
			Truck.append(file_names[i])
		elif label == 'Pedestrian':
			Pedestrian.append(file_names[i])
		elif label == 'Person_sitting':
			Person_sitting.append(file_names[i])
		elif label == 'Cyclist':
			Cyclist.append(file_names[i])
		elif label == 'Tram':
			Tram.append(file_names[i])
		elif label == 'Misc':
			Misc.append(file_names[i])
	return Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc




def get_class_num(file_names):
	Car = 0
	Van = 0
	Truck = 0
	Pedestrian = 0
	Person_sitting = 0
	Cyclist = 0
	Tram = 0
	Misc = 0
	for i in range(0, len(file_names)):
		label = file_names[i].split('-')[0]
		if label == 'Car':
			Car += 1
		elif label == 'Van':
			Van += 1
		elif label == 'Truck':
			Truck += 1
		elif label == 'Pedestrian':
			Pedestrian += 1
		elif label == 'Person_sitting':
			Person_sitting += 1
		elif label == 'Cyclist':
			Cyclist += 1
		elif label == 'Tram':
			Tram += 1
		elif label == 'Misc':
			Misc +=1
	return Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc

def print_class_num(Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc):
	print('File number in total',file_num)
	print('Car: %d %4.2f%%' %(Car,100*Car/file_num))
	print('Van: %d %4.2f%%' %(Van,100*Van/file_num))
	print('Truck: %d %4.2f%%' %(Truck,100*Truck/file_num))
	print('Pedestrian: %d %4.2f%%' %(Pedestrian,100*Pedestrian/file_num))
	print('Person_sitting: %d %4.2f%%' %(Person_sitting,100*Person_sitting/file_num))
	print('Cyclist: %d %4.2f%%' %(Cyclist,100*Cyclist/file_num))
	print('Tram: %d %4.2f%%' %(Tram,100*Tram/file_num))
	print('Misc: %d %4.2f%%' %(Misc,100*Misc/file_num))

def test_image_size(path):
	import glob
	image_list = []
	sum_size = 0
	image_num = 0
	for filename in glob.glob(path+'*.png'):
		im=Image.open(filename)
		image_num += 1
		sum_size += im.size[0]*im.size[1]
		#image_list.append(im)
	print('sum size:', sum_size)
	print('mean size:', sum_size/image_num)

def get_image_size(file_names,path):
	direct2train(path)
	#Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc
	IM_size = []
	IM_size_mean_class = []
	classnum = 8
	for i in range (0,classnum):
		IM_size.append([0,0])
		IM_size_mean_class.append([0,0])
	Class_size = np.zeros(8)
	np.set_printoptions(precision=3)

	for image in file_names:
		im = Image.open(image)
		W,H = im.size
		label = image.split('-')[0]
		if label == 'Car':
			IM_size[0][0] += W
			IM_size[0][1] += H
			Class_size[0] += 1
		elif label == 'Van':
			IM_size[1][0] += W
			IM_size[1][1] += H
			Class_size[1] += 1
		elif label == 'Truck':
			IM_size[2][0] += W
			IM_size[2][1] += H
			Class_size[2] += 1
		elif label == 'Pedestrian':
			IM_size[3][0] += W
			IM_size[3][1] += H
			Class_size[3] += 1
		elif label == 'Person_sitting':
			IM_size[4][0] += W
			IM_size[4][1] += H
			Class_size[4] += 1
		elif label == 'Cyclist':
			IM_size[5][0] += W 
			IM_size[5][1] += H
			Class_size[5] += 1
		elif label == 'Tram':
			IM_size[6][0] += W
			IM_size[6][1] += H
			Class_size[6] += 1
		elif label == 'Misc':
			IM_size[7][0] += W
			IM_size[7][1] += H
			Class_size[7] += 1
	
	total_W = 0
	total_H = 0

	for i in range(0,classnum):
		IM_size_mean_class[i][0] = IM_size[i][0]/Class_size[i]
		IM_size_mean_class[i][1] = IM_size[i][1]/Class_size[i]
		total_W += IM_size[i][0]
		total_H += IM_size[i][1]
	IM_size_mean_total = [total_W/Class_size.sum(), total_H/Class_size.sum()]
	return Class_size, IM_size, IM_size_mean_class, IM_size_mean_total

def resize_image(file_names,sourcepath):
	direct2train(sourcepath)
	newW = 102
	newH = 73
	for image in file_names:
		im = Image.open(image)	
		resized = im.resize((newW, newH), Image.ANTIALIAS)
		resized.save('/home/asy/Documents/train-meansize/'+image)
	print('Finished')


def move_file_per_split(file_names, sourcepath, targetpath):
	ratio = 1-0.75
	index = []
	for i in range(0, math.ceil(len(file_names)*ratio)): #int(len(file_names)*ratio)
		x = randint(0,len(file_names)-1)
		while np.in1d(x,index):
			#print('Same value Detected')
			x = randint(0,len(file_names)-1)
		index.append(x)
		os.rename(sourcepath+file_names[x],targetpath+file_names[x])


def main():
	print('enter main...')
	#file_num, file_names = get_file_names(imagePath)
	#Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc = split_class_name(file_names)
	#move_file_per_split(Misc, imagePath, splitPath)
	# file_num, file_names = get_file_names(imagePath)
	# Class_size, IM_size, IM_size_mean_class, IM_size_mean_total = get_image_size(file_names,imagePath)
	# print('Mean size total',list(np.round_(IM_size_mean_total)))
	# print('Mean size class',list(np.round_(IM_size_mean_class)))
	# print('Class_size',Class_size)
	# print('IM_size',IM_size)
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
	import os
	from os import listdir
	from os.path import isfile, join
	import numpy as np
	from PIL import Image
	from random import randint
	import math
	imagePath = '/home/asy/Documents/train-meansize/'
	splitPath = '/home/asy/Documents/split-val/split-misc/'
	main()


# meanpath = '/home/asy/Documents/train-meansize/'
# splitPathTrain = '/home/asy/Documents/split-train/'
# splitPathValidate = '/home/asy/Documents/split-misc/'

# print('Program running, not dead ...')
# file_num, file_names = get_file_names(sourcepath)
# print(file_num)
# print('Object num in total:',len(file_names))
# Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc = split_class_name(file_names)
# print('select Cyclist',len(Misc),len(Misc)*0.25)
