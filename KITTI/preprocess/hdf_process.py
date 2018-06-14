def direct2(path):
	os.chdir(path)

def PIL2array(im):
	imArray = np.asarray(im)
	return imArray

def array2PIL(imArray):
	im = Image.fromarray(np.uint8(imArray))
	return im

def write_h5py(targetName,sourceData1,sourceData2):
	direct2(hdfPath)
	h5f = h5py.File(targetName,'w')
	h5f.create_dataset('dataset_1',data=sourceData1)
	h5f.create_dataset('dataset_2',data=sourceData2)
	h5f.close()

def read_h5py(filename,filePath):
	direct2(filePath)
	h5f = h5py.File(filename,'r')
	output1 = h5f['dataset_1'][:]
	output2 = h5f['dataset_2'][:]
	h5f.close()
	return output1,output2

def generate_2D_imageArray(imagePath):
	file_num, file_names = SP.get_file_names(imagePath)
	shuffle(file_names)
	image_list = []
	label_list = []
	direct2(imagePath)
	for filename in file_names:
		im = Image.open(filename)
		imArray = PIL2array(im)
		label = label_name2num(filename)
		image_list.append(imArray)
		label_list.append([label])
	print('generate_2D_imageArray works fine')
	return image_list, label_list
	

#Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc
def label_name2num(filename):
	name = filename.split('-')[0]
	num = 0
	if name == 'Car':
		num = 0
	elif name == 'Van':
		num = 1
	elif name == 'Truck':
		num = 2
	elif name == 'Pedestrian':
		num = 3
	elif name == 'Person_sitting':
		num = 4
	elif name == 'Cyclist':
		num = 5
	elif name == 'Tram':
		num = 6
	elif name == 'Misc':
		num = 7
	return num

def hdf2RGB(filename):
	imArray, labelArray = read_h5py(filename,hdfPath)
	im_num = len(imArray)
	RGB = np.zeros(3)
	for i in range(0,im_num-1):
		# im = array2PIL(imArray[i])
		# imArray = PIL2array(im)
		RGB += np.mean(imArray[i], axis=(0,1))
	print('Average RGB:',RGB/im_num)
	

def connect_h5py(hdf1, hdf2, hdfNew, path):
	output11, output12 = read_h5py(hdf1,path)
	output21, output22 = read_h5py(hdf2,path)
	sourceData3 = np.concatenate((output11,output21), axis=0)
	sourceData4 = np.concatenate((output12,output22), axis=0)
	write_h5py(hdfNew, sourceData3, sourceData4)

def connect_list():
	path1 = '/home/asy/Documents/temp1/'
	file_num, file_names = SP.get_file_names(path1)
	imageArray1 = []
	labelArray1 = []
	direct2(path1)
	for filename in file_names:
		imageArray1.append(PIL2array(Image.open(filename)))
		labelArray1.append(1)

	path2 = '/home/asy/Documents/temp2/'
	file_num, file_names = SP.get_file_names(path2)
	imageArray2 = []
	labelArray2 = []
	direct2(path2)
	for filename in file_names:
		imageArray2.append(PIL2array(Image.open(filename)))
		labelArray2.append(1)
	imageArray3 = imageArray1+imageArray2
	labelArray3 = labelArray1+labelArray2


def main():
	print('enter main ...')
	image_list, label_list = generate_2D_imageArray(imagePath)
	targetName = 'test-validation.h5'
	write_h5py(targetName,image_list,label_list)
	#filename = 'fulldata.h5'
	# hdf2RGB(filename)
	# filename = 'Car-000016-3.png'
	# image2RGB(filename)
	# output1, output2 = read_h5py('train.h5',hdfPath);
	# hdf1 = 'train_original.h5'
	# hdf2 = 'validation_original.h5'
	# hdfNew = 'fulldata.h5'
	# connect_h5py(hdf1, hdf2, hdfNew, hdfPath)
	# output5, output6 = read_h5py(hdfNew,hdfPath)
	# print('new hdf len',len(output5))
	#connect_list()
#Car 1, Van 2, Truck 3, Pedestrian 4, Person_sitting 5, Cyclist 6, Tram 7, Misc 8

if __name__ == '__main__':
	import os
	os.chdir('/home/asy/Documents/KITTI_process')
	import split_process as SP
	import numpy as np
	import h5py
	from PIL import Image
	import glob
	from random import shuffle
	hdfPath = '/home/asy/Documents/hdf-data/'
	imagePath = '/home/asy/Documents/train-test/'
	main()
