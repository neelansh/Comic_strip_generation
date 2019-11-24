import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

data_dir = "garfield_data_original/"
mask_dir = "target/"
targ_dir = "saved/"
img = os.listdir(data_dir)
masks = os.listdir(mask_dir)
for name in tqdm((masks)):
	image = cv2.imread(data_dir + name)[:,:,::-1]
	mask = cv2.imread(mask_dir + name,cv2.IMREAD_GRAYSCALE)
	t, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
	for i in range(len(mask)):
		for j in range(len(mask[i])):
			if(mask[i][j] > 100):
				mask[i][j] = 1
	kernel = np.ones((5,5)) 
	mask = cv2.dilate(mask, kernel, iterations=2) 
	imgc = image.copy()

	if imgc.shape[:2]!=mask.shape:
		coldiff = abs(imgc.shape[1]-mask.shape[1])
		rowdiff = abs(imgc.shape[0]-mask.shape[0])
		mask = mask[rowdiff:, coldiff:]

	for channel in range(3):
		imgc[:,:,channel] *= mask

	final = Image.fromarray(imgc)
	final.save(targ_dir + name)
