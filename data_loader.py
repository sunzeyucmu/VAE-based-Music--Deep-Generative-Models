import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# importing PIL
import PIL
from PIL import Image


GENRES = {'metal': 0, 'disco': 1, 'classical': 2, 'rock': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'hiphop': 9}

IMG_DATA_PATH = "../../../GENERATIVE_DATA/GTZAN/images_original/"





if __name__ == '__main__':

	for genere, _ in GENRES.items():
		folder = IMG_DATA_PATH + genere
		print(f"Loading Audio Img files under Genere: {genere}")
		for root, subdirs, files in os.walk(folder):
			for file in files[:1]:
				file_name = folder + "/" + file
				print(f"fileName: {file_name}")

				print("Loading Img...")
				img = Image.open(file_name)
				img_arr = np.array(img)
				print(f"Image Shape: {img_arr.shape}")

				plt.imshow(img_arr)

	print(IMG_DATA_PATH)