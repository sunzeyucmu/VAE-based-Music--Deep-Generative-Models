import os
import tensorflow as tf
import argparse

def main():
	print(f"{tf.__version__}")
	print(f"Hello World!")

	print(tf.test.is_gpu_available())
	print(tf.config.list_physical_devices('CPU'))
	print(tf.config.list_physical_devices('GPU'))
	print(tf.config.list_physical_devices('TPU'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_arg', type=str, default='hhh test')
	conf = parser.parse_args()

	print("Hello World", conf.test_arg)

	main()
