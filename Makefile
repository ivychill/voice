prepare:
	make -C input/person


train:
	CUDA_VISIBLE_DEVICES=2 python train.py --config configs/000_ResNet50.yml
