import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
from typing import List
from PIL import Image
import tqdm
import timm
import logging

BATCH_SIZE = 16
EXP_NAME = "DeiTBase_100epochs_lr1e-4_step10_gamma0.85"

def random_crop(image, size=0.8):

	image = np.array(image, dtype=np.float32)

	height, width, _ = image.shape
	crop_size = int(min(height, width) * size)

	top = np.random.randint(0, height - crop_size)
	left = np.random.randint(0, width - crop_size)
	bottom = top + crop_size
	right = left + crop_size
	image = image[top:bottom, left:right, :]

	return Image.fromarray(np.uint8(image))

class Dataset(torch.utils.data.Dataset):
	def __init__(self, files, mode):
		self.files = files
		self.mode = mode # mode 0...train, 1...valid, 2...test(without label)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		file = self.files[idx]
		image = Image.open(file).convert("RGB")
		image = random_crop(image, size=0.8)
		image = image.resize((224, 224))
		image = torch.from_numpy(np.array(image).astype(np.float32))/255.0
		image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

		label = Path(file).parent.name

		if label == "Cat":
			label = 0
		elif label == "Dog":
			label = 1
		else:
			label = -1

		if self.mode == "train":
			image = transforms.RandomHorizontalFlip()(image)

		return image, label


def get_dataset(root: str, batch_size: int=BATCH_SIZE):
	cat_file = glob.glob(f"{root}\\PetImages\\Cat\\*")
	dog_file = glob.glob(f"{root}\\PetImages\\Dog\\*")
	train_files, test_files = train_test_split(cat_file + dog_file, test_size=0.2, random_state=42)

	train_dataset = Dataset(train_files, mode="train")
	test_dataset = Dataset(test_files, mode="test")

	return train_dataset, test_dataset


def debugging(dataset, img_name):
	img, label = dataset[0]
	pil_image = Image.fromarray(np.uint8(img.permute(1, 2, 0).numpy() * 255))
	plt.imshow(pil_image)
	plt.savefig(f"output/{img_name}_sample.png")
	print(label)
	print(pil_image.size)


class CNN(nn.Module):
	def __init__(self, in_channels=3, num_classes=2):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(64, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.pool(x)
		x = self.classifier(x)
		return x


class DeiTTiny(nn.Module):
	def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
		super().__init__()
		self.model = timm.create_model(
			'deit_tiny_patch16_224.fb_in1k', 
			pretrained=pretrained, 
			num_classes=num_classes
		)

		if freeze_backbone:
			for param in self.model.parameters():
				param.requires_grad = False
			for param in self.model.get_classifier().parameters():
				param.requires_grad = True

	def forward(self, x):
		return self.model(x)


class DeiTSmall(nn.Module):
	def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
		super().__init__()
		self.model = timm.create_model(
			'deit_small_patch16_224.fb_in1k', 
			pretrained=pretrained, 
			num_classes=num_classes
		)

		if freeze_backbone:
			for param in self.model.parameters():
				param.requires_grad = False
			for param in self.model.get_classifier().parameters():
				param.requires_grad = True

	def forward(self, x):
		return self.model(x)


class DeiTBase(nn.Module):
	def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
		super().__init__()
		self.model = timm.create_model(
			'deit_base_patch16_224.fb_in1k', 
			pretrained=pretrained, 
			num_classes=num_classes
		)

		if freeze_backbone:
			for param in self.model.parameters():
				param.requires_grad = False
			for param in self.model.get_classifier().parameters():
				param.requires_grad = True

	def forward(self, x):
		return self.model(x)


def train_model(model, train_loader, criterion, optimizer, device):
	model.train()
	total_loss = 0
	for images, labels in tqdm.tqdm(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * images.size(0)

	return total_loss / len(train_loader.dataset)


def evaluate_model(model, test_loader, device):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_loader:
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return correct * 100 / total


def logger(filename: str):
	Path(filename).parent.mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s [%(levelname)s] %(message)s',
		handlers=[
			logging.FileHandler(filename),
			logging.StreamHandler()
		]
	)


if __name__ == "__main__":
	epochs = 10
	learning_rate = 1e-4
	step_size = 10
	gamma = 0.85
	
	train_dataset, test_dataset = get_dataset(root="C:\\Users\\intern\\compression")

	# For Debugging: 
	"""
	print("torch :", torch.__version__)
	print("CUDA runtime :", torch.version.cuda)
	print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
	debugging(train_dataset, "train")
	debugging(test_dataset, "test")
	"""
	
	# Dataset loading:
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

	# Prepare optimizer, hyperparameters, model, and loss function:
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = DeiTBase(num_classes=2, pretrained=True, freeze_backbone=True).to(device)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	# Training:
	logger(f"output/log/{EXP_NAME}.log")
	logging.info(f"Experiment: {EXP_NAME} | epochs={epochs}, lr={learning_rate}, step_size={step_size}, gamma={gamma}")
	for epoch in range(epochs):
		train_loss = train_model(model, train_loader, criterion, optimizer, device)
		test_accuracy = evaluate_model(model, test_loader, device)
		logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%")
		scheduler.step()