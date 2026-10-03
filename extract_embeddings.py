import cv2
import os
import numpy as np
import pickle
from config import DATASET_DIR
from face_preprocessing import preprocess_face
from model_loader import load_facenet_model, predict_embedding


class Extract_Embeddings():

	def __init__(self,model_path):
		self.model_path = model_path		
		self.dataset_dir = str(DATASET_DIR)


	def load_model(self):
		return load_facenet_model(self.model_path)

	def get_embedding(self, model, preprocessed_face):
		return predict_embedding(model, preprocessed_face)

	def check_pretrained_file(self,embeddings_model):
		self.embeddings_model = embeddings_model
		data = pickle.loads(open(embeddings_model, "rb").read())
		names = np.array(data["names"])
		unique_names = np.unique(names).tolist()
		return [data,unique_names]

	def get_staff_details(self):
		details = os.listdir(self.dataset_dir)
		staff_details = {}
		for item in details:
			name = item.split("_")[0]
			id = item.split("_")[1]
			staff_details[name] = id
		return staff_details

	def get_remaining_names(self,dictionaries,unique_names):
		self.dictionaries = dictionaries
		self.unique_names = unique_names
		remaining_names = np.setdiff1d(list(dictionaries.keys()),unique_names).tolist()
		return remaining_names

	def get_all_face_pixels(self,dictionaries):
		image_ids = []
		image_paths = []
		image_arrays = []
		names = []
		face_ids = []
		for category in list(dictionaries.keys()):
			path = os.path.join(self.dataset_dir,category + "_" + dictionaries[category])
			for img in os.listdir(path):
				img_array = cv2.imread(os.path.join(path,img))
				image_paths.append(os.path.join(path,img))
				image_ids.append(img)
				image_arrays.append(img_array)
				names.append(category)
				face_ids.append(dictionaries[category])
		return [image_ids,image_paths,image_arrays,names,face_ids]


	def get_remaining_face_pixels(self,dictionaries,remaining_names):
		self.dictionaries = dictionaries
		self.remaining_names = remaining_names
		image_ids = []
		image_paths = []
		image_arrays = []
		names = []
		face_ids = []
		if len(remaining_names) != 0:	
			for category in list(remaining_names):
				path = os.path.join(self.dataset_dir,category + "_" + dictionaries[category])
				for img in os.listdir(path):
					img_array = cv2.imread(os.path.join(path,img))
					image_paths.append(os.path.join(path,img))
					image_ids.append(img)
					image_arrays.append(img_array)
					names.append(category)
					face_ids.append(dictionaries[category])
			return [image_ids,image_paths,image_arrays,names,face_ids]
		else:
			return None

	def normalize_pixels(self,imagearrays):
		"""Compatibility name for shared per-face FaceNet preprocessing."""
		face_pixels = np.asarray(imagearrays)
		if face_pixels.ndim == 3:
			return preprocess_face(face_pixels)
		if face_pixels.ndim != 4:
			raise ValueError("Expected one image or a collection of face images.")
		return np.asarray([preprocess_face(face) for face in face_pixels])






				
