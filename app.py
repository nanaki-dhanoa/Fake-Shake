import os
from PIL import Image
import streamlit as st
import streamlit.components.v1 as com
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown(f"""
<style>
	body {{
	background: "logo_design/logo_dalle2.png";
	}}
</style>
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" target="_blank" style="font-size:35px;" >FAKE <b>SHAKE</b></a>
</nav>
""", unsafe_allow_html=True)

image_dimensions = {'height':256, 'width':256, 'channels':3}

class Classifier:
	def __init__():
		self.model = 0

	def predict(self, x):
		return self.model.predict(x)

	def fit(self, x, y):
		return self.model.train_on_batch(x, y)

	def get_accuracy(self, x, y):
		return self.model.test_on_batch(x, y)

	def load(self, path):
		self.model.load_weights(path)


class Meso4(Classifier):
	def __init__(self, learning_rate=0.001):
		self.model = self.init_model()
		optimizer = Adam(lr=learning_rate)
		self.model.compile(optimizer=optimizer,
						   loss='mean_squared_error',
						   metrics=['accuracy'])

	def init_model(self):
		x = Input(shape=(image_dimensions['height'],
						 image_dimensions['width'],
						 image_dimensions['channels']))

		x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
		x1 = BatchNormalization()(x1)
		x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

		x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
		x2 = BatchNormalization()(x2)
		x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

		x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
		x3 = BatchNormalization()(x3)
		x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

		x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
		x4 = BatchNormalization()(x4)
		x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

		y = Flatten()(x4)
		y = Dropout(0.5)(y)
		y = Dense(16)(y)
		y = LeakyReLU(alpha=0.1)(y)
		y = Dropout(0.5)(y)
		y = Dense(1, activation='sigmoid')(y)

		return Model(inputs=x, outputs=y)


def predict(np_image):
	meso = Meso4()
	meso.load('./weights/Meso4_DF')

	result = meso.predict(np_image)[0][0]
	return result

def save_uploaded_file(uploaded_file):
	try:
		with open(os.path.join('static', uploaded_file.name), 'wb') as f:
			f.write(uploaded_file.getbuffer())
		return 1
	except:
		return 0

def main():
	st.image("logo_design/logo_dalle2.png", use_column_width = True )


	image_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
	font_size = 20
	if image_file is not None:
		our_image = Image.open(image_file)
		np_image = np.array(our_image).astype('float32') / 255
		np_image = transform.resize(np_image, (256, 256, 3))
		np_image = np.expand_dims(np_image, axis=0)
		st.text("Uploaded Image")
		st.image(our_image)
		# save_uploaded_file(image_file)
		
		if st.button("Submit"):
			path = os.path.join('static')
			res = float(predict(np_image))
			if round(res) == 0:
				score = round(((0.5 - res)/0.5)*100, 2)
				html_str = f"""
				<style>
				span.css-10trblm {{
					width = 100%
				}}
				p.a {{
				  font: bold {font_size}px Courier;
				}}
				div.result-fail {{
					background: rgba(178, 34, 34, .8);
					padding: 10px;
					border: 5px solid white;
				}}
				
				</style>
				<div class = "result-fail">
				<p class="a">This image is Fake</p>
				<p class="a">Confidence Score: {score} %</p>
				</div>
				"""
				# st.error('''
				# 		This image is **Fake**
				# 		Confidence Score:
				# 		''')
				st.markdown(html_str, unsafe_allow_html=True)

			else:
				score = round(((res - 0.5) / 0.5) * 100, 2)
				html_str = f"""
				<style>
				p.a {{
				  font: bold {font_size}px Courier;
				}}
				div.result-suc {{
					background: rgba(34, 139, 34, .5);
					padding: 10px;
					border: 5px solid white;
				}}
				
				</style>
				<div class = "result-suc">
				<p class="a">This image is Real</p>
				<p class="a">Confidence Score: {score} %</p>
				</div>
				"""

				st.markdown(html_str, unsafe_allow_html=True)
				st.balloons()

if __name__ == '__main__':
	main()
