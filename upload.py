import requests
from PIL import Image
import numpy as np

# f = Image.open("img.jpg")
f = Image.open("4.png")
data = np.array(f.getdata())
# data = np.reshape(data, (32,32,3))
print(data.size)
data = np.reshape(data,(-1))
print(data.size)
print(len(data.tostring()))
r = requests.post('http://localhost:8081', data = data.tostring())
print(r.text)