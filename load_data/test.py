import itk
import numpy as np
import zlib

file_path = '/data/8T/lyh/data/data_k/Test/10284/10284.nii'
image = itk.imread(file_path)
image_array = itk.array_view_from_image(image)
print(image_array)
max_val = np.max(image_array)
min_val = np.min(image_array)
image_array = ((image_array - min_val) / (max_val - min_val)) * 255
image_array = image_array.astype(np.uint8)
array_compress = zlib.compress(image_array.flatten())
shape = image_array.shape

data = {'shape': shape,
        'data': array_compress, }
print(data)
