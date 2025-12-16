# tugas1_glcm.py
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import numpy as np

def process(filepath, angle_deg):
    image = io.imread(filepath)
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    image = img_as_ubyte(image)

    angle_rad = np.deg2rad(angle_deg)
    glcm = graycomatrix(image, [1], [angle_rad], symmetric=True, normed=True)

    return {
        "Contrast": float(graycoprops(glcm, 'contrast')[0, 0]),
        "Correlation": float(graycoprops(glcm, 'correlation')[0, 0]),
        "Energy": float(graycoprops(glcm, 'energy')[0, 0]),
        "Homogeneity": float(graycoprops(glcm, 'homogeneity')[0, 0])
    }
