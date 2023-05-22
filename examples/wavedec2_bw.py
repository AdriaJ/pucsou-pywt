import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pycsou_pywt as pwt

path = "./data/"
stairs = Image.open(path + "stairs-bc.png")
stairs_bw = stairs.convert("L")
img = np.asarray(stairs_bw)[:-1, :]
print(f"Shape of the image: {img.shape}")

plt.figure()
plt.imshow(
    img,
    cmap="Greys",
    interpolation="none",
)
plt.title("Initial image")
plt.show()

# Wavelet operator
input_shape = img.shape
level = 4
wl_name = "coif5"
op = pwt.WaveletDec2(input_shape=input_shape, wavelet_name=wl_name, level=level)

# Decomposition
double_im = np.stack([img, np.flip(img)]).reshape(2, -1)
flatcoeffs = op(double_im)
coeffs = flatcoeffs.reshape((2, *op.coeff_shape))

plt.figure()
plt.imshow(coeffs[0], cmap="Greys", interpolation="none")
plt.title("Wavelet coefficients")
plt.show()

# Remove HF components on the first image
slices_hf = [s for d in op.coeff_slices[-2:] for s in d.values()]
for s in slices_hf:
    coeffs[0][s] = 0

# Remove LF components on the second image
slices_lf = op.coeff_slices[0:1]
for s in slices_lf:
    coeffs[1][s] = 0

# Reconstruction
img_recon = op.adjoint(coeffs.reshape((2, -1))).reshape((2, *op.input_shape))
print(f"Shape of the reconstructed images: {img_recon.shape}")

plt.figure(figsize=(10, 12))
plt.subplot(221)
plt.imshow(img_recon[0], cmap="Greys", interpolation="none")
plt.title("Low frequency image")

plt.subplot(222)
plt.imshow(np.clip(img - img_recon[0], a_min=0, a_max=255), cmap="Greys", interpolation="none")
plt.title("Difference with input")

plt.subplot(223)
plt.imshow(img_recon[1], cmap="Greys", interpolation="none")
plt.title("High frequency flipped image")

plt.subplot(224)
plt.imshow(np.clip(np.flip(img) - img_recon[1], a_min=0, a_max=255), cmap="Greys", interpolation="none")
plt.title("Differrence with input")
plt.show()
