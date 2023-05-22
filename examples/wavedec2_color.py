import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pycsou_pywt as pwt

path = "./data/"
stairs = Image.open(path + "stairs-bc.png")
img = np.asarray(stairs)[..., :-1]
print(f"Shape of the image: {img.shape}")

plt.figure()
plt.imshow(
    img,
    interpolation="none",
)
plt.title("Initial image")
plt.show()

# Wavelet operator
imgt = np.moveaxis(img, -1, 0)
input_shape = imgt.shape[1:]
level = 4
wl_name = "coif5"
op = pwt.WaveletDec2(input_shape=input_shape, wavelet_name=wl_name, level=level)

# Decomposition
double_im = np.stack([imgt, np.flip(imgt)]).reshape((2, 3, -1))
flatcoeffs = op(double_im)
coeffs = flatcoeffs.reshape((2, 3, *op.coeff_shape))


fig = plt.figure(figsize=(14, 4))
axes = fig.subplots(1, 3)
for i in range(3):
    ax = axes[i]
    ax.imshow(coeffs[0, i], cmap="Greys", interpolation="none")
    ax.set_title(f"Coefficients on channel {i:d}")
plt.show()


# Remove HF components on first image
slices_hf = [s for d in op.coeff_slices[-2:] for s in d.values()]
for s in slices_hf:
    coeffs[0][(slice(None, None, None),) + s] = 0

# Remove LF components on first image
slices_lf = op.coeff_slices[0:1]
for s in slices_lf:
    coeffs[1][(slice(None, None, None),) + s] = 0


def clip_round(im: np.ndarray):
    return np.rint(np.clip(np.moveaxis(im, 0, -1), a_min=0, a_max=255)).astype("int")


# Reconstruction
img_recon = op.adjoint(coeffs.reshape((2, 3, -1))).reshape((2, 3, *op.input_shape))
print(f"Shape of the reconstructed images: {img_recon.shape}")

plt.figure(figsize=(10, 12))
plt.subplot(221)
plt.imshow(clip_round(img_recon[0]), interpolation="none")
plt.title("Low frequency image")

plt.subplot(222)
plt.imshow(clip_round(imgt - img_recon[0]), interpolation="none")
plt.title("Differrence with input")

plt.subplot(223)
plt.imshow(clip_round(img_recon[1]), interpolation="none")
plt.title("High frequency flipped image")

plt.subplot(224)
plt.imshow(clip_round(np.flip(imgt) - img_recon[1]), interpolation="none")
plt.title("Differrence with input")
plt.show()
