import numpy as np

# Initial dataset
X = np.random.rand(1000, 2)  # Features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Original concept: Sum of features

# Introduce drift: Change the concept
def apply_drift(X, drift_type="gradual", iteration=0):
    if drift_type == "abrupt":
        return (X[:, 0] - X[:, 1] > 0).astype(int)  # Abrupt drift
    elif drift_type == "gradual":
        threshold = 0.5 + (iteration / 1000)  # Gradual drift
        return (X[:, 0] > threshold).astype(int)
    elif drift_type == "recurring":
        if iteration % 200 < 100:
            return (X[:, 0] + X[:, 1] > 1).astype(int)
        else:
            return (X[:, 0] - X[:, 1] > 0).astype(int)
    return y  # Default

# Simulate drift over 1000 iterations
for i in range(1000):
    y_drifted = apply_drift(X, drift_type="gradual", iteration=i)



#Feature Distribution Drift
#Change the pixel distributions in the images.
import numpy as np
def add_noise(images, labels, epoch, noise_level=0.1):
    noise = np.random.normal(0, noise_level, images.shape)
    return np.clip(images + noise, 0, 1).float(), labels  # Keep pixel values in [0, 1]

def apply_transform(images, transform_type="rotate"):
    from scipy.ndimage import rotate
    if transform_type == "rotate":
        return np.array([rotate(img, angle=15, reshape=False) for img in images])
    return images  # Add more transforms as needed

# noisy_images = add_noise(original_images)
# rotated_images = apply_transform(original_images)

#Spatial Drift
#Change the position of digits within the image.

import numpy as np
from scipy.ndimage import shift

def shift_images(images, max_shift=3):
    return np.array([shift(img, shift=(np.random.randint(-max_shift, max_shift),
                                       np.random.randint(-max_shift, max_shift)), mode='constant') for img in images])

# shifted_images = shift_images(original_images)


#Feature Masking
#Description: Mask or occlude parts of the images.

import numpy as np

def mask_images(images, mask_size=5):
    masked_images = images.copy()
    for img in masked_images:
        x = np.random.randint(0, images.shape[1] - mask_size)
        y = np.random.randint(0, images.shape[2] - mask_size)
        img[x:x+mask_size, y:y+mask_size] = 0  # Mask with black
    return masked_images

# masked_images = mask_images(original_images)

#Class Imbalance Drift
#Change the frequency of classes in the training dataset.
import numpy as np

def create_imbalanced_data(images, labels, rare_classes=[1, 7], common_classes=[0, 9], imbalance_rate=0.5):
    rare_indices = np.where(np.isin(labels, rare_classes))[0]
    common_indices = np.where(np.isin(labels, common_classes))[0]

    # Reduce rare classes
    rare_indices = rare_indices[:int(len(rare_indices) * imbalance_rate)]
    return images[np.concatenate([rare_indices, common_indices])], labels[np.concatenate([rare_indices, common_indices])]

# imbalanced_images, imbalanced_labels = create_imbalanced_data(original_images, original_labels)


#Temporal Drift
#Gradually alter the distribution of features or labels over time.
def temporal_drift(images, labels, epoch, max_time_steps=999, start_epoch=10):
    if epoch <= start_epoch:
        return images, labels
    drift_factor = epoch / max_time_steps
    noise = np.random.normal(0, drift_factor * 0.1, images.shape)
    return np.clip(images + noise, 0, 1).float(), labels

# for time_step in range(100):  # Example over 100 time steps
#     drifted_images = temporal_drift(original_images, time_step, 100)



#Recurring Drift
#Alternate between different types of image transformations or feature distributions.
def recurring_drift(images, iteration, interval=10):
    if (iteration // interval) % 2 == 0:
        return add_noise(images)  # Add noise
    else:
        return apply_transform(images, transform_type="rotate")  # Rotate

#. Synthetic Feature Drift
#Alter pixel intensity distributions in specific areas of the images.
def modify_pixel_intensity(images, region=(10, 15), scale=1.5):
    modified_images = images.copy()
    modified_images[:, region[0]:region[1], region[0]:region[1]] *= scale
    return np.clip(modified_images, 0, 1)

# intensity_drift_images = modify_pixel_intensity(original_images)

#Style Drift
# Change the style or texture of the digits.
from scipy.ndimage import gaussian_filter

def blur_images(images, sigma=1):
    return np.array([gaussian_filter(img, sigma=sigma) for img in images])

# blurred_images = blur_images(original_images)
