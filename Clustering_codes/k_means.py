import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

image = plt.imread("ic_images/um_000008.png")
img_main = cv2.imread('ic_images/um_000008.png')

height, width, _ = image.shape

image_pixels = np.reshape(img_main, (height * width, 3))#Reshape the image to a 2D array of pixels and 3 color values (RGB)

#See the best number of clusters
# sum_qr_errors = []
# for k in range(1, 10):
#     kmeans = KMeans(n_clusters=k)
#     cluster_labels = kmeans.fit_predict(image_pixels)
#     sum_qr_errors.append(kmeans.inertia_)


# plt.plot(range(1, 10), sum_qr_errors)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sum of square errors')
# plt.show()

number_clusters = 7
kmeans = KMeans(n_clusters=number_clusters)
cluster_labels = kmeans.fit_predict(image_pixels)
cluster_centers = kmeans.cluster_centers_

print(cluster_labels[0:100]) #Resulting cluster labels

#Reshape back the image to the original dimension (width, height, 3channels)
reconstructed_image = np.reshape(cluster_centers[cluster_labels], (height, width, 3)).astype(np.uint8)

# Display the original and clustered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Clustered Image')
plt.imshow(reconstructed_image)
plt.axis('off')
plt.show()

cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()