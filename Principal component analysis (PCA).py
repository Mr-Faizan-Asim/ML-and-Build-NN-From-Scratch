import numpy as np
from sklearn.decomposition import PCA

# Sample data
data = np.array([[1, 76, 3], [4, 5, 6], [7, 8, 9]])

# Apply PCA
pca = PCA(n_components=3)
transformed_data = pca.fit_transform(data)

print("Original Data:")
print(data)
print("Transformed Data (PCA):")
print(transformed_data)
