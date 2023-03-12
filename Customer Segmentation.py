import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def customer_segmentation(data_file):
    # Load the data
    data = pd.read_csv(data_file)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(scaled_data)

    # Add the cluster labels to the data
    data['Cluster'] = kmeans.labels_

    return data

# Get input from the user
data_file = input("Enter the path to the customer data file: ")

# Perform customer segmentation
segmented_data = customer_segmentation(data_file)

# Print the segmented data
print(segmented_data.head())
