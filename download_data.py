
import os
import urllib.request

# Create the data folder if it does not exist
os.makedirs("data", exist_ok=True)

# Official Stanford SNAP dataset URL
url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"

# Location where the dataset will be saved
file_path = os.path.join("data", "facebook_combined.txt.gz")

try:
    print("Downloading Facebook dataset...")

    urllib.request.urlretrieve(url, file_path)

    print("\nDataset downloaded successfully!")
    print("File saved at:", os.path.abspath(file_path))
    print("File size:", round(os.path.getsize(file_path) / (1024 * 1024), 2), "MB")

except Exception as error:
    print("Download failed:", error)
