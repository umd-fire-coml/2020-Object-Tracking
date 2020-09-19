import os 
import requests 
from tqdm import tqdm
import zipfile


train_url = 'http://images.cocodataset.org/zips/train2017.zip'
val_url = 'http://images.cocodataset.org/zips/val2017.zip'
annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

def download_url(url, save_path, chunk_size=128):
    print ("Downloading file from %s, saving to %s" % (url, save_path))
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):  
                f.write(chunk)

def unzip(zip_path, save_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

def dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


if os.path.exists("data") == False:
    os.mkdir("data")

if os.path.isfile("data/train2017.zip") == False:
    download_url(train_url, "data/train2017.zip")

if os.path.isfile("data/val2017.zip") == False:
    download_url(val_url, "data/val2017.zip")

if os.path.isfile("data/annotations_trainval2017.zip") == False:
    download_url(annotations_url, "data/annotations_trainval2017.zip")

if os.path.exists("data/train") == False:
    os.mkdir("data/train")
    unzip("data/train2017.zip", "data/train")

if os.path.exists("data/val") == False:
    os.mkdir("data/val")
    unzip("data/val2017.zip", "data/val")

if os.path.exists("data/annotations") == False:
    os.mkdir("data/annotations")
    unzip("data/annotations_trainval2017.zip", "data/annotations")


print ("All files successfully downloaded to data/*")
print ("data/train2017.zip unzipped to data/train")
print ("data/val2017.zip unzipped to data/val")
print ("data/annotations_trainval2017.zip unzipped to data/annotations")
print ("total dataset size: %s bytes" % (dir_size("data/train") + dir_size("data/annotations") + dir_size("data/val")))