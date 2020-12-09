import os 
import requests 
from tqdm import tqdm
import zipfile
import os 


command = "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=https://docs.google.com/uc?export=download&id={}\" -O {} && rm -rf /tmp/cookies.txt"

# COMES FROM LASOT DATASET, https://drive.google.com/drive/folders/1v09JELSXM_v7u3dF7akuqqkVG8T1EK2_

airplane = "1D6xOE5NZ7T8fRYl-ZKcE8R05jXkQ0iev"
bird = "1rSghPD62pKlRE2Owd_nqgUmlsY3ZfwBH"
cat = "1wzeGBT7kKziGCizuS7j7zbH8A1ckG7EJ"
dog = "1bDpvh6GPnkhVFv3jag1Kob99D1ItyeSF"
skateboard = "1dT0tcujrHl3uhSIg9fqIIGXjpttDJ5GX"

def download_url(id_, name, save_path, chunk_size=128):
    # print ("Downloading file from %s, saving to %s" % (url, save_path))
    # with requests.get(url, stream=True) as r:
    #     r.raise_for_status()
    #     with open(save_path, 'wb') as f:
    #         for chunk in tqdm(r.iter_content(chunk_size=8192)):  
    #             f.write(chunk)
    os.system(command.format(id_, id_, name + ".zip"))

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

categories = {"airplane": airplane, "cat": cat, "bird": bird, "dog": dog, "skateboard": skateboard}

if os.path.exists("data") == False:
    os.mkdir("data")

if os.path.isfile("data/airplane.zip") == False:
    download_url(airplane, "airplane", "data/airplane.zip")
if os.path.isfile("data/bird.zip") == False:
    download_url(bird, "bird", "data/bird.zip")
if os.path.isfile("data/cat.zip") == False:
    download_url(cat, "cat", "data/cat.zip")
if os.path.isfile("data/dog.zip") == False:
    download_url(dog, "dog", "data/dog.zip")
if os.path.isfile("data/skateboard.zip") == False:
    download_url(skateboard, "skateboard", "data/skateboard.zip")



for c in categories:
    if os.path.exists("data/%s" % c) == False:
        os.mkdir("data/%s" % c)
        unzip("data/%s.zip" % c, "data/%s" % c)


print ("All files successfully downloaded to data/*")
print ("total dataset size: %s bytes" % (dir_size("data/")))