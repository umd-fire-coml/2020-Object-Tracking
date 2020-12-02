import os 
import requests 
from tqdm import tqdm
import zipfile

# COMES FROM LASOT DATASET, https://drive.google.com/drive/folders/1v09JELSXM_v7u3dF7akuqqkVG8T1EK2_

airplane = "https://doc-0c-4g-docs.googleusercontent.com/docs/securesc/72qdeuqpu88c54igairjr150srdupp9l/66hqv4ruus5qea1bblja96g3mpc9j93g/1606931400000/02212077843227030137/09233892213268940709/1D6xOE5NZ7T8fRYl-ZKcE8R05jXkQ0iev?e=download&authuser=0&nonce=4qgp38ca51uku&user=09233892213268940709&hash=vtspbsau429rk0m2233iuej4h1fru7mn"
bird = "https://doc-14-4g-docs.googleusercontent.com/docs/securesc/72qdeuqpu88c54igairjr150srdupp9l/6liqmg52beajfeu916p10an13vhqtuon/1606931625000/02212077843227030137/09233892213268940709/1rSghPD62pKlRE2Owd_nqgUmlsY3ZfwBH?e=download&authuser=0"
cat = "https://doc-10-4g-docs.googleusercontent.com/docs/securesc/72qdeuqpu88c54igairjr150srdupp9l/tt4h38vs6cu4kuh3vkuo1j8k20clmpuv/1606931550000/02212077843227030137/09233892213268940709/1wzeGBT7kKziGCizuS7j7zbH8A1ckG7EJ?e=download&authuser=0"
dog = "https://doc-0k-4g-docs.googleusercontent.com/docs/securesc/72qdeuqpu88c54igairjr150srdupp9l/ns52og16dv5d6f37mqhrekffi7mesf26/1606931550000/02212077843227030137/09233892213268940709/1bDpvh6GPnkhVFv3jag1Kob99D1ItyeSF?e=download&authuser=0"
skateboard = "https://doc-08-4g-docs.googleusercontent.com/docs/securesc/72qdeuqpu88c54igairjr150srdupp9l/0ft07v61fkgijqb70j1cifuuqv0ebrvm/1606931475000/02212077843227030137/09233892213268940709/1dT0tcujrHl3uhSIg9fqIIGXjpttDJ5GX?e=download&authuser=0"


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

categories = {"airplane": airplane, "cat": cat, "bird": bird, "dog": dog, "skateboard": skateboard}

if os.path.exists("data") == False:
    os.mkdir("data")

if os.path.isfile("data/airplane.zip") == False:
    download_url(airplane, "data/airplane.zip")
if os.path.isfile("data/bird.zip") == False:
    download_url(bird, "data/bird.zip")
if os.path.isfile("data/cat.zip") == False:
    download_url(cat, "data/cat.zip")
if os.path.isfile("data/dog.zip") == False:
    download_url(dog, "data/dog.zip")
if os.path.isfile("data/skateboard.zip") == False:
    download_url(skateboard, "data/skateboard.zip")



for c in categories:
    if os.path.exists("data/%s" % c) == False:
        os.mkdir("data/%s" % c)
        unzip("data/%s.zip" % c, "data/%s" % c)


print ("All files successfully downloaded to data/*")
print ("total dataset size: %s bytes" % (dir_size("data/"))