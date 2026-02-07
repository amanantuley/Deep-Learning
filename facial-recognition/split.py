import os, shutil, random

for person in ["aman","azlan" , "sumaiya", "others"]:

    src = f"data/train/{person}"
    dst = f"data/test/{person}"

    os.makedirs(dst, exist_ok=True)

    images = os.listdir(src)
    random.shuffle(images)

    split = int(0.2 * len(images))   # 20% test

    for img in images[:split]:
        shutil.move(f"{src}/{img}", f"{dst}/{img}")
