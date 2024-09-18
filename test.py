import os


image_folder = './arrow_images'

if not os.path.exists(image_folder):
    print("The folder does not exist.")
else:
    print(f"Files in the folder: {os.listdir(image_folder)}")