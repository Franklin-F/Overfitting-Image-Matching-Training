import os
image_files = [f for f in os.listdir('../arrow_images') if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('A_')]
for i in range(360):
    di = str(i) + '.png'
    if di not in image_files:
        print(di)