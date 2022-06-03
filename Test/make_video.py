
import cv2
import os

image_folder = 'vis_hotdog_spp5'
video_name = f'{image_folder}.avi'

images = []
for i in range(256):
    images.append(f'output_vis{i}.jpg')



frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()