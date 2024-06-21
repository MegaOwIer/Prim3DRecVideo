import os
import cv2
import io
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

folder_path = './test/baseline/output_dir/laptop/visualize_results/'
image_path = './test/video/images/'
video_path = './test/video/video/'

file_name = []
for name in os.listdir(folder_path):
    if name[8] == "4": 
        file_name.append(name)

file_name = sorted(file_name)

for name in file_name: 
    file_path = os.path.join(folder_path, name)
    mesh = trimesh.load(file_path)

    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)
    
    center = [sum(mesh.vertices[:,0])/len(mesh.vertices[:,0]), sum(mesh.vertices[:,1])/len(mesh.vertices[:,1]), sum(mesh.vertices[:,2])/len(mesh.vertices[:,2])]

    ax.plot_trisurf(mesh.vertices[:,0] - center[0], mesh.vertices[:,1] - center[1], triangles=mesh.faces, Z=mesh.vertices[:,2] - center[2], color='yellow', edgecolor='none')
    ax.view_init(30, 90, 60)

    ax.axis('off')

    output_name = image_path + name[19:21] + '.png'

    #print(type(ax))

    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    #exit(0)

print('Video making started...')

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
cap_fs = 5
size = (2079, 2079)
video = cv2.VideoWriter(video_path + 'result' + '_0004' + '.mp4', fourcc, cap_fs, size)

image_name =  []
for name in os.listdir(image_path):
    image_name.append(name)

image_name = sorted(image_name)

for name in image_name: 
    print(name)
    img_tmp = cv2.imread(image_path + name)
    video.write(img_tmp)

video.release()