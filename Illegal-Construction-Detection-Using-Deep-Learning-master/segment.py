import numpy
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np

map_base_dir = 'D:/College_CP/SDM/Segment'
map_img_dir = 'D:/College_CP/SDM/Segment/train/images'
output_dir = 'C:/Users/USER/Desktop/SEM-VI/Deep Learning/CourseProject/static/predicted_images_segment'

json_path = os.path.join(map_base_dir, 'annotation-small.json')
with open(json_path, 'r') as f:
    annot_data = json.load(f)

image_df = pd.DataFrame(annot_data['images'])
annot_df = pd.DataFrame(annot_data['annotations'])
full_df = pd.merge(annot_df, image_df, how='left', left_on = 'image_id', right_on='id').dropna()

def create_boxes(in_rows):
    #TODO: this seems to get a few of the boxes wrong so we stick to segmentation polygons instead
    box_list = []
    for _, in_row in in_rows.iterrows():
        # bbox from the coco standard
        (start_y, start_x, wid_y, wid_x) = in_row['bbox']
        
        box_list += [Rectangle((start_x, start_y), 
                         wid_y , wid_x
                         )]
    return box_list

def plot_image(image_id):
    # print(type(image_id))
    # image_id = image_id.lstrip("0")
    print(image_id)
    # Get the dataframe for the selected image
    c_df = full_df[full_df['image_id'] == image_id]

    # Load the image
    img_data = imread(os.path.join(map_img_dir, c_df['file_name'].values[0]))

    # Plot the image and annotations
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.imshow(img_data)
    ax.add_collection(PatchCollection(create_boxes(c_df), alpha = 0))
    for _, c_row in c_df.iterrows():
        xy_vec = np.array(c_row['segmentation']).reshape((-1, 2))
        ax.plot(xy_vec[:, 0], xy_vec[:, 1], label = c_row['id_x'])

    # Save the figure in the output directory
    output_path = os.path.join(output_dir, 'segment.png')
    plt.savefig(output_path)
    return output_path