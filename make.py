from __future__ import print_function

import numpy as np
from PIL import Image
import os
import sys
import glob
import shutil

from argparser import args

image_w = args.image_width
image_h = args.image_height

max_x_noise = args.noise
max_y_noise = args.noise

min_train_size = args.min_train_size
max_train_size = args.max_train_size
size_scale_factor = args.image_scale

test_size = args.t
train_sizes = []

for i in range(min_train_size, test_size - args.d + 1):
    train_sizes.append(i)

for i in range(test_size + args.d, max_train_size + 1):    
    train_sizes.append(i)


sim_id_t = 0

shapes_hash = {}

def clean_up(output_dir):
    for f in glob.glob('{}/*'.format(output_dir)):
        shutil.rmtree(f)
        
def gen_stimulus(shape_no, size, output_dir, category):
    global sim_id_t, shapes, image_h, image_w, size_scale_factor
    
    img = Image.new(mode='RGB', size=(image_w, image_h), color=(255, 255, 255))
    shape_size_id = "{}.{}".format(shape_no, size)


    shape_w = int(image_w / size_scale_factor) * size
    shape_h = int(image_h / size_scale_factor) * size


    if shape_size_id in shapes_hash:
        shape = shapes_hash[shape_size_id]
    else:
        shape = Image.open("{}/{}.png".format(shapes_dir, shape_no))
               
        shape = shape.resize((shape_w, shape_h), Image.ANTIALIAS)
        
        shapes_hash[shape_size_id] = shape

               
    noise_x = int((max_x_noise / 2.0) - np.random.rand() * max_x_noise)
    noise_y = int((max_y_noise / 2.0) - np.random.rand() * max_y_noise)
    
    hor_location = int(image_w / 2.0 - (shape_w / 2.0))
    top_location = int(image_h / 2.0 - (shape_h / 2.0))
    
    x = hor_location + noise_x
    y = top_location + noise_y
    
    img.paste(shape, box=(x, y))

    shape_output_dir = "{}/{}".format(output_dir, category)
    
    if not(os.path.exists(shape_output_dir)):
        os.makedirs(shape_output_dir)
        
    sim_id_t += 1

    stim_id = sim_id_t
    stim_file = "{}/{}.png".format(shape_output_dir, stim_id)
    
    with open(stim_file, "w") as f:
        img.save(f)
    
#cleaning up trainng and test set    
clean_up("./data/train/{}.{}.{}".format(args.t, args.d, args.noise))
clean_up("./data/test/{}.{}.{}".format(args.t, args.d, args.noise))
    
#Ryan's data, Leek    
shapes_dir = "./shapes/"
if args.v:
    n_stimuli = (args.n * len(train_sizes) * args.n_shapes, args.n * args.n_shapes)
    print("Generating train (n = {}) and test (n = {}) data...".format(
            n_stimuli[0], n_stimuli[1]), end='')
    sys.stdout.flush()

shapes = np.arange(1, args.n_shapes + 1)
np.random.shuffle(shapes)

for i in range(args.n):    
    for shape_no in range(args.n_shapes):
        category = 1 if shape_no < int(args.n_shapes / 2) else 0
        
        for sz in train_sizes:
            gen_stimulus(
                    shapes[shape_no], 
                    sz, 
                    "./data/train/{}.{}.{}".format(
                            args.t, 
                            args.d, 
                            args.noise
                        ), 
                    category = category
                )
                    
        gen_stimulus(
                shapes[shape_no], 
                args.t, 
                "./data/test/{}.{}.{}".format(
                        args.t, 
                        args.d, 
                        args.noise
                    ), 
                category = category
            )        
    
if args.v:
    print("Done.")