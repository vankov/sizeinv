Use make.py to create the data sets. Run `make.py --help` for a list of avialble parameters.

For example:

  python make.py -t 2 -d 4 -noise 0 -min_train_size 1 -max_train_size 10 -image_scale 10
  
This configuration will create a training set of images containing object sized 240x240, 280x280, 320x320, 360x360 and 400x400 px. 
The size of the objects in the test set will be 80x80 px.

The model is trained by running train.py ('train.py --help' lists parameters).

For example:

  python train.py -t 2 -d 4 -noise 0 -gap -pretrained -stop_patience 2 -stop_min_delta 0.01 -epochs 10 -train_steps 2400
  
