# Second run: 

Using training/run2_train_cfg.yaml and default Yolo Augmentations: 

- albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, num_output_channels=3, method='weighted_average'), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

# Learnings:

- autobatch only works with CUDA: default of 16 is too large
- check why images: d1_02_04_2020_IMG_1092.JPG and d1_02_04_2020_IMG_1093.JPG are corrupt / out of bounds labels
- dont use optimizer=auto but set it to Adam

## Dataset holo_v1

### train: 891 imgs
- D1: 98 
- D2: 150 
- D3: 643

  ### val: 144 imgs
  - D1: 15
  - D2: 23
  - D3: 106
 
### test: 67 imgs
- D1: 12
- D2: 24
- D3: 31
  
