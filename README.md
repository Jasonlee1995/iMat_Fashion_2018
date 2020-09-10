# iMaterialist Challenge on Fashion 2018


## 1. Directory Structure
```
iMat_Fashion_2018
├── data
│   ├── train.json
│   ├── train.csv
│   ├── validation.json
│   ├── val.csv
│   ├── train
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── val
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
├── download_image.ipynb
├── label_map_228.csv
├── README.md
├── Train.ipynb
└── utils
    ├── dataset.py
    ├── download_utils.py
    └── train.py
```
- Cause of file size problem, I only provide source codes not the data
- If you want to follow up, see Section 2. Steps to Follow-up


## 2. Steps to Follow-up
1. Download train.json and place it following Section 1. Directory Structure [[link]](https://drive.google.com/file/d/1oh_GDZY2IQwB_eKCV1ZbWiXkVe5WGEG-/view)
2. Download validation.json and place it following Section 1. Directory Structure [[link]](https://drive.google.com/file/d/11FiOABXkkidTZbNse1zg6HnqLay_0XL5/view)
3. Run download_image.ipynb
  * this scripts make train.csv, val.csv and download train images, val images using multiprocessing
  * train.csv, val.csv : labels of each train/val images
4. Run Train.ipynb


## 3. Baseline
- Baseline is provided using resnet of pytorch with 18, 34, 50, 101 depth


## 4. Reference
- Official dataset github : [imat_fashion_comp](https://github.com/visipedia/imat_fashion_comp)
