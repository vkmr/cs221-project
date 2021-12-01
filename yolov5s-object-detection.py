#!/usr/bin/env python
# coding: utf-8

# ## **Training Yolov5 Object Detection on Website Screenshot Dataset**
# 
# ### **Overview**
# 
# This notebook walks through how to train a Yolov5 object detection model using the [Yolov5-Pytorch repo](https://github.com/ultralytics/yolov5).
# 
# In this specific example, we'll training an object detection model to recognize typesof field on a website page among 8 classes.
# 
# Everything in this notebook is also hosted on this [GitHub repo](https://github.com/vkmr/ensemble-object-detection).
# 
# 
# **Credit to [Roboflow](https://roboflow.ai/) **, whom wrote the first notebook on which much of this is example is based. 
# 
# ### **Our Data**
# 
# We'll be using an open source Website screen dataset called Website Screen Dataset. Our dataset contains 1206 images (and 54,215 annotations!) is hosted publicly on Roboflow [here](https://public.roboflow.com/object-detection/website-screenshots).
# 
# When adapting this example to our own data, we created three datasets in Roboflow: `train`, `valid`  and `test`. In order to train our custom model, we need to assemble a dataset of representative images with bounding box annotations around the objects that we want to detect. And we need our dataset to be in YOLOv5 format.
# In Roboflow, you can choose between two paths: 
# * Convert an existing dataset to YOLOv5 format. Roboflow supports over 30 formats object detection formats for conversion.
# * Upload raw images and annotate them in Roboflow with Roboflow Annotate.
# 
# For baseline, we used the first path of directly using open source dataset mentioned above. In later stages of model evaulation, we used second path to add to existing annotation, followed by adding new images and corresponding annotation. At pre-processing, we enhanced training data by adding augmentation of vertical flip and rotate by 90 degress.
# 
# ### **Our Model**
# 
# We'll be training a Yolov5s model with resolution of input image as 640x640. Yolov5 is one-stage detector and is evolution of original Yolov3 DarkNet architecture. YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset. With Yolov5s trained on coco as pretrained checkpoint, we are doing transfer learning for custom dataset of website page screenshots using PyTorch Framework.
# 
# Mode details on the model is available at [YOLOv5 Docs](https://docs.ultralytics.com/).
# 
# ### **Training**
# We used Google Colab as training vehichle. Model is trained for 150 epochs with a batch size 16 using SGD, weight_decay as 0.0005 and learning rate as 0.01.
#  
# 
# ### **Inference**
# 
# We run inference directly in this notebook, and on three test images contained in the "test" folder from our Dataset. 
# We did run local inference for validating working of Ensemble between Yolov5 and Faster-RCNN.

# # Step 1: Install Requirements

# In[1]:


#clone YOLOv5 and 
get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone repo')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt # install dependencies')
get_ipython().run_line_magic('pip', 'install -q roboflow')

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")


# # Step 2: Assemble Our Dataset
# 
# In order to train our custom model, we need to assemble a dataset of representative images with bounding box annotations around the objects that we want to detect. And we need our dataset to be in YOLOv5 format.
# 
# # Export
# 
# ![](https://github.com/roboflow-ai/yolov5/wiki/images/roboflow-export.png)
# 
# # Download Code 
# 
# ![](https://github.com/roboflow-ai/yolov5/wiki/images/roboflow-snippet.png)

# In[8]:


from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")


# In[9]:


# set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"


# In[10]:


get_ipython().system('pip install roboflow')

from roboflow import Roboflow
rf = Roboflow(api_key="dhRTaBhaFiX3fMKYccYI")
project = rf.workspace().project("website-detection")
dataset = project.version(1).download("yolov5")


# # Step 3: Train Our Custom YOLOv5s model
# 
# Here, we are able to pass a number of arguments:
# - **img:** define input image size
# - **batch:** determine batch size
# - **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
# - **data:** Our dataset locaiton is saved in the `dataset.location`
# - **weights:** specify a path to weights to start transfer learning from. Here we choose the generic COCO pretrained checkpoint.
# - **cache:** cache images for faster training

# In[11]:


get_ipython().system('python train.py --img 640 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache')


# # Evaluate Custom YOLOv5 Detector Performance
# Training losses and performance metrics are saved to Tensorboard and also to a logfile.
# 
# If you are new to these metrics, the one you want to focus on is `mAP_0.5` - learn more about mean average precision [here](https://blog.roboflow.com/mean-average-precision/).

# In[12]:


# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs')


# #Run Inference  With Trained Weights
# Run inference with a pretrained checkpoint on contents of `test/images` folder downloaded from Roboflow.

# In[13]:


get_ipython().system('python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf-thres 0.25 --line-thickness 1 --max-det 30 --source {dataset.location}/test/images')


# In[16]:


#Save  conf_thresh and output to text file
get_ipython().system('python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf-thres 0.25 --line-thickness 1 --max-det 30 --save-txt --save-conf --source {dataset.location}/test/images')


# In[17]:


#display inference on ALL test images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp3/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")


# # Conclusion and Next Steps
# 
# We have trained a custom YOLOv5 model to recognize our custom objects.
# 
# To improve our model's performance, we followed coverage and quality. See this guide for [model performance improvement](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results).

# In[18]:


#export your model's weights for future use
from google.colab import files
files.download('./runs/train/exp/weights/best.pt')


# In[20]:


get_ipython().system('zip -r /content/exp6.zip /content/yolov5/runs/detect/exp3/')


# In[ ]:


from google.colab import files
files.download('/content/exp6.zip')

