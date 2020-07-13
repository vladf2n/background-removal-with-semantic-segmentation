from torchvision import models
from opencv_transforms import transforms as cvtransforms
import torch
import numpy as np
import cv2
import os, ssl


# Function used to create a mask to segment people in images
def segmentation_mask(input_image):

    mask = np.zeros_like(input_image).astype(np.uint8)

    # The class label associated with people in our trained dataset is 15
    # so we will transform every pixel == 15 to 255 and the others will become 0 (background)
    person_class = 15
    pixels_from_person = input_image == person_class
    mask[pixels_from_person] = 255
    
    # Just some suavization to make the borders better
    mask = cv2.GaussianBlur(mask,(3,3),0)

    return mask



# Disable ssl verification (This was used to download the trained dataset)
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Defining the convolutional network with a pretrained dataset that will be used to detect the person in the image
fully_convolutional_network = models.segmentation.fcn_resnet101(pretrained=True).eval()

input_image = cv2.imread('test_images/input.jpg')

# Preprocessing the image and making it become a tensor so it can be used in the convolutional network
image_to_tensor_transform = cvtransforms.Compose([cvtransforms.ToTensor(), 
                                        cvtransforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

input_tensor = image_to_tensor_transform(input_image).unsqueeze(0)

# Passing the input structure through the net to get the ouput model with the person classified
output_model = fully_convolutional_network(input_tensor)['out']

# Transforming the tensor into n labeled image
labeled_image = torch.argmax(output_model.squeeze(), dim=0).detach().cpu().numpy()

# Generating the segmentation mask to remove the background
mask = segmentation_mask(labeled_image)

# Removing the background leaving only the person
result_image = cv2.bitwise_and(input_image, input_image, mask=mask.astype(np.uint8))

cv2.imshow('test', result_image) 
cv2.waitKey(0)  
cv2.destroyAllWindows()