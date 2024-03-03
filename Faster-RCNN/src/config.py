import torch
BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 30 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '/home/haziq/Documents/VIP-Project/Detection/VOC2007/RCNN/dataset/Pest/train_3_classes/'
# validation images and XML files directory
VALID_DIR = '/home/haziq/Documents/VIP-Project/Detection/VOC2007/RCNN/dataset/Pest/valid_3_classes/'
# classes: 0 index is reserved for background
#CLASSES = [
#    'background', 'army worm', 'legume blister beetle', 'red spider', 'rice gall midge', 
#    'rice leaf roller', 'rice leafhopper', 'rice water weevil', 'wheat phloeothrips', 
#    'white backed plant hopper', 'yellow rice borer'
#]
CLASSES = [
    'background', 'yellow rice borer','rice leaf roller', 'army worm'
]
NUM_CLASSES = 4 # number of classes
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs