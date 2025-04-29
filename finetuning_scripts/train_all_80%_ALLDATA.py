#%% Install the required libraries

#!pip install git+https://github.com/facebookresearch/segment-anything.git
#!pip install -q git+https://github.com/huggingface/transformers.git
#!pip install datasets
#!pip install -q monai
#!pip install patchify

#SAM, Transformers, Datasets to prepare data and monai if you want to use special loss functions
#Patchify to divide large images into smaller patches for training. (Not necessary for smaller images)


import numpy as np
import matplotlib.pyplot as plt
import tifffile

from datasets import concatenate_datasets
import os
from patchify import patchify  #Only to handle large images

#### - PARAMETERS - ####


#Load the model and the saved weights
model_name = "flaviagiammarino/medsam-vit-base" #medsam
saved_weights="save/save_12/model_checkpoint_12.pth"

#select the number of datasets to train on
first_dataset=18
last_dataset=19

#Desired patch size for smaller images and step size.
patch_size = 256
step = 128
batchsize = 2

#Number of epochs
num_epochs = 10

main_folder = "data"

def create_dataset(large_images,large_masks,patch_size,step):
    all_img_patches = []
    for img in range(large_images.shape[0]):
        large_image = large_images[img]
        patches_img = patchify(large_image, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):

                single_patch_img = patches_img[i,j,:,:]
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)

    #Let us do the same for masks
    all_mask_patches = []
    for img in range(large_masks.shape[0]):
        large_mask = large_masks[img]
        patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):

                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
                all_mask_patches.append(single_patch_mask)

    masks = np.array(all_mask_patches)

    #reduce the number of patches to 10 for testing
    #images=images[:10]
    #masks=masks[:10]

    # Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]

    # Filter the image and mask arrays to keep only the non-empty pairs
    filtered_images = images[valid_indices]
    filtered_masks = masks[valid_indices]
    #print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
    #print("Mask shape:", filtered_masks.shape)

    from datasets import Dataset # Import the Dataset class from the datasets module here cuz problem with dataset from pythorch
    from PIL import Image

    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    dataset_dict = {
        "image": [Image.fromarray(img) for img in images],
        "label": [Image.fromarray(mask) for mask in masks],
    }

    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(dataset_dict)

    return dataset

def save_new_name(i,epoch_losses):
    i+=1
    save_dir=f"save/save_{i}"
    
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    save_name = f"model_checkpoint_{i}.pth"

    # Save the model's state dictionary to a file

    np.save(os.path.join(save_dir, f"pertes_{save_name}.npy"), epoch_losses)  # Enregistrer les pertes dans un fichier .npy
    
    return save_dir,save_name

def training_with_one_set(dataset,model_name,saved_weights,num_epochs):
    #Get bounding boxes from mask.
    def get_bounding_box(ground_truth_map):
      # get bounding box from mask
      y_indices, x_indices = np.where(ground_truth_map > 0)
      x_min, x_max = np.min(x_indices), np.max(x_indices)
      y_min, y_max = np.min(y_indices), np.max(y_indices)
      # add perturbation to bounding box coordinates
      H, W = ground_truth_map.shape
      x_min = max(0, x_min - np.random.randint(0, 20))
      x_max = min(W, x_max + np.random.randint(0, 20))
      y_min = max(0, y_min - np.random.randint(0, 20))
      y_max = min(H, y_max + np.random.randint(0, 20))
      bbox = [x_min, y_min, x_max, y_max]

      return bbox

    from torch.utils.data import Dataset

    class SAMDataset(Dataset):
      """
      This class is used to create a dataset that serves input images and masks.
      It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
      """
      def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

      def __len__(self):
        return len(self.dataset)

      def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # get bounding box prompt
        #prompt = get_bounding_box(ground_truth_mask)
        prompt = [0,0,255,255] #This line is added to avoid the error "TypeError: object of type 'int' has no len()"

        # Convert the image to RGB by stacking it three times
        image = np.stack((image,) * 3, axis=-1) #This line is added to convert the image to RGB format.


        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

    # Initialize the processor
    from transformers import SamProcessor
    processor = SamProcessor.from_pretrained(model_name)

    # Create an instance of the SAMDataset
    train_dataset = SAMDataset(dataset=dataset, processor=processor)

    # Create a DataLoader instance for the training dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=False)

    import torch

    # Load the model
    from transformers import SamModel
    model = SamModel.from_pretrained(model_name)
    if saved_weights != "None": model.load_state_dict(torch.load(f"{saved_weights}"))

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
      if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

    from torch.optim import Adam
    import monai

    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    from tqdm import tqdm
    from statistics import mean
    from torch.nn.functional import threshold, normalize

    #Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

        save_dir,save_name=save_new_name(epoch,epoch_losses)
        torch.save(model.state_dict(), os.path.join(save_dir, save_name))


def main():
   
    #Load tiff stack images and masks
    #large_images = tifffile.imread("data/data1/D1-1M-sample1-hydrogel.tif")
    #large_masks = tifffile.imread("data/data1/D1-1M-sample1-spheroid.tif")


    datasets = []

    
   # Parcourir les 19 sous-dossiers
    for i in range(first_dataset, last_dataset+1):  # Les dossiers vont de data1 à data19. On s'arrête à 16 pour garder un jeu de test de 20% du dataset total
        subfolder = os.path.join(main_folder, f"data{i}")
        if os.path.exists(subfolder) and os.path.isdir(subfolder):
        
            for filename in os.listdir(subfolder):
    
                # Trouver le fichier TIFF contenant "hydrogel" dans le nom
                if "hydrogel" in filename and filename.endswith(".tif"):
                    image_path = os.path.join(subfolder, filename)

                # Trouver le fichier TIFF contenant "spheroid" dans le nom                            
                if "spheroid" in filename and filename.endswith(".tif"):
                    mask_path = os.path.join(subfolder, filename)
        
        print("dataset", i,)
        print("image_path:", image_path, "mask_path", mask_path)

        datasets.append(create_dataset(tifffile.imread(image_path),tifffile.imread(mask_path),patch_size,step))

    combined_dataset = concatenate_datasets([e for e in datasets]) 
    
    print("Combined dataset:", combined_dataset)

    training_with_one_set(combined_dataset,model_name,saved_weights,num_epochs)
    

if __name__ == "__main__":
    main()
