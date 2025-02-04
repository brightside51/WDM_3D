# Library Imports
import os
import random
import argparse
import numpy as np
import pydicom
import torch
import torchvision
import matplotlib.pyplot as plt

# Function Imports
from pathlib import Path
from ipywidgets import interactive, IntSlider
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import SimpleITK as sitk

# N4ITK Bias Correction Algorithm
def n4_bias_correction(image):
    
    # N4ITK Bias Field Correction
    image = numpy_to_sitk(image.cpu().numpy())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image)
    return torch.Tensor(sitk.GetArrayFromImage(corrected_image)).unsqueeze(0)
    
    # Save the corrected image
    #sitk.WriteImage(corrected_image, output_path)

def numpy_to_sitk(image_np, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    """
    Convert a NumPy 3D array to a SimpleITK Image with metadata.
    """
    sitk_image = sitk.GetImageFromArray(image_np)
    sitk_image.SetSpacing(spacing)      # Set voxel spacing if available
    sitk_image.SetOrigin(origin)        # Set origin (usually [0, 0, 0])
    sitk_image.SetDirection(direction)  # Set direction cosine matrix
    return sitk_image

# Non-Conditional MetaBrest Dataset Reader Class (V4)
class NCDataset(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        dataset: str = 'private',
        mode: str = 'train',
    ):  
        
        # Dataset Choice
        super(NCDataset).__init__(); self.settings = settings
        self.mode = mode; self.dataset = dataset
        if self.dataset == 'public': self.data_folderpath = self.settings.public_data_folderpath
        elif self.dataset == 'lung': self.data_folderpath = self.settings.lung_data_folderpath
        elif self.dataset == 'private': self.data_folderpath = self.settings.private_data_folderpath
        else: print("ERROR: Chosen Dataset / Directory does not exist!")

        # Subject Indexing (Existing or New Version)
        subj_listpath = Path(f"{self.settings.reader_folderpath}/V{self.settings.data_version}" +\
                             f"/{self.dataset}_{self.mode}_setV{self.settings.data_version}.txt")
        if subj_listpath.exists():
            print(f"Reading {self.dataset} Dataset Save Files for {self.mode} Set | Version {settings.data_version}")
            self.subj_list = subj_listpath.read_text().splitlines()
        else:
            print(f"Generating New Save Files for {self.dataset} Dataset | Version {settings.data_version}")
            self.subj_list = os.listdir(self.data_folderpath)       # Complete List of Subjects in Dataset
            self.subj_list.remove('video_data')
            self.subj_list = self.subj_split(self.subj_list)        # Selected List of Subjects in Dataset
        #assert len(self.subj_list) == self.num_subj, f"WARNING: Number of subjs does not match Dataset Version!"

        # --------------------------------------------------------------------------------------------
        
        # Dataset Transformations Initialization
        self.transform = transforms.Compose([
                                        transforms.Resize(( self.settings.img_size,
                                                            self.settings.img_size)),
                                        transforms.ToTensor()])
        self.h_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1)])
        self.v_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1)])

    # ============================================================================================

    # DataLoader Length / No. Subjects Computation Functionality
    def __len__(self): return len(self.subj_list)
    
    # Subject Splitting Functionality
    def subj_split(self, subj_list: list):

        # Dataset Splitting
        train_subj = len(subj_list) - self.settings.val_subj - self.settings.test_subj if self.settings.train_subj == 0 else self.settings.train_subj
        assert 0 < (train_subj + self.settings.val_subj + self.settings.test_subj) <= len(subj_list),\
               f"ERROR: Dataset does not contain {train_subj + self.settings.val_subj + self.settings.test_subj} Subjects!"
        if self.settings.val_subj != 0:
            val_subj = np.sort(np.array(random.sample(subj_list, self.settings.val_subj), dtype = 'str'))
            subj_list = [subj for subj in subj_list if subj not in val_subj]                                    # Validation Set Splitting
        if self.settings.test_subj != 0:
            test_subj = np.sort(np.array(random.sample(subj_list, self.settings.test_subj), dtype = 'str'))
            subj_list = [subj for subj in subj_list if subj not in test_subj]                                   # Test Set Splitting
        train_subj = np.sort(np.array(random.sample(subj_list, train_subj), dtype = 'str'))
        subj_list = [subj for subj in subj_list if subj not in train_subj]                                      # Training Set Splitting
        subj_list = np.sort(np.array(subj_list, dtype = 'str'))
        assert len(subj_list) + len(train_subj) + self.settings.val_subj + self.settings.test_subj == len(self.subj_list),\
               f"ERROR: Dataset Splitting went Wrong!"

        # Dataset Split Saving
        if not os.path.isdir(f"V{self.settings.data_version}"): os.mkdir(f"V{self.settings.data_version}")
        if len(train_subj) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_train_setV{self.settings.data_version}.txt", train_subj, fmt='%s')
        if len(subj_list) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_rest_setV{self.settings.data_version}.txt", subj_list, fmt='%s')
        if self.settings.test_subj != 0:
            if len(test_subj) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_test_setV{self.settings.data_version}.txt", test_subj, fmt='%s')
        if self.settings.val_subj != 0:
            if len(val_subj) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_val_setV{self.settings.data_version}.txt", val_subj, fmt='%s')
 
    # ============================================================================================
        
    # Single Batch / Subject Generation Functionality
    def __getitem__(self, idx: int = 0 or str, save: bool = False):
        subj_idx = idx if type(idx) == str else self.subj_list[idx]

        # MP4 Subject File Reading
        if self.settings.data_format == 'mp4':

            # Subject Data Access
            subj_folderpath = f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}/{subj_idx}.mp4"
            img_data = (torchvision.io.read_video(subj_folderpath, pts_unit = 'sec')[0][:, :, :, 0] / 255.0).type(torch.float32)

        # DICOM Subject File Reading
        elif self.settings.data_format == 'dicom':

            # Subject Folder Access
            subj_folderpath = f"{self.data_folderpath}/{subj_idx}"
            subj_filelist = os.listdir(subj_folderpath)
            for i, path in enumerate(subj_filelist):
                subj_folderpath = f"{self.data_folderpath}/{subj_idx}/{path}"
                subj_filelist = os.listdir(subj_folderpath)
                while os.path.splitext(subj_filelist[0])[1] not in ['.dcm', '.xlm']:
                    subj_folderpath = Path(f"{subj_folderpath}/{subj_filelist[0]}")
                    subj_filelist = os.listdir(subj_folderpath)
                if len(subj_filelist) >= 50: break
            subj_filelist = np.ndarray.tolist(np.sort(subj_filelist))
            
            # Subject General Information Access
            subj_filepath = Path((f"{subj_folderpath}/{subj_filelist[0]}"))
            while os.path.splitext(subj_filepath)[1] not in ['', '.dcm']:
                i += 1; subj_filepath = Path((f"{subj_folderpath}/{subj_filelist[i]}"))
            subj_info = pydicom.dcmread(subj_filepath, force = True)
            og_idx = int(subj_info[0x0020, 0x0013].value)
            subj_ori = subj_info[0x0020, 0x0037].value
            subj_v_flip = (np.all(subj_ori == [-1, 0, 0, 0, -1, 0]))
            subj_h_flip = (torch.rand(1) < (self.settings.h_flip / 100))

            # --------------------------------------------------------------------------------------------
                
            # Subject Slice Data Access
            og_frame = len(subj_filelist) + og_idx + 50 if self.dataset == 'lung' else 100
            img_data = torch.empty((og_frame, self.settings.img_size, self.settings.img_size)); slice_list = []
            for i, slice_filepath in enumerate(np.sort(subj_filelist)):
                if os.path.splitext(slice_filepath)[1] in ['', '.dcm']:
                    
                    # Slice Data Access
                    slice_filepath = Path(f"{subj_folderpath}/{slice_filepath}")
                    slice_data = pydicom.dcmread(slice_filepath, force = True)
                    slice_idx = int(slice_data[0x0020, 0x0013].value)
                    if slice_idx >= len(subj_filelist) + 1: slice_idx -= og_idx 
                    slice_list.append(slice_idx)
                    img_slice = slice_data.pixel_array.astype(float)

                    # Slice Image Pre-Processing | Rescaling, Resizing & Flipping
                    if self.settings.data_prep:
                        img_slice = np.uint8((np.maximum(img_slice, 0) / img_slice.max()) * 255)
                        img_slice = Image.fromarray(img_slice).resize(( self.settings.img_size,
                                                                        self.settings.img_size)) 
                        if subj_h_flip: img_slice = self.h_flip(img_slice)
                        if subj_v_flip: img_slice = self.v_flip(img_slice)
                        img_slice = np.array(self.transform(img_slice))
                    img_data[slice_idx, :, :] = torch.Tensor(img_slice); del img_slice
                else: subj_filelist.remove(slice_filepath)
            #print(f"Accessing Subject {subj_idx}: {len(subj_filelist)} -> {self.settings.num_slice} Slices")
            img_data = img_data[np.sort(slice_list)]

            # --------------------------------------------------------------------------------------------
            
            # Slice Cropping | Spaced-Out Slices
            if self.settings.slice_spacing:
                s_array = slice_array = np.linspace(0 + self.settings.slice_bottom_margin,
                    len(subj_filelist) - self.settings.slice_top_margin - 1, self.settings.num_slice)
                slice_array[0 : int(np.floor(self.settings.num_slice / 2))] = np.ceil(s_array[0 : int(np.floor(self.settings.num_slice / 2))]).astype(int)
                slice_array[int(np.ceil(self.settings.num_slice / 2) + 1)::] = np.floor(s_array[int(np.ceil(self.settings.num_slice / 2) + 1)::]).astype(int)
                slice_array[int(np.floor(self.settings.num_slice / 2))] = np.round(s_array[int(np.floor(self.settings.num_slice / 2))])
                img_data = img_data[slice_array, :, :]

            # Slice Cropping | Middle Slices Only
            else:
                extra_slice = self.settings.num_slice - img_data.shape[0]
                if img_data.shape[0] < self.settings.num_slice:             # Addition of Repeated Peripheral Slices
                    for extra in range(extra_slice):
                        if extra % 2 == 0: img_data = torch.cat((img_data, img_data[-1].unsqueeze(0)), dim = 0)
                        else: img_data = torch.cat((img_data[0].unsqueeze(0), img_data), dim = 0)
                elif img_data.shape[0] > self.settings.num_slice:           # Removal of Emptier Peripheral Slices
                    img_data = img_data[int(np.ceil(-extra_slice / 2)) :\
                        int(len(img_data) - np.floor(-extra_slice / 2))]
            assert(img_data.shape[0] == self.settings.num_slice)
          
            # Item Dictionary Returning
            if save:
                print(f"Saving Patient Data for {subj_idx} into Video Format")
                if not os.path.isdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}"):
                    os.mkdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}")
                if not os.path.isdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}"):
                    os.mkdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}")
                torchvision.io.write_video(f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}/{subj_idx}.mp4",
                    (img_data.unsqueeze(3).repeat(1, 1, 1, 3) * 255).type(torch.uint8), fps = self.settings.num_fps)
        else: raise(NotImplementedError)
        #return img_data.unsqueeze(0)
        #return n4_bias_correction(img_data)
        #if self.dataset == "private" or self.dataset == "public":
        #    return ((img_data - 0.09635346) / 0.16435577).unsqueeze(0)
        return img_data.unsqueeze(0)
    