"""
Contains the `Dataset` class `SkeView` designed for the SkeView dataset and skeletonization task.

Defines the `SkeView` class, which is responsible for loading the original images, ground truth labels, thumbs and filtering and preprocessing of these.
"""

from torch.utils.data import Dataset
import os
from PIL import Image
from data.utils import clean_labels


class SkeView(Dataset):
    """
    Pytorch Dataset class.

    The class reads in the images from the given directories, it filters them by the given size and transforms them for training, validation and or testing.

    **Attributes**:
        `original_dir (`str`): The path to directory with the original images (.jpg).
        `gt_dir (`str`): The path to directory with the ground truth images (.png).
        `thumbs_dir (`str`): The path to directory with the thumb images (.png).
        `transform (`callable`, optional): Optional transformation(s) for the images.
        `size_filter (`int`): Maximum size of the images on any dimension.
        `all_labels (`list`): The list of all the filenames in the `original_dir` directory.
        `labels` (`list`): The list of the names of the files that `size_filter` did not filter out. Thus the list of filenames which the corresponding image is under the size_filter in both height and width.
    """
   
    def __init__(self, original_dir, gt_dir, thumbs_dir, size_filter, transform=None):
        """
        Initializes SkeView dataset/

        **Args**:
            `original_dir` (`str`): Az eredeti képek könyvtárának útvonala.
            `gt_dir` (`str`): A valós maszkok könyvtárának útvonala.
            `thumbs_dir` (`str`): A bélyegképek könyvtárának útvonala.
            `size_filter` (`int`): Size filter in pixels.
                **Warning**: The dataset will filter out images that are on any dimension larger then this number!
            `transform` (`callable`, optional): Optional transformation(s) for the images. Defaults to None.
        """
        self.original_dir = original_dir
        self.gt_dir = gt_dir
        self.thumbs_dir = thumbs_dir
        self.transform = transform
        self.size_filter = size_filter

        self.all_labels = clean_labels(os.listdir(original_dir))
        self.labels = self.filter_by_size()

        print(f"{len(self.labels)}/{len(self.all_labels)} kept. Filter size: {size_filter}")


    def __len__(self):
        """
        Gives back the length of the filtered dataset.

        **Returns**:
            `int`: The remaining number of items.
        """
        return len(self.labels)
    
    def __getitem__(self, idx): 
        """
        Gets the items for the given index. 

        **Args**:
            `idx` (`int`): The index of the item quieried.

        **Returns**:
            `tuple`: 4 item `tuple` consisting of:
                - `original` (`PIL.Image vagy torch.Tensor`): Original image.
                - `gt` (`PIL.Image vagy torch.Tensor`): Ground truth (skeleton).
                - `thumb` (`PIL.Image vagy torch.Tensor`): Thumbs (Ground truth overlayed the original image).
                - `label` (`str`): Filename / label of the image.
        """
        label = self.labels[idx]

        original, gt, thumb, label = self.retrieve_image(label)

        if self.transform is not None:
            
            original = self.transform(original)
            gt = self.transform(gt)
            thumb = self.transform(thumb)


        return original, gt, thumb, label



    def retrieve_image(self, label): # use label cause we have diff. names for each image!
        """
        Loads images from the filesystem and returns them.

        **Args**:
            `label` (`str`): The filename of the image to be loaded without format.

        `Returns`:
            `tuple`: contains 4 items:
                - `PIL.Image` : original image
                - `PIL.Image` : ground truth (skeleton) image
                - `PIL.Image` : thumb image
                - `str`: labels
        """
        jpg_filename = label + ".jpg"
        png_filename = label + ".png"
        original_path = os.path.join(self.original_dir, jpg_filename)
        gt_path = os.path.join(self.gt_dir, png_filename)
        thumb_path = os.path.join(self.thumbs_dir, png_filename)

        # set mode to binary!!!
        original = Image.open(original_path).convert(mode="1")
        gt = Image.open(gt_path).convert(mode="1")
        thumb = Image.open(thumb_path).convert(mode="1")

        return original, gt, thumb, label
    
    
    def filter_by_size(self):
        """
        Filters the images that are larger than the given size constraint.

        Iterates over all available labels, opens the original image that belongs to it, and keeps the labels that are under or equal to the size constraint `size_filter` on both dimensions (width, height).

        **Returns**:
            `list`: List of the image labels which correspond to images under or equal to the size constraint `size_filter`.
        """
        labels = []
        for label in self.all_labels:
            filepath = os.path.join(self.original_dir, f'{label}.jpg')

            with Image.open(filepath) as img:
                width, height = img.size

                if width <= self.size_filter and height <= self.size_filter:
                    labels.append(label)

        return labels

    def get_labels(self):
        """
        Returns the valid labels (labels valid after filtering by size).

        **Returns**:
            `list`: List of valid labels.
        """

        return self.labels