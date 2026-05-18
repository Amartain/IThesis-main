"""
This module defines the dataset directory structures and paths.

The module contains dictionaries which map out the path of directionaries containing the datasets to the `DatasetSelection` Enums.

"""

from configurations.selection import DatasetSelection

# maps out the paths to directories containing the original images
originals = {
    DatasetSelection.ANIMAL2000 : "datasets/Animal2000/Animal2000-Original/Animal2000-Original",
    DatasetSelection.KIMIA99 : "datasets/Kimia99/Kimia99-Original/Kimia99-Original",
    DatasetSelection.KIMIA216 : "datasets/Kimia216/Kimia216-Original/Kimia216-Original",
    DatasetSelection.MPEG7: "datasets/MPEG7/MPEG7-Original/MPEG7-Original",
    DatasetSelection.MPEG400: "datasets/MPEG400/MPEG400-Original"

}
# maps out the path to the directories containing the skeleton ground truth images
skeletons = {
    DatasetSelection.ANIMAL2000 : "datasets/Animal2000/Animal2000-GT/Animal2000-GT",
    DatasetSelection.KIMIA99 : "datasets/Kimia99/Kimia99-GT/Kimia99-GT",
    DatasetSelection.KIMIA216 : "datasets/Kimia216/Kimia216-GT/Kimia216-GT",
    DatasetSelection.MPEG7: "datasets/MPEG7/MPEG7-GT/MPEG7-GT",
    DatasetSelection.MPEG400: "datasets/MPEG400/MPEG400-GT"
    


}
# maps out the path to the directories containing the "thumb" images (ground truth overlayed on the original image)
thumbs = {
    DatasetSelection.ANIMAL2000 : "datasets/Animal2000/Animal2000-Thumb/Animal2000-Thumb",
    DatasetSelection.KIMIA99 : "datasets/Kimia99/Kimia99-Thumb/Kimia99-Thumb",
    DatasetSelection.KIMIA216 : "datasets/Kimia216/Kimia216-Thumb/Kimia216-Thumb",
    DatasetSelection.MPEG7: "datasets/MPEG7/MPEG7-Thumb/MPEG7-Thumb",
    DatasetSelection.MPEG400: "datasets/MPEG400/MPEG400-Thumb"

}


def get_dataset_dirs(dataset_choice):
    """
    Gets the paths to the dataset of choice. 

    The function uses the dictionaries defined in the module to give pack the needed file paths.

    **Args**: 
        `dataset_choice` (`DatasetSelection`): Enumeration value that serves to represent the dataset choice.

    **Returns**: 3 item `tuple` (`str`, `str`, `str`)
        - (`str`) The original images' directory path.
        - (`str`) The skeleton ground truth images' directory path.
        - (`str`) The thumb images' directory path.
    """

    return originals[dataset_choice], skeletons[dataset_choice], thumbs[dataset_choice]