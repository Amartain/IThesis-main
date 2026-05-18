from configurations.selection import DatasetSelection


originals = {
    DatasetSelection.ANIMAL2000 : "datasets/Animal2000/Animal2000-Original/Animal2000-Original",
    DatasetSelection.KIMIA99 : "datasets/Kimia99/Kimia99-Original/Kimia99-Original",
    DatasetSelection.KIMIA216 : "datasets/Kimia216/Kimia216-Original/Kimia216-Original",
    DatasetSelection.MPEG7: "datasets/MPEG7/MPEG7-Original/MPEG7-Original",
    DatasetSelection.MPEG400: "datasets/MPEG400/MPEG400-Original"

}
skeletons = {
    DatasetSelection.ANIMAL2000 : "datasets/Animal2000/Animal2000-GT/Animal2000-GT",
    DatasetSelection.KIMIA99 : "datasets/Kimia99/Kimia99-GT/Kimia99-GT",
    DatasetSelection.KIMIA216 : "datasets/Kimia216/Kimia216-GT/Kimia216-GT",
    DatasetSelection.MPEG7: "datasets/MPEG7/MPEG7-GT/MPEG7-GT",
    DatasetSelection.MPEG400: "datasets/MPEG400/MPEG400-GT"
    


}
thumbs = {
    DatasetSelection.ANIMAL2000 : "datasets/Animal2000/Animal2000-Thumb/Animal2000-Thumb",
    DatasetSelection.KIMIA99 : "datasets/Kimia99/Kimia99-Thumb/Kimia99-Thumb",
    DatasetSelection.KIMIA216 : "datasets/Kimia216/Kimia216-Thumb/Kimia216-Thumb",
    DatasetSelection.MPEG7: "datasets/MPEG7/MPEG7-Thumb/MPEG7-Thumb",
    DatasetSelection.MPEG400: "datasets/MPEG400/MPEG400-Thumb"

}


def get_dataset_dirs(dataset_choice):
    
    return originals[dataset_choice], skeletons[dataset_choice], thumbs[dataset_choice]