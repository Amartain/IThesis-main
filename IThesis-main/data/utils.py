"""Segédfüggvények modulja az adathalmazok címkéinek kezeléséhez.

Utilitu module.

Contains functions to:
    - data preparation, like converting raw filenames into labels.
"""
def clean_labels(jpg_filenames):
    """Megtisztítja a fájlnevek listáját a ".jpg" kiterjesztés eltávolításával.
    Cleans the list of filenames by removing ".jpg" format suffix.

    **Args**:
        `jpg_filenames` (`list`): List of filenames with ".jpg" format suffix.

    **Returns**:
        `list`: The clean list of labels without the format suffix.
    """
    labels = [filename.replace(".jpg", "") for filename in jpg_filenames]

    return labels


