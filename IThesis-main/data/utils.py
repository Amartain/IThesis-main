def clean_labels(jpg_filenames):
    labels = [filename.replace(".jpg", "") for filename in jpg_filenames]

    return labels


