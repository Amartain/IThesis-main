import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from configurations.configure import start


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    start()
