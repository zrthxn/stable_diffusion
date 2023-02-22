from os import path, listdir
from torchvision.io import read_image, ImageReadMode
from logging import info

from . import ImageDataset
from src.config import defaults

class FacesDataset(ImageDataset):
    """
    Smiling or Not Face Data from Kaggle.
    """
    
    path = 'data/faces'

    def __init__(self, norm = True) -> None:
        info("Building Faces Dataset")
        paths = list()
        for subdir in ['smile', 'non_smile', 'test']:
            for f in listdir(path.join(self.path, subdir)):
                if f.endswith('jpg'): 
                    paths.append(path.join(self.path, subdir, f))

        for f in paths:
            try:
                im = read_image(f, mode=ImageReadMode.RGB).to(defaults.device)
                if norm:
                    im = ((im / 255.) * 2.) - 1.
                self.images.append(im)
            except Exception as e:
                print(Warning(e))
                continue

        info(f"Read {len(self.images)} face images")
