# Stuff for ray parallelism. Thanks to neggles.
import pathlib
from typing import Any
from .taggers_core import OnnxLoader
import pandas
from .distortion import folder_images, image_suffixes
from ray.data.datasource.image_datasource import _ImageDatasourceReader, _ImageFileMetadataProvider

from ray.data.datasource import (
    BaseFileMetadataProvider,
    BinaryDatasource,
    DefaultFileMetadataProvider,
    FastFileMetadataProvider,
    FileExtensionFilter,
    Partitioning,
    PathPartitionFilter,
    Reader,
)

try:
    import ray
except ImportError:
    print("Ray not present. Unable to become gay.")



class RayOnnxClassifier:


    def __init__(self,
                 model:OnnxLoader,
                 thresholds:dict={}
                 ):
        self.model = model

    def __call__(self, batch: pandas.DataFrame) -> pandas.DataFrame:
        self.model.predict

class RayDataset():
    

    def get_files(self,path:pathlib.Path, recurse=False):
        files = []
        if path.is_dir():
            files = folder_images(path, recurse=recurse)
        if path.is_file():
            if path.suffix.lower() in image_suffixes:
                files = [path]
            else:
                raise Exception(f"path: {path} suffix is not {image_suffixes}.")
        else:

        if not path.is_dir() and not path.is_file():
            raise Exception(f"path: {path} is not a file or directory")
        return files

    def __init__(self, path:pathlib.Path, recursive:bool=False) -> None:
        self.generator = file_generator

# Class Executor