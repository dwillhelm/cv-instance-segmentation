
import logging 
from os import PathLike
import os
from pathlib import Path
from typing import Union
import requests 
import zipfile
from tqdm import tqdm

from src.utils._typing import FilePath

logger = logging.getLogger(__name__)

class IOMixIn: 

    def download_url(self, url:str, save_path:FilePath, chunk_size=128):
        """Download file from URL."""
        save_path = str(save_path)
        with requests.get(url, stream=True) as r:
            r.raise_for_status() 
            with open(save_path, 'wb') as fd:
                progress_bar =  tqdm(total=int(r.headers['Content-Length']))
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
                    progress_bar.update(len(chunk))
    
    def unzip_file(self, target:FilePath, dst:FilePath=None):
        target = Path(target)
        dst = target.with_suffix("") if dst is None else dst
        with zipfile.ZipFile(target, 'r') as fh: 
            fh.extractall(dst)

class PennFundanDataset(IOMixIn): 

    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    dataset_dir_name = "penn-fund-test"

    def __init__(self, location:FilePath):
        """Prepare the Penn Fundan Pedestrian dataset. 

        Parameters
        ----------
        location : FilePath
            The location (i.e., folder) to store the dataset. 
        """
        self.location = Path(location)
        self.root = self.location / self.dataset_dir_name

        if self.location.exists() is False: 
            logging.error("FileNotFound")
            msg = "The folder containing the dataset directory does not exist."
            raise FileNotFoundError(msg)

        if self.root.exists() is False:
            logger.warn("A local dataset was not found, downloading from URL.")
            zip_path = self.root.with_suffix(".zip")
            self.download_url(self.url, save_path=zip_path)
            self.unzip_file(target=zip_path, dst=self.root)

    def get_path(self): 
        return Path(self.root)
