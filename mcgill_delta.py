# -*- coding: utf-8 -*-

import csv
from configparser import ConfigParser
from operator import itemgetter
from functools import wraps
import os
import numpy as np
from npstreams import average
from iris import AbstractRawDataset, check_raw_bounds
from skued import diffread

    
def csv_to_kvstore(sname, fname):
    """ Parse CSV file into key-value store where keys are 
    always fileos.abspaths, and values are always floats """
    ffname = os.path.join(sname , fname)
    with open(ffname, mode="r") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip one row of headers
        dictdata={}
        for row in reader:
            key = os.path.join(sname, row[0]).replace(os.sep, '/')
            dictdata[key] = float(row[1])
        return dictdata#{os.path.abspath(row[0]): float(row[1]) for row in reader}


def asfarray(f):
    """ Cast the result array from a function as a floating-point array """

    @wraps(f)
    def newf(*args, **kwargs):
        return np.asfarray(f(*args, **kwargs))

    return newf

def newgetitem(self, key):
    key = str(key)
    return dict.__getitem__(self, key)
    
class McGillRawDatasetDelta(AbstractRawDataset):
    """
    Raw dataset from the Siwick Research Group Diffractometer, in use 
    starting late 2019.

    Parameters
    ----------
    source : str
        Raw data directory
    
    Raises
    ------
    ValueError : if the source directory does not exist.
    """

    display_name = "McGill Raw Dataset v. Delta"

    def __init__(self, source, *args, **kwargs):
        source = os.path.abspath(source)
        if not os.path.isdir(source):
            raise ValueError(f"{source} does not point to an existing directory")

        metadata_dict = self.parse_metadata(os.path.join(source , "metadata.cfg"))
        super().__init__(source, metadata_dict)

        # Key-value stores where keys are always fileos.abspaths, and values are always floats
        self.timestamps = csv_to_kvstore(self.source , "timestamps.csv")
        #print(f"Source is {self.source}")
        #self.timestamps.__getitem__ = newgetitem
        self.ecounts = csv_to_kvstore(self.source , "ecounts.csv")
        self.room_temperature = csv_to_kvstore(self.source , "room_temp.csv")
        self.room_humidity = csv_to_kvstore(self.source , "room_humidity.csv")

    def nearest_laserbg(self, timestamp):
        """ Laser background taken nearest to ``timestamp`` """
        fnames = {
            abs(v) - timestamp: k
            for (k, v) in self.timestamps.items() if "laser_background" in k
            #if k.parent == os.path.abspath("laser_background")
        }
        nearest_timestamp = min(fnames.keys())
        return diffread(os.path.join(self.source , fnames[nearest_timestamp]))

    def nearest_pumpoff(self, timestamp):
        """ pumpoff image taken nearest to ``timestamp`` """
        fnames = {
            abs(v) - timestamp: k
            for (k, v) in self.timestamps.items() if "pump_off" in k
            #if k.parent == os.path.abspath("pump_off")
        }
        nearest_timestamp = min(fnames.keys())
        return diffread(os.path.join(self.source , fnames[nearest_timestamp]))

    def nearest_dark(self, timestamp):
        """ Dark image taken nearest to ``timestamp`` """
        fnames = {
            abs(v) - timestamp: k
            for (k, v) in self.timestamps.items() if "dark_image" in k
            #if k.parent == os.path.abspath("dark_image")
        }
        nearest_timestamp = min(fnames.keys())
        return diffread(os.path.join(self.source , fnames[nearest_timestamp]))

    def parse_metadata(self, fname):
        """ 
        Translate metadata from experiment into Iris's metadata format. 
        
        Parameters
        ----------
        fname : str or os.path.abspath-like
            Filename to the config file.
        """
        metadata = dict()

        parser = ConfigParser(inline_comment_prefixes=("#"))
        parser.read(fname)
        exp_params = parser["EXPERIMENTAL PARAMETERS"]

        # Translation is required between metadata formats
        metadata["energy"] = exp_params["electron energy"]
        metadata["acquisition_date"] = exp_params["acquisition date"]
        metadata["fluence"] = exp_params["fluence"]
        metadata["temperature"] = exp_params["temperature"]
        metadata["exposure"] = exp_params["exposure"]
        metadata["notes"] = exp_params["notes"]
        metadata["pump_wavelength"] = exp_params["pump wavelength"]

        metadata["scans"] = list(range(1, int(exp_params["nscans"]) + 1))
        metadata["time_points"] = eval(exp_params["time points"])

        return metadata

    @check_raw_bounds
    def electron_count(self, timedelay, scan):
        """
        Return the electron count for the picture acquired at 
        time-delay ``timedelay`` ans scan ``scan``

        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1
        
        Returns
        -------
        e : float
            Electron number.

        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        fname = (
            os.path.join(os.path.abspath(self.source) , f"scan_{scan:04d}" , f"pumpon_{timedelay:+010.3f}ps.tif")
        )
        if not fname.exists():
            raise IOError(
                f"Expected the file for {timedelay}ps and scan {scan} to exist, but could not find it."
            )
        return self.ecounts[img_fname.relative_to(self.source)]

    @asfarray
    @check_raw_bounds
    def raw_data(self, timedelay, scan=1, bgr=True, **kwargs):
        """
        Returns an array of the image at a timedelay and scan. Laser background is
        removed by default.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        bgr : bool, optional
            If True (default), the laser background is removed as well.
        kwargs
            Extra keyword arguments are ignored.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        fname = os.path.join(os.path.abspath(self.source) , f"scan_{scan:04d}" , f"pumpon_{timedelay:+010.3f}ps.tif")
        

        if not os.path.exists(fname):
            raise IOError(
                f"Expected the file for {timedelay}ps and scan {scan} to exist, but could not find it."
            )

        im = diffread(fname)
        if bgr:
            laser_bg = self.nearest_laserbg(
                timestamp=self.timestamps[fname]#fname.relative_to(self.source)
            )
            im -= laser_bg

        return im


class McGillRawDatasetDeltaPumpoff(McGillRawDatasetDelta):
    """
    Diagnostic raw dataset from the Siwick Research Group Diffractometer, in use 
    starting late 2019. This dataset will reduce only pumpoff pictures

    Parameters
    ----------
    source : str
        Raw data directory
    
    Raises
    ------
    ValueError : if the source directory does not exist.
    """

    display_name = "McGill Raw Dataset v. Delta [Diagnostic pump-off]"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scans = [1]

        # Determine time-stamps from filenames
        fnames = {
            k for (k, v) in self.timestamps.items() if k.parent == os.path.abspath("pump_off")
        }
        self.time_points = np.asfarray(
            sorted(self.timestamps[fname] for fname in fnames)
        )

    @asfarray
    @check_raw_bounds
    def raw_data(self, timedelay, scan=1, bgr=True, **kwargs):
        """
        Returns an array of the image at a timedelay and scan. Laser background is
        removed by default.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        bgr : bool, optional
            If True (default), the laser background is removed as well.
        kwargs
            Extra keyword arguments are ignored.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        # In this case, the time-delay is the pump-off timestamp
        fname = os.path.join(self.source , "pump_off" , f"pump_off_epoch_{timedelay:010.0f}s.tif")

        if not fname.exists():
            raise IOError(
                f"Expected the file for {timedelay}ps and scan {scan} to exist, but could not find it."
            )

        im = diffread(fname)
        if bgr:
            dark_bg = self.nearest_dark(
                timestamp=self.timestamps[fname.relative_to(self.source)]
            )
            im -= dark_bg

        return im
