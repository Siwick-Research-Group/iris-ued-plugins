# -*- coding: utf-8 -*-

from configparser import ConfigParser
from pathlib import Path
from operator import itemgetter
import csv

import numpy as np

from npstreams import average
from skued import diffread

from iris import AbstractRawDataset, check_raw_bounds


def csv_to_kvstore(fname):
    """ Parse CSV file into key-value store where keys are 
    always filepaths, and values are always floats """
    with open(fname, mode="r") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip one row of headers
        return {Path(row[0]): float(row[1]) for row in reader}


class McGillRawDatasetGamma(AbstractRawDataset):
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

    display_name = "McGill Raw Dataset v. Gamma"

    def __init__(self, source, *args, **kwargs):
        source = Path(source)
        if not source.is_dir():
            raise ValueError(f"{source} does not point to an existing directory")

        metadata_dict = self.parse_metadata(source / "metadata.cfg")
        super().__init__(source, metadata_dict)

        # Key-value stores where keys are always filepaths, and values are always floats
        self.timestamps = csv_to_kvstore(self.source / "timestamps.csv")
        self.ecounts = csv_to_kvstore(self.source / "ecounts.csv")
        self.room_temperature = csv_to_kvstore(self.source / "room_temp.csv")
        self.room_humidity = csv_to_kvstore(self.source / "room_humidity.csv")

    def nearest_laserbg(self, timestamp):
        """ Laser background taken nearest to ``timestamp`` """
        fnames = {
            abs(v) - timestamp: k
            for (k, v) in self.timestamps.items()
            if k.parent == Path("laser_background")
        }
        nearest_timestamp = min(fnames.keys())
        return diffread(self.source / fnames[nearest_timestamp])

    def nearest_pumpoff(self, timestamp):
        """ pumpoff image taken nearest to ``timestamp`` """
        fnames = {
            abs(v) - timestamp: k
            for (k, v) in self.timestamps.items()
            if k.parent == Path("pump_off")
        }
        nearest_timestamp = min(fnames.keys())
        return diffread(self.source / fnames[nearest_timestamp])

    def nearest_dark(self, timestamp):
        """ Dark image taken nearest to ``timestamp`` """
        fnames = {
            abs(v) - timestamp: k
            for (k, v) in self.timestamps.items()
            if k.parent == Path("dark_image")
        }
        nearest_timestamp = min(fnames.keys())
        return diffread(self.source / fnames[nearest_timestamp])

    def parse_metadata(self, fname):
        """ 
        Translate metadata from experiment into Iris's metadata format. 
        
        Parameters
        ----------
        fname : str or path-like
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
        fname = (
            Path(self.source) / f"scan_{scan:04d}" / f"pumpon_{timedelay:+010.3f}ps.tif"
        )

        if not fname.exists():
            raise IOError(
                f"Expected the file for {timedelay}ps and scan {scan} to exist, but could not find it."
            )

        im = diffread(fname)
        if bgr:
            laser_bg = self.nearest_laserbg(
                timestamp=self.timestamps[fname.relative_to(self.source)]
            )
            im -= laser_bg

        return im
