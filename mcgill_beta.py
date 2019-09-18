# -*- coding: utf-8 -*-
from configparser import ConfigParser
from glob import iglob
from os.path import isdir, join
from pathlib import Path

import numpy as np

from skued import diffread

from iris import AbstractRawDataset, check_raw_bounds


class McGillRawDatasetBeta(AbstractRawDataset):
    """
    Raw dataset from the Siwick Research Group Diffractometer, in use 
    from 2017 to late 2019.

    Parameters
    ----------
    source : str
        Raw data directory
    
    Raises
    ------
    ValueError : if the source directory does not exist.
    """

    display_name = "McGill Raw Dataset v. Beta"

    def __init__(self, source, *args, **kwargs):
        if not isdir(source):
            raise ValueError(f"{source} does not point to an existing directory")

        metadata_dict = self.parse_metadata(join(source, "metadata.cfg"))
        super().__init__(source, metadata_dict)

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
    def raw_data(self, timedelay, scan=1, **kwargs):
        """
        Returns an array of the image at a timedelay and scan. Dark background is
        always removed.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
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
        # scan directory looks like 'scan 0132'
        # Note that a glob pattern is required because every diffraction pattern
        # has a timestamp in the filename.
        directory = join(self.source, f"scan {scan:04d}")
        try:
            fname = next(iglob(join(directory, f"pumpon_{timedelay:+010.3f}ps_*.tif")))
        except StopIteration:
            raise IOError(
                f"Expected the file for {timedelay}ps and scan {scan} to exist, but could not find it."
            )

        return diffread(fname)
