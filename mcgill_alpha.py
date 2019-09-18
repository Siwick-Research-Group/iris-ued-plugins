# -*- coding: utf-8 -*-
from contextlib import suppress
from functools import lru_cache
from glob import iglob
from os import listdir
from os.path import isdir, isfile, join
from re import search, sub

import numpy as np

from npstreams import average
from skued import diffread

from iris import AbstractRawDataset, check_raw_bounds


class McGillRawDatasetAlpha(AbstractRawDataset):
    """
    Raw dataset from the Siwick Research Group Diffractometer, in use 
    from ~2008 to 2017.

    Parameters
    ----------
    source : str
        Raw data directory
    
    Raises
    ------
    ValueError : if the source directory does not exist.
    """

    display_name = "McGill Raw Dataset v. Alpha"

    def __init__(self, source, *args, **kwargs):
        if not isdir(source):
            raise ValueError(f"{source} does not point to an existing directory")
        super().__init__(source)

        # Populate experimental parameters
        # from a metadata file called 'tagfile.txt'
        _metadata = self.parse_tagfile(join(self.source, "tagfile.txt"))
        self.fluence = _metadata.get("fluence") or 0
        self.resolution = (2048, 2048)
        self.current = _metadata.get("current") or 0
        self.exposure = _metadata.get("exposure") or 0
        self.energy = _metadata.get("energy") or 90

        # Determine acquisition date
        # If directory name doesn't match the time pattern, the
        # acquisition date will be the default value
        with suppress(AttributeError):
            self.acquisition_date = search(r"(\d+[.])+", str(self.source)).group()[
                :-1
            ]  # Last [:-1] removes a '.' at the end

        # To determine the scans and time-points, we need a list of all files
        image_list = [
            f
            for f in listdir(self.source)
            if isfile(join(self.source, f)) and f.endswith((".tif", ".tiff"))
        ]

        # Determine the number of scans
        # by listing all possible files
        scans = [
            search("[n][s][c][a][n][.](\d+)", f).group()
            for f in image_list
            if "nscan" in f
        ]
        self.scans = tuple({int(string.strip("nscan.")) for string in scans})

        # Determine the time-points by listing all possible files
        time_data = [
            search("[+-]\d+[.]\d+", f).group() for f in image_list if "timedelay" in f
        ]
        time_list = list(
            set(time_data)
        )  # Conversion to set then back to list to remove repeated values
        time_list.sort(key=float)
        self.time_points = tuple(map(float, time_list))

    @staticmethod
    def parse_tagfile(path):
        """ Parse a tagfile.txt from a raw dataset into a dictionary of values """
        metadata = dict()
        with open(path) as f:
            for line in f:
                key, value = sub("\s+", "", line).split(
                    "="
                )  # \s+ means all white space , including 'unicode' white space
                try:
                    value = float(
                        value.strip("s")
                    )  # exposure values have units of seconds
                except ValueError:
                    value = None  # value might be 'BLANK'
                metadata[key.lower()] = value
        return metadata

    @property
    @lru_cache(maxsize=1)
    def background(self):
        """ Laser background """
        backgrounds = map(diffread, iglob(join(self.source, "background.*.pumpon.tif")))
        return average(backgrounds)

    @check_raw_bounds
    def raw_data(self, timedelay, scan=1, bgr=True, **kwargs):
        """
        Returns an array of the image at a timedelay and scan.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        bgr : bool, optional
            If True (default), laser background is removed before being returned.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        # Template filename looks like:
        #    'data.timedelay.+1.00.nscan.04.pumpon.tif'
        sign = "" if timedelay < 0 else "+"
        str_time = sign + f"{timedelay:.2f}"
        filename = (
            "data.timedelay."
            + str_time
            + ".nscan."
            + str(scan).zfill(2)
            + ".pumpon.tif"
        )

        im = diffread(join(self.source, filename)).astype(np.float)
        if bgr:
            im -= self.background
            im[im < 0] = 0

        return im
