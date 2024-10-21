#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#
from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat

from isofit.configs import Config
from isofit.core.common import envi_header, load_spectrum, load_wavelen


class Surface:
    """A wrapper for the specific surface models"""

    def __init__(self, full_config: Config):
        config = full_config.forward_model.surface
        self.model_dict = loadmat(config.surface_file)

        config = full_config.forward_model.surface
        self.surfaces = config
        for i, surf_dict in config.items():
            self.surfaces[i]["surface_model"] = Surfaces[surf_dict["surface_category"]]
        # surfaces = config
        for i, surf_dict in config.items():
            config[i]["surface_model"] = Surfaces[surf_dict["surface_category"]]

        # These are overwritten by specific surface model
        self.wl = None
        self.fwhm = None
        self.n_wl = None

        if config.wavelength_file is not None:
            self.wl, self.fwhm = load_wavelen(config.wavelength_file)

        elif "wl" in self.model_dict:
            self.wl = self.model_dict["wl"][0]

        elif full_config.implementation.mode == "simulation":
            logging.info(
                "No surface wavelength_file provided, getting wavelengths from"
                " input.reflectance_file"
            )
            _, self.wl = load_spectrum(full_config.input.reflectance_file)

        if self.wl is not None:
            self.n_wl = len(self.wl)

    def match_class(self, row, col):
        matches = np.zeros((len(self.groups))).astype(int)
        for i, group in enumerate(self.groups):
            if [row, col, 0] in group:
                matches[i] = 1
            else:
                matches[i] = 0

        if len(matches[np.where(matches)]) > 1:
            raise ValueError(
                "Pixel did not match any class. \
                             Something is wrong"
            )

        elif len(matches[np.where(matches)]) > 1:
            raise ValueError(
                "Pixel matches too many classes. \
                             Something is wrong"
            )

        return matches[np.where(matches)][0]

    def call_rowcol_surface(self, row, col):
        # Easy case, no classification is propogated through
        if len(self.surfaces) == 1 or not self.surfaces[0]["surface_class_file"]:
            return self.surfaces[0]["surface_model"](self.full_config)

        elif len(self.surfaces) > 1:
            return self.surfaces[self.match_class(groups, row, col)]["surface_model"](
                self.full_config
            )
