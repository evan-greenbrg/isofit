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
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import os
from typing import Dict, List, Type

import numpy as np

from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.statevector_config import StateVectorConfig


class SurfaceConfig(BaseConfigSection):
    """
    Surface configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._multi_surface_flag_type = bool
        self.multi_surface_flag = None

        self._sub_surface_class_file_type = str
        self.sub_surface_class_file = None

        self._surface_class_file_type = str
        self.surface_class_file = None

        self._Surfaces_type = dict
        self.Surfaces = None

        self._surface_params_type = dict
        self.surface_params = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_surface_categories = [
            "surface",
            "multicomponent_surface",
            "glint_surface",
            "thermal_surface",
            "lut_surface",
        ]
        if self.surface_category is None:
            errors.append("surface->surface_category must be specified")
        elif self.surface_category not in valid_surface_categories:
            errors.append(
                "surface->surface_category: {} not in valid surface categories: {}".format(
                    self.surface_category, valid_surface_categories
                )
            )

        if self.surface_category is None:
            errors.append("surface->surface_category must be specified")

        valid_normalize_categories = ["Euclidean", "RMS", "None"]
        if self.normalize not in valid_normalize_categories:
            errors.append(
                "surface->normalize: {} not in valid normalize choices: {}".format(
                    self.normalize, valid_normalize_categories
                )
            )

        return errors
