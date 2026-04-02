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
# Author: Evan Greenberg, evan.greenberg@jpl.nasa.gov

import logging
import multiprocessing
import os
import time
from collections import OrderedDict
from functools import partial
from glob import glob

import click
import numpy as np
import torch
from spectf.model import SpecTfEncoder

# from spectf import SpecTfEncoder, import, spectf.model
from spectral.io import envi

import isofit.utils.template_construction as tmpl
from isofit import ray
from isofit.configs import configs
from isofit.core import units
from isofit.core.common import envi_header, load_spectrum, resample_spectrum
from isofit.core.fileio import IO, write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.data import env
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_simple import invert_algebraic
from isofit.utils.atm_interpolation import atm_interpolation
from isofit.utils.template_construction import Pathnames


class Irradiance:
    def __init__(self, path, doy):
        self._irr = np.loadtxt(irr_path)
        self._irr_factors = IO.load_esd()[int(doy) - 1, 1]

    def __call__(self, wl, fwhm):
        return np.array(
            resample_spectrum(self._irr[:, 1], self._irr[:, 0], wl, fwhm), dtype=float
        ) / (self._irr_factors**2)


class ModelInput:
    def __init__(self, model_path, bands):
        self.banddef = torch.tensor(bands, dtype=torch.float32)

        self.model_npz = np.load(model_path)
        self._arch = {}
        self._state_dict = {}
        num_state_keys = 0
        num_arch_keys = 0
        for key in self.model_npz.files:
            if "state_" in key:
                num_state_keys += 1
                self._state_dict[key.split("state_")[-1].replace("/", ".")] = (
                    torch.from_numpy(self.model_npz[key])
                )
            elif "arch_" in key:
                num_arch_keys += 1
                self._arch[key.split("arch_")[-1]] = self.model_npz[key]

        if not len(self._arch):
            raise ValueError("No architecture provided check .npz keys")

        if not len(self._state_dict):
            raise ValueError("No state_dict provided check .npz keys")

        if not (num_state_keys + num_arch_keys) == len(self.model_npz.files):
            print(
                "Warning: Number of keys unpacked does not match number of keys in file"
            )
            print(f"Num keys in file: {len(self.model_npz.files)}")
            print(f"Num state keys found: {num_state_keys}")
            print(f"Num arch keys found: {num_arch_keys}")

    @property
    def arch(self):
        types = {
            "dim_output": int,
            "num_heads": int,
            "dim_proj": int,
            "dim_ff": int,
            "dropout": float,
            "agg": str,
            "use_residual": bool,
            "num_layers": int,
        }
        for t, v in types.items():
            try:
                self._arch[t] = v(self._arch[t])
            except KeyError as e:
                raise KeyError(f"Missing key: '{t}' type in definitions")

        return self._arch

    @property
    def state_dict(self):
        return self._state_dict


def replace_bad_rowcol(row, col, ar, criteria):
    while np.any(criteria(ar[row, col])):
        idx = np.where(criteria(ar[row, col]))[0]
        row_nan = np.random.randint(0, ar.shape[0], len(idx))
        col_nan = np.random.randint(0, ar.shape[1], len(idx))
        row[idx] = row_nan
        col[idx] = col_nan

    return row, col


def sample_rowcol(nsamples, rdn_im, iter_limit=10):
    # Initial sample
    iters = 0
    row = np.random.randint(0, rdn_im.shape[0], nsamples)
    col = np.random.randint(0, rdn_im.shape[1], nsamples)
    rowcol = np.array([row, col]).T
    check = True

    # Don't sample bad pixels
    while check:
        n = len(rowcol) - len(np.unique(rowcol, axis=0))
        row_add = np.random.randint(0, rdn_im.shape[0], n)
        col_add = np.random.randint(0, rdn_im.shape[1], n)
        rowcol = np.vstack([np.unique(rowcol, axis=0), np.array([row_add, col_add]).T])
        row, col = rowcol[:, 0], rowcol[:, 1]

        # Don't use negatives (-9999), nans, infs
        row, col = replace_bad_rowcol(
            row, col, rdn_im[..., 10], lambda x: ((x < 0) | np.isnan(x)) | np.isinf(x)
        )
        if len(np.unique(rowcol, axis=0)) != len(rowcol):
            check = False

        rowcol = np.array([row, col]).T
        iters += 1

        if iters < iter_limit:
            break

    return row, col, iters


def presolve(
    rdn_path,
    obs_path,
    sensor,
    model_path,
    irr_path="",
    wavelength_path="",
    nsamples=300,
    quantiles=[0.001, 0.999],
    model_version="spectf",
):
    """\
    Perform a ML-style presolve calculation for a radiance cube

    \b
    Parameters
    ----------
    rdn_path: str
        Radiance data cube. Expected to be ENVI format
    obs_path: str
        Location data cube of shape (Lon, Lat, Elevation). Expected to be ENVI format
    sensor: str
        Instrument name string. This must match the code-specific convention.
    model_path: str (optional)
        Model .npz path. Must be serialized in a specific way. Serializer script provided in this document.
    wavelength_path: str (optional)
        Wavelength grid to resample to. Will default to the radiance cube.
    nsamples: int
        Number of samples to use in the sample-wise predictor.
    quantiles: list(float)
        Output quantiles
    model_version: str
        Potentially support multiple model versions
    """
    rdn = envi.open(envi_header(rdn_path))
    rdn_im = rdn.open_memmap(interleave="bip")
    obs = envi.open(envi_header(obs_path))
    coszen = np.cos(np.deg2rad(obs.open_memmap(interleave="bip")[..., 4]))

    if wavelength_path:
        # Assumes isofit-style wavelength file
        # Band 0: index
        # Band 1: Wavelength center (micron)
        # Band 2: fwhm (micron)
        bands = np.loadtxt(wavelength_path)
        # Heuristic check for wavelength unit
        if not bands[0, 1] // 10:
            bands[:, 1] = units.micron_to_nm(bands[:, 1])
            bands[:, 2] = units.micron_to_nm(bands[:, 2])

        wl = bands[:, 1]
        fwhm = bands[:, 2]

    else:
        wl = np.array(rdn.metadata.get("wavelength", [])).astype(float)
        fwhm = np.array(rdn.metadata.get("fwhm", [])).astype(float)

    if not len(wl) and not len(fwhm):
        raise ValueError("No wavelength provided in function arg or rdn metadata")

    # Sample and RDN to TOA RFL
    dayofyear = (
        tmpl.sensor_name_to_dt(sensor, Pathnames.parse_fid(rdn_path, sensor))[0]
        .timetuple()
        .tm_yday
    )

    if not irr_path:
        irr_path = [
            "examples",
            "20151026_SantaMonica",
            "data",
            "prism_optimized_irr.dat",
        ]
        if sensor == "oci":
            irr_path = ["data", "oci", "tsis_f0_0p1.txt"]

        irr_path = str(env.path(*irr_path))

    irr = Irradiance(irr_path, dayofyear)(wl, fwhm)

    row, col, iters = sample_rowcol(nsamples, rdn_im)
    rdn_sample = resample_spectrum(
        rdn_im[row, col, :].copy(),
        np.array(rdn.metadata["wavelength"]).astype(float),
        wl,
        fwhm,
    )
    sample_toa = units.rdn_to_transm(rdn_sample, coszen[row, col, None], irr).astype(
        np.float32
    )[None, ...]
    sample_toa = np.moveaxis(sample_toa, 0, -1)

    # Model prediction
    if model_version == "spectf":
        model_input = ModelInput(model_path, wl)
        model = SpecTfEncoder(model_input.banddef, **model_input.arch)
        model.load_state_dict(model_input.state_dict)
    else:
        raise ValueError(f"Model: {model_version} not implemented")

    pred = model(torch.from_numpy(sample_toa)).detach().numpy()

    return np.quantile(pred, quantiles[0]), np.quantile(pred, quantiles[1])


@click.command(name="Presolve", help=presolve.__doc__, no_args_is_help=True)
@click.argument("rdn_path")
@click.argument("obs_path")
@click.argument("sensor")
@click.option("--model_path", default="")
@click.option("--irr_path", default="")
@click.option("--wavelength_path", default="")
@click.option("--nsamples", "-n", default=300)
@click.option("--quantiles", "-q", type=float, multiple=True, default=[0.001, 0.999])
@click.option("--model_version", default="spectf")
def cli(**kwargs):
    """Perform a ML-style presolve calculation for a radiance cube"""
    presolve(**kwargs)
    click.echo("Done")
