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
#
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from isofit.core import units
from isofit.core.common import eps, resample_spectrum, svd_inv_sqrt
from isofit.data import env
from isofit.surface.surface import DefaultState
from isofit.surface.surface_multicomp import MultiComponentSurface

DefaultAlphaPrior = DefaultState(
    bounds=[0.5, 1.5],
    scale=1.0,
    prior_mean=1,
    prior_sigma=0.1,
    init=1,
)

DefaultBetaPrior = DefaultState(
    bounds=[0.5, 1.5],
    scale=1.0,
    prior_mean=1,
    prior_sigma=0.1,
    init=1,
)


class AdjacentModelSurface(MultiComponentSurface):
    def __init__(self, full_config: Config):
        super().__init__(full_config)

        config = full_config.forward_model.surface

        self.statevec_names.extend(["ALPHA", "BETA"])
        self.adjacent_ind = len(self.statevec_names) - 2
        self.alpha_ind = self.adjacent_ind
        self.beta_ind = self.adjacent_ind + 1
        self.n_state = self.n_state + 2
        self.idx_surface = np.arange(len(self.statevec_names))

        self.analytical_iv_idx = np.arange(len(self.statevec_names))

        self.init.extend(
            [
                config.statevector.ALPHA.get("init"),
                config.statevector.BETA.get("init"),
            ]
        )

        self.scale.extend(
            [
                config.statevector.ALPHA.get("scale"),
                config.statevector.BETA.get("scale"),
            ]
        )

        self.bounds.extend(
            [
                config.statevector.ALPHA.get("bounds"),
                config.statevector.BETA.get("bounds"),
            ]
        )

        self.alpha_mean = config.statevector.ALPHA.get("prior_mean")
        self.alpha_sigma = (config.statevector.ALPHA.get("prior_sigma")) ** 2

        self.beta_mean = config.statevector.BETA.get("prior_mean")
        self.beta_sigma = (config.statevector.BETA.get("prior_sigma")) ** 2

        # Compute and and cache normalized Sa inversions for glint case
        Cov = np.array([[self.alpha_sigma, 0], [0, self.beta_sigma]])
        self.Sa_inv_glint, self.Sa_inv_sqrt_glint = svd_inv_sqrt(
            Cov / np.mean(np.diag(Cov))
        )
        # Check for fwhm, if not in surface -> use instrument
        if not len(self.fwhm):
            q = np.loadtxt(full_config.forward_model.instrument.wavelength_file)

            # Assume that fwhm is the second dim, and that wl file has multi-dim
            assert q.shape[1] > 1

            fwhm = q[:, 1]
            if q[0, 0] < 100:
                fwhm = units.micron_to_nm(fwhm)
            self.fwhm = fwhm

        self.drdn_drfl = self.drdn_drfl_heterogeneous_bgrfl
        self.evaluate_theta = self.evaluate_theta_heterogeneous_bgrfl
        self.surface.use_background_rfl = False

    def update_heuristic_prior_means(self, x_surface, geom):
        """Update the sun glint prior to match initial guess (x_surface"""
        mu = MultiComponentSurface.update_heuristic_prior_means(self, x_surface, geom)
        mu[self.alpha_ind] = self.alpha_mean
        mu[self.beta_ind] = self.beta_mean

        return mu

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""
        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.alpha_ind] = self.alpha_mean
        mu[self.beta_ind] = self.beta_mean

        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Sa_unnormalized, Sa_inv_normalized, Sa_inv_sqrt_normalized = (
            MultiComponentSurface.Sa(self, x_surface, geom)
        )
        Sa_unnormalized[self.alpha_ind, self.alpha_ind] = self.alpha_sigma
        Sa_unnormalized[self.beta_ind, self.beta_ind] = self.beta_sigma

        # Append normalized Sa inv and sqrt from glint model
        Sa_inv_normalized = block_diag(Sa_inv_normalized, self.Sa_inv_glint)
        Sa_inv_sqrt_normalized = block_diag(
            Sa_inv_sqrt_normalized, self.Sa_inv_sqrt_glint
        )

        return Sa_unnormalized, Sa_inv_normalized, Sa_inv_sqrt_normalized

    def drdn_dsurface(
        self,
        rho_dif_dir,
        drfl_dsurface,
        dLs_dsurface,
        s_alb,
        t_total_up,
        L_tot,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
    ):
        drdn_dsurface = np.zeros(drfl_dsurface.shape)
        drdn_drfl = self.drdn_drfl(
            L_tot,
            s_alb,
            rho_dif_dir,
            L_dir_dir=L_dir_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dir=L_dif_dir,
            L_dif_dif=L_dif_dif,
        )
        drdn_dsurface[:, : self.n_wl] = np.multiply(
            drdn_drfl[:, np.newaxis], drfl_dsurface[:, : self.n_wl]
        )

        dalpha, dbeta = self.drdn_dadjacency(
            L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif, s_alb, rho_dif_dir
        )

        drdn_dsurface[:, self.alpha_ind] = dalpha * drfl_dsurface[:, self.alpha_ind]
        drdn_dsurface[:, self.beta_ind] = dbeta * drfl_dsurface[:, self.beta_ind]

        # Get the derivative w.r.t. surface emission
        drdn_dLs = np.multiply(self.drdn_dLs(t_total_up)[:, np.newaxis], dLs_dsurface)

        return np.add(drdn_dsurface, drdn_dLs)

    def drdn_dadjacency(
        self, L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif, s_alb, rho_bg
    ):
        dalpha = L_dir_dif * rho_bg
        dbeta = (L_dif_dif * rho_bg) + (
            ((L_dir_dir + L_dif_dir + L_dif_dir + L_dif_dif) * s_alb * (rho_bg**2))
            / (1 - (s_alb * rho_bg))
        )

        return dalpha, dbeta

    def analytical_model(
        self,
        L_tot,
        geom,
        rho_dif_dif=None,
        s_alb=None,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
    ):
        """
        Linearization of the glint terms to use in AOE inner loop.
        Function will fetch the linearization of the rho terms and
        add the matrix components for the direct glint term.
        Currently we set the diffuse glint scaling term to constant
        value, which makes the AOE inner loop inversion possible.
        """
        # Construct the H matrix from:
        # theta (rho portion)
        # gam (sun glint portion)
        # ep (sky glint portion)
        H = super().analytical_model(
            L_tot,
            geom,
            s_alb,
            L_dir_dir,
            L_dir_dif,
            L_dif_dir,
            L_dif_dif,
        )

        # NOTE: The order of ep and gam respectively is important
        # It must match the alphabeitcal order of the staevector terms
        ep, gam = self.drdn_dadjacency(
            L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif, s_alb, rho_dif_dif
        )

        ep = np.reshape(ep, (len(ep), 1))
        H = np.append(H, ep, axis=1)

        gam = np.reshape(gam, (len(gam), 1))
        H = np.append(H, gam, axis=1)

        return H
