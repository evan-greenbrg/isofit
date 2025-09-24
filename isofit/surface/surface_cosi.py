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

from isofit.surface.surface_multicomp import MultiComponentSurface


class CosiSurface(MultiComponentSurface):
    """A model of the surface based on a collection of multivariate
    Gaussians, extended with surface topography term (cos_i)."""

    def __init__(self, full_config: Config):
        super().__init__(full_config)
        self.statevec_names.extend(["COSI"])
        self.scale.extend([1.0])

        # What to initialize with?
        self.init.extend([0.9])
        self.bounds.extend([[0.01, 1.0]])

        # Special glint bounds
        self.n_state = self.n_state + 1

        self.cosi_ind = len(self.statevec_names) - 1
        self.idx_surface = np.arange(len(self.statevec_names))

        self.analytical_interp_names = []
        self.analytical_iv_idx = np.arange(len(self.statevec_names))

        # self.f = np.array([[(0.02 * np.array(self.scale[self.cosi_ind :])) ** 2]])
        self.f = np.array([[(10 * np.array(self.scale[self.cosi_ind :])) ** 2]])

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.cosi_ind :] = self.init[self.cosi_ind :]
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        # Unclear if this should be a fully correlated block or a diagonal
        Cov[self.cosi_ind :, self.cosi_ind :] = np.eye(1) * self.f
        return Cov

    def fit_topography(self, x_surface, geom):
        """To accomodate cos_i within the statevector
        and to update it within RT, we update the value within
        the geom object during inversion"""
        geom.cos_i = x_surface[self.cosi_ind]

        return geom

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate and one or more emissive parameters,
        fit a state vector.
        """
        verified_geom = geom.verify(np.nan)
        coszen, cos_i = verified_geom["coszen"], verified_geom["cos_i"]

        x = MultiComponentSurface.fit_params(self, rfl_meas, geom)
        x[self.cosi_ind] = self.init[self.cosi_ind]
        self.init[self.cosi_ind] = cos_i

        return x

    def drdn_drfl(self, L_down_dir, L_down_dif, cos_i, s_alb, rho_dif_dir):
        """Partial derivative of radiance with respect to
        surface reflectance"""

        L1 = L_down_dir * cos_i
        return L1 + L_down_dif + (L1 + L_down_dif) / ((1.0 - s_alb * rho_dif_dir) ** 2)

    def drfl_dsurface(self, x_surface, geom, L_down_dir=None, L_down_dif=None):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def drdn_dcosi(self, L_down_dir, s_alb, rho_dir_dir, rho_dif_dir):
        """Derivative of radiance with respect to the cosi term"""

        # Retrieve local view from geom object
        # coszen=NaN will return geom-saved coszen
        drdn_dcosi = (L_down_dir * rho_dir_dir) + (
            (L_down_dir * s_alb * rho_dif_dir**2) / (1 - (s_alb * rho_dif_dir))
        )

        return drdn_dcosi

    def drdn_dsurface(
        self,
        rho_dir_dir,
        rho_dif_dir,
        drfl_dsurface,
        dLs_dsurface,
        s_alb,
        t_total_up,
        L_tot,
        L_down_dir,
        L_down_dif,
        geom,
    ):
        """Derivative of radiance with respect to
        full surface vector"""
        verified_geom = geom.verify(np.nan)
        coszen, cos_i = verified_geom["coszen"], verified_geom["cos_i"]

        # Element wise multiplication between
        # drdn_drfl (vector) and eye matrix to construct
        # drdn_drfl (diagonal)
        drdn_drfl = np.multiply(
            self.drdn_drfl(L_down_dir, L_down_dif, cos_i, s_alb, rho_dif_dir)[
                :, np.newaxis
            ],
            np.eye(len(self.wl), drfl_dsurface.shape[1]),
        )

        # Cosi derivatives
        drdn_dcosi = self.drdn_dcosi(L_down_dir, s_alb, rho_dir_dir, rho_dif_dir)
        drdn_drfl[:, -1] = drdn_dcosi

        # Chain rule to get derivative w.r.t. surface complete state
        # drdn_dsurface = np.multiply(drdn_drfl, drfl_dsurface)
        drdn_dsurface = drdn_drfl
        # Get the derivative w.r.t. surface emission
        drdn_dLs = np.multiply(self.drdn_dLs(t_total_up)[:, np.newaxis], dLs_dsurface)

        return np.add(drdn_dsurface, drdn_dLs)

    def analytical_model(
        self,
        bg_rho,
        s,
        L_down_dir,
        L_down_dif,
        L_tot,
        geom,
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
        verified_geom = geom.verify(np.nan)
        coszen, bg_cos_i = verified_geom["coszen"], verified_geom["cos_i"]
        print(bg_cos_i)

        background = bg_rho * s

        # Construct the H matrix - different from multicomponent
        # rho = L_tot + (L_tot * background / (1 - background))
        rho = L_down_dir * bg_cos_i + L_down_dif
        H = np.eye(self.n_wl, self.n_wl)
        H = rho[:, np.newaxis] * H

        # ep = (L_down_dir / bg_cos_i * bg_rho) / (1 - background)
        ep = L_down_dir * bg_rho
        ep = np.reshape(ep, (len(ep), 1))
        H = np.append(H, ep, axis=1)

        # Constant from Taylor expansion around background_rfl, bg_cos_i
        # O = (
        #     -1
        #     * bg_rho
        #     * (((L_down_dir * bg_cos_i) + (L_down_dif * background)))
        #     / (1 - background) ** 2
        # )
        O = -1 * L_down_dir * bg_rho * bg_cos_i

        return H, O

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return MultiComponentSurface.summarize(
            self, x_surface, geom
        ) + " Sun Glint: %5.3f, Sky Glint: %5.3f" % (x_surface[-2], x_surface[-1])
