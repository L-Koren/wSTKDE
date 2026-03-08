# MIT License

# Copyright (c) 2026 L. Koren

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



# Local imports
from src.static_kdtree import StaticKDTree
from src.bandwidth_selection import BandwidthSelector
from src.stkde import STKDE

# Python imports
import unittest

# Third-party imports
import geopandas as gpd
from shapely import Point
from numba import njit
import numpy as np
from numpy import testing as npt



## Test classes
class TestStaticKDTree(unittest.TestCase):
    
    def setUp(self):
        self.min_value = 0.4
        self.max_value = 0.6
        np.random.seed(42)
        self.weighted_points   = np.random.rand(1000, 4)
        self.unweighted_points = np.random.rand(1000, 3)
        
    def _query_numpy(self, points):
        """Helper funcion to get expected results via numpy filtering."""
        lo, hi = self.min_value, self.max_value
        mask = (
            (points[:, 0] > lo) & (points[:, 0] < hi) &
            (points[:, 1] > lo) & (points[:, 1] < hi) &
            (points[:, 2] > lo) & (points[:, 2] < hi)
        )
        return points[mask]
    
    def _assert_kdtree_matches_numpy(self, points, weighted):
        """Helper function to compare KDTree query and NumPy filter."""
        lo, hi = self.min_value, self.max_value
        np_result = self._query_numpy(points)
        kdtree = StaticKDTree(points=points, weighted=weighted)
        kdtree_result = kdtree.query_aabb_3d(lo, hi, lo, hi, lo, hi)

        npt.assert_array_equal(
            np_result[np.lexsort(np_result.T)],
            kdtree_result[np.lexsort(kdtree_result.T)],
            err_msg=f"KDTree result does not match numpy for weighted={weighted}"
        )

    def test_unweighted_points(self):
        """Test if 3D AABB query for the StaticKDTree class returns the same points as a full scan."""
        self._assert_kdtree_matches_numpy(self.unweighted_points, weighted=False)
        
    def test_weighted_points(self):
        """Test if 3D AABB query for the StaticKDTree class returns the same points as a full scan."""
        self._assert_kdtree_matches_numpy(self.weighted_points, weighted=True)
        
class TestBandwidthSelector(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.points = np.random.rand(1000, 3)
        self.weighted_points = np.random.rand(1000, 4)
        
    def _estimate(self, method, weighted):
        """Helper function to estimate bandwidths."""
        bws = BandwidthSelector(method)
        if weighted:
            result = bws.estimate(self.weighted_points)
        else:
            result = bws.estimate(self.points)
        return result

    def _assert_bandwidths(self, result, expected, rtol=1e-7):
        """Helper function to assert bandwidth computation matches expected result."""
        npt.assert_allclose(
            actual=result, 
            desired=expected,
            rtol=rtol
        )

    def test_scott_value(self):
        """Test value returned for Scott bandwidth method."""
        result = self._estimate("scott", False)
        expected = (
            np.float64(0.1069209574939819),
            np.float64(0.1104403706632645),
            np.float64(0.10873166419476847)
        )
        self._assert_bandwidths(result, expected)
    
    def test_scott_weighted(self):
        """Test value returned for weighted Scott bandwidth method."""
        result = self._estimate("scott", True)
        expected = (
            np.float64(0.10882504935214922),
            np.float64(0.11425805977070032),
            np.float64(0.11270830510160088)
        )
        self._assert_bandwidths(result, expected)

    def test_silverman_value(self):
        """Test value returned for Silverman bandwidth method."""
        result = self._estimate("silverman", False)
        expected = (
            np.float64(0.10356632164000504),
            np.float64(0.10697531352351401),
            np.float64(0.10532021757364304)
        )
        self._assert_bandwidths(result, expected)
    
    def test_silverman_weighted(self):
        """Test value returned for weighted Silverman bandwidth method."""
        result = self._estimate("silverman", True)
        expected = (
            np.float64(0.10541067277973526),
            np.float64(0.1106732229632466),
            np.float64(0.10917209171372438)
        )
        self._assert_bandwidths(result, expected)
    
    def test_cross_validation(self):
        """Test if no errors are raised for cross-validation bandwidth method.
        NOTE: We cannot test values as this method is non-deterministic."""
        result = self._estimate("cross-validation", False)
    
    def test_cross_validation_weighted(self):
        """Test if no errors are raised for cross-validation bandwidth method.
        NOTE: We cannot test values as this method is non-deterministic."""
        result = self._estimate("cross-validation", True)
        
    def test_log_likelihood_value(self, bw=0.25):
        """Test log likelihood calculation method."""
        bws = BandwidthSelector("cross-validation")
        kdtree = bws._create_kdtree(self.points, False)
        ll = bws._log_likelihood(self.points, kdtree.nodes, kdtree.leaves, kdtree.leaf_size, bw, bw, bw)
        npt.assert_almost_equal(ll, -379.29487544616006)
    
    def test_log_likelihood_weights(self, bw=0.25):
        """Test weighted log likelihood calculation method."""
        bws = BandwidthSelector("cross-validation")
        kdtree = bws._create_kdtree(self.weighted_points, True)
        ll = bws._log_likelihood_weights(self.weighted_points, kdtree.nodes, kdtree.leaves, kdtree.leaf_size, bw, bw, bw)
        npt.assert_almost_equal(ll, -763.6758826707483)
        
    def test_log_likelihood_eq(self, bw=0.25):
        """Test if weighted log likelihood with weights of 1 is equal to log likelihood."""
        bws = BandwidthSelector("cross-validation")
        kdtree = bws._create_kdtree(self.points, False)
        ll = bws._log_likelihood(self.points, kdtree.nodes, kdtree.leaves, kdtree.leaf_size, bw, bw, bw)
        
        non_weighted_points = np.column_stack([self.points, np.ones(self.points.shape[0])])
        kdtree_w = bws._create_kdtree(non_weighted_points, weighted=True)
        ll_w = bws._log_likelihood_weights(non_weighted_points, kdtree_w.nodes, kdtree_w.leaves, kdtree_w.leaf_size, bw, bw, bw)
        npt.assert_almost_equal(ll, ll_w)
        
class TestSTKDE(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.points = np.random.rand(1000, 3)
        self.w_points = np.column_stack([self.points, np.random.rand(1000,)])
        self.gdf = gpd.GeoDataFrame(
            {"time": self.w_points[:, 2], "weight": self.w_points[:, 3]},
            geometry=[Point(xy) for xy in self.w_points[:, :2]]
        )
    
    @staticmethod
    @njit
    def _naive_stkde(points, grid, bandwidths, weighted):
        """Naive implementation of STKDE algorithm."""
        xgrid, ygrid, tgrid = grid
        hx, hy, ht = bandwidths
        if weighted:
            weights = points[:, 3]
            n_eff = np.sum(weights)**2 / np.sum(weights**2)
        else:
            n_eff = len(points)
        normalization_factor = 1 / (n_eff * hx * hy * ht) * (0.75**3)
        stkde_values = np.empty((len(xgrid), len(ygrid), len(tgrid)), np.float64)
        
        # Loop through each voxel
        for i, x_voxel in enumerate(xgrid):
            for j, y_voxel in enumerate(ygrid):
                for k, t_voxel in enumerate(tgrid):
                    stkde_value_voxel = 0.0
                    
                    # Calculate bounds of influence for voxel
                    min_x = x_voxel - hx
                    max_x = x_voxel + hx
                    min_y = y_voxel - hy
                    max_y = y_voxel + hy
                    min_t = t_voxel - ht
                    max_t = t_voxel + ht
                    
                    # Loop through all points and calculate Epanechnikov kernel if in range
                    if weighted:
                        for point in points:
                            x_point, y_point, t_point, weight = point
                            if min_x < x_point < max_x and min_y < y_point < max_y and min_t < t_point < max_t:
                                # Calculate Epanechnikov kernel
                                x_epan = (1 - ((x_voxel - x_point) / hx) ** 2)
                                y_epan = (1 - ((y_voxel - y_point) / hy) ** 2)
                                t_epan = (1 - ((t_voxel - t_point) / ht) ** 2)
                                stkde_value_voxel += x_epan * y_epan * t_epan * weight
                    else:    
                        for point in points:
                            x_point, y_point, t_point = point
                            if min_x < x_point < max_x and min_y < y_point < max_y and min_t < t_point < max_t:
                                # Calculate Epanechnikov kernel
                                x_epan = (1 - ((x_voxel - x_point) / hx) ** 2)
                                y_epan = (1 - ((y_voxel - y_point) / hy) ** 2)
                                t_epan = (1 - ((t_voxel - t_point) / ht) ** 2)
                                stkde_value_voxel += x_epan * y_epan * t_epan
                    
                    # Multiply with normalization factor and store in grid
                    stkde_values[i,j,k] = stkde_value_voxel * normalization_factor
        return stkde_values
    
    def test_stkde(self):
        """Test if STKDE result matches naive result."""
        stkde = STKDE(self.gdf, "time", (2,2,2), (0.25, 0.25, 0.25))
        naive_stkde = self._naive_stkde(self.points, stkde.grid, stkde.bandwidths, False)
        npt.assert_allclose(stkde.result, naive_stkde)
    
    def test_weighted_stkde(self):
        """Test if weighted STKDE result matches naive result."""
        stkde = STKDE(self.gdf, "time", (32,32,32), (0.25, 0.25, 0.25), "weight")
        naive_stkde = self._naive_stkde(self.w_points, stkde.grid, stkde.bandwidths, True)
        npt.assert_allclose(stkde.result, naive_stkde) 

if __name__ == "__main__":
    unittest.main()
