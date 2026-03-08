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

# Standard library imports
from typing import Literal, Tuple
from math import log, ceil, log2
import warnings

# Third-party imports
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from scipy.optimize import minimize



class BandwidthSelector:
    """Class for estimating 3D KDE bandwidths.
    method : {"scott", "silverman", "cross_validation"}
        Bandwidth selection method. ``"scott"`` and ``"silverman"`` are fast
        rules of thumb that assume normality. ``"cross_validation"`` is slower
        but data-driven with no distribution assumption.
    """
    __slots__ = ("method",)
    
    def __init__(self, method:Literal["scott", "silverman", "cross-validation"]):              
        valid_methods = {"scott", "silverman", "cross-validation"}
        if method not in valid_methods:
            raise ValueError(
                    f"Unknown KDE bandwidth method '{method}'."
                    f"Choose from {list(valid_methods)}."
                )    
        self.method = method
        
    ## Private methods
    def _scott(self, points: NDArray[np.float64]) -> Tuple[np.float64, np.float64, np.float64]:
        """Scott's rule, see: https://w.wiki/G7TK"""
        n, d = points.shape
        n_eff = self._calculate_neff(points[:, 3]) if d == 4 else n
        std_devs = np.std(points[:, :3], axis=0, ddof=1)
        return tuple(std_devs * n_eff**(-1/7))

    def _silverman(self, points: NDArray[np.float64]) -> Tuple[np.float64, np.float64, np.float64]:
        """Silverman's rule, see: https://w.wiki/8Nz4"""
        n, d = points.shape
        n_eff = self._calculate_neff(points[:, 3]) if d == 4 else n
        std_devs = np.std(points[:, :3], axis=0, ddof=1)
        bandwidths = std_devs * (n_eff * 5 / 4)**(-1/7)
        return tuple(bandwidths)
    
    def _calculate_neff(self, weights: NDArray[np.float64]) -> np.float64:
        """Kish's effective sample size: (sum of weights)^2 / (sum of squared weights) (Kish, 1965)."""
        return np.sum(weights)**2 / np.sum(weights**2)
    
    def _create_kdtree(self, points:NDArray[np.float64], weighted:bool, leaf_size:int = 32) -> StaticKDTree:
        """Helper function to create KDTree.

        Args:
            points (NDArray[np.float64]):  Array of shape (n, 3) for [x, y, t] coordinates, or (n, 4) for [x, y, t, weights].
            weighted (bool): True if the points have associated weights.
            leaf_size (int, optional): Number of points stored in the KDTree leaves, multiples of 2 generally recommended. Defaults to 32.

        Returns:
            StaticKDTree: StaticKDTree instance.
        """
        kdtree = StaticKDTree(points=points, weighted=weighted, leaf_size=leaf_size)
        return kdtree

    def _cross_validation(
        self,
        points:NDArray[np.float64],
        replace_inf_value:float = 1e9,
    ) -> Tuple[np.float64,np.float64,np.float64]:
        """Leave-one-out cross validation, see Hu et al. (2018).
        Uses a static KDTree internally to speed up finding of points in range.

        Args:
            points (NDArray[np.float64]): Array of shape (n, 3) for [x, y, t] coordinates, or (n, 4) for [x, y, t, weights].
            replace_inf_value (float, optional): Value to replace NaN or infinite log likelihood values with. Defaults to 1e9.

        Returns:
            Tuple[np.float64,np.float64,np.float64]: Tuple of calculated bandwidths (x, y, t).
        """
        
        # Extract number of points and dimensions
        n, d = points.shape
                
        # Choose log likelihood function depending on if weights are present
        if d == 3:
            weighted = False
            ll_func = self._log_likelihood
        else:
            weighted = True
            ll_func = self._log_likelihood_weights
            
        # Create a KDTree for easy range querying inside these functions.
        # Range queries with the KDTree improve finding points in range from O(n) to O(n^(2/3) + m).
        kdtree = self._create_kdtree(points, weighted)
        nodes, leaves = kdtree.nodes, kdtree.leaves
        
        # Create the function which will be minimized. We want the maximum log likelihood so we minimize the negative log likelihood.
        def minimize_function(params):
            hx, hy, ht = params
            result = -ll_func(
                points=points,
                nodes=nodes,
                leaves=leaves,
                leaf_size=kdtree.leaf_size,
                x_bandwidth=hx,
                y_bandwidth=hy,
                t_bandwidth=ht
            )
            
            # If resulting value is -infinite or NaN ensure we replace it with a very high number. 
            # This allows the optimisation algorithm to ignore this area of the optimisation grid.
            if not np.isfinite(result):
                result = replace_inf_value
            return result
        
        # Define bounds for L-BFGS-B
        h_scott = self._scott(points)
        h_silverman = self._silverman(points)
        h_min = tuple(min(s, silv) for s, silv in zip(h_scott, h_silverman))
        h_max = tuple(max(s, silv) for s, silv in zip(h_scott, h_silverman))
        final_bounds = [(0.01 * hmin, 100.0 * hmax) for hmin, hmax in zip(h_min, h_max)]
        
        # Execute minimization
        result = minimize(minimize_function, x0=h_scott, bounds=final_bounds, method="L-BFGS-B")
        
        # Check if the result is degenerate
        boundary_dims = [
            (dim, value, lower, upper)
            for dim, value, (lower, upper) in zip(["x", "y", "t"], result.x, final_bounds)
            if np.isclose(value, lower, rtol=1e-6, atol=1e-10) or np.isclose(value, upper, rtol=1e-6, atol=1e-10)
        ]
        
        # Raise warning and fall back to Scott if degenerate
        if boundary_dims:
            boundary_str = ", ".join(
                f"{dim}={value:.3e} at [{lower:.3e}, {upper:.3e}]" 
                for dim, value, lower, upper in boundary_dims
            )
            warnings.warn(
                f"LOOCV bandwidth optimization hit search boundary ({boundary_str}). "
                "Cross-validation is degenerate, try eliminating duplicate points in dataset. "
                "Falling back to Scott's rule.",
                UserWarning,
                stacklevel=2
            )
            hx, hy, ht = h_scott
        else:
            hx, hy, ht = result.x     
        return hx, hy, ht
    
    # Numba internal methods @staticmethod to avoid compilation issues with passing self
    @staticmethod
    @njit(cache=True, parallel=True)
    def _log_likelihood(
        points:NDArray[np.float64],
        nodes:NDArray[np.void],
        leaves:NDArray[np.void],
        leaf_size:int,
        x_bandwidth:np.float64,
        y_bandwidth:np.float64,
        t_bandwidth:np.float64
    ) -> np.float64:
        """Calculates the log likelihood in parallel for every point.
        
        Args:
            points (NDArray[np.float64]): Array of shape (n, 3) for [x, y, t] coordinates. Must be float64 dtype.
            nodes (NDArray[np.void]): Structured array containing KDTree nodes.
            leaves (NDArray[np.void]): Structured array containing KDTree leaves.
            leaf_size (int): KDTree leaf size.
            x_bandwidth (np.float64): Testing bandwidth for x dimension.
            y_bandwidth (np.float64): Testing bandwidth for y dimension.
            t_bandwidth (np.float64): Testing bandwidth for t dimension.

        Returns:
            np.float64: Log likelihood value (maximum is better).
        """
        
        # Create empty array to store log values
        n = points.shape[0]
        log_stkde_results = np.empty(n, dtype=np.float64)
        
        # Calculate normalization factor (with n-1 to exclude the point itself)
        normalization_factor = 1 / ((n-1) * x_bandwidth * y_bandwidth * t_bandwidth) * 0.421875
        log_normalization_factor = log(normalization_factor)
        
        # Calculate STKDE for every point (focus point) in parallel
        for index in prange(n):
            
            # Create the boundingbox for the focus point
            x_value, y_value, t_value = points[index]
            xmin = x_value - x_bandwidth
            xmax = x_value + x_bandwidth
            ymin = y_value - y_bandwidth
            ymax = y_value + y_bandwidth
            tmin = t_value - t_bandwidth
            tmax = t_value + t_bandwidth
            bounds_min = np.array((xmin, ymin, tmin))
            bounds_max = np.array((xmax, ymax, tmax))
                    
            # Accumulate the STKDE value for the focus point by finding all points within bandwidth range in the KDTree.
            # To exclude the focus point we start the STKDE at -1.0. Epan = 1-u**2, with u=0.
            # We could exclude the focus point result from the KDTree, but using -1 is much faster and mathematically equal.
            stkde_value = -1.0
            
            # Stack for iterative traversal: (node_idx, axis)
            max_stack_size = ceil(log2(n / leaf_size)) + 2
            stack = np.empty((max_stack_size, 2), dtype=np.uint32)
            stack_ptr = 0
            stack[stack_ptr, 0] = 0  # root node
            stack[stack_ptr, 1] = 0  # x-axis
            stack_ptr += 1
            
            # Start traversal
            while stack_ptr > 0:
                stack_ptr -= 1 
                node_idx = stack[stack_ptr, 0]
                axis = stack[stack_ptr, 1]
                node = nodes[node_idx]
                
                # Check if this is a leaf node
                if node['leaf_idx'] != -1:
                    leaf = leaves[node['leaf_idx']]
                    count = leaf['count']
                    leaf_points = leaf['points']
                    
                    # Filter points within AABB
                    for i in range(count):
                        leaf_point = leaf_points[i]
                        leaf_point_x = leaf_point[0]
                        leaf_point_y = leaf_point[1]
                        leaf_point_t = leaf_point[2]
                        
                        # Leaf point is within bounding box so contributes to KDE for the original point in the outermost loop
                        if xmin < leaf_point_x < xmax and ymin < leaf_point_y < ymax and tmin < leaf_point_t < tmax:
                            # Calculate 3D Epanechnikov kernel
                            x_epan = (1 - ((x_value - leaf_point_x) / x_bandwidth) ** 2)
                            y_epan = (1 - ((y_value - leaf_point_y) / y_bandwidth) ** 2)
                            t_epan = (1 - ((t_value - leaf_point_t) / t_bandwidth) ** 2)
                            stkde_value += x_epan * y_epan * t_epan
                    continue
                
                # Internal node -> check which children to visit
                split_val = node['split_val']
                next_axis = (axis + 1) % 3
                
                # Check if AABB overlaps with left child
                if bounds_min[axis] <= split_val:
                    stack[stack_ptr, 0] = node['left_idx']
                    stack[stack_ptr, 1] = next_axis
                    stack_ptr += 1
                
                # Check if AABB overlaps with right child
                if bounds_max[axis] > split_val:
                    stack[stack_ptr, 0] = node['right_idx']
                    stack[stack_ptr, 1] = next_axis
                    stack_ptr += 1
            
            # Convert the STKDE value into log space, adding small Epsilon value to avoid log(0)
            # Multiply STKDE value with normalization factor, using ln(a) + ln(b) = ln(a*b), a ≠ 0 because of Epsilon
            log_stkde_results[index] = log(stkde_value + 1e-300) + log_normalization_factor
            
        # Calculate sum of natural logarithm for the entire array 
        # NOTE: Atomic add would avoid loop in np.sum but isn't available in Numba yet
        log_likelihood = np.sum(log_stkde_results)
        return log_likelihood
    
    
    @staticmethod
    @njit(cache=True, parallel=True)
    def _log_likelihood_weights(
        points:NDArray[np.float64],
        nodes:NDArray[np.void],
        leaves:NDArray[np.void],
        leaf_size:int,
        x_bandwidth:np.float64,
        y_bandwidth:np.float64,
        t_bandwidth:np.float64
    ) -> np.float64:
        """Calculates the log likelihood in parallel for every point.
        
        Args:
            points (NDArray[np.float64]): Array of shape (n, 4) for [x, y, t, weights] coordinates. Must be float64 dtype.
            nodes (NDArray[np.void]): Structured array containing KDTree nodes.
            leaves (NDArray[np.void]): Structured array containing KDTree leaves.
            leaf_size (int): KDTree leaf size.
            x_bandwidth (np.float64): Testing bandwidth for x dimension.
            y_bandwidth (np.float64): Testing bandwidth for y dimension.
            t_bandwidth (np.float64): Testing bandwidth for t dimension.

        Returns:
            np.float64: Log likelihood value (maximum is better).
        """
        
        # Create empty array to store log values
        n = points.shape[0]
        log_stkde_results = np.empty(n, dtype=np.float64)
        
        # Calculate total sum of weights, and sum of squared weights, to calculate neff (Kish, 1965) in the parallel loop.
        sum_weights = 0.0
        sum_squared_weights = 0.0
        for i in range(n):
            val = points[i, 3]
            sum_weights += val
            sum_squared_weights += val ** 2
            
        # Precompute reusable part of the normalization factor. See: (0.421875)/(neff * hx * hy * ht) = 1/neff * (0.421875)/(hx * hy * ht)
        not_neff_normalization_factor = (0.421875)/(x_bandwidth * y_bandwidth * t_bandwidth)
        
        # Calculate STKDE for every point (focus point) in parallel
        for index in prange(n):
            
            # Create the boundingbox for the focus point
            x_value, y_value, t_value, weight = points[index]
            xmin = x_value - x_bandwidth
            xmax = x_value + x_bandwidth
            ymin = y_value - y_bandwidth
            ymax = y_value + y_bandwidth
            tmin = t_value - t_bandwidth
            tmax = t_value + t_bandwidth
            bounds_min = np.array([xmin, ymin, tmin])
            bounds_max = np.array([xmax, ymax, tmax])  
            
            # Calculate Neff and normalization factor for the focus point
            neff = ((sum_weights - weight)**2)/(sum_squared_weights - (weight**2))
            normalization_factor = 1/neff * not_neff_normalization_factor
            log_normalization_factor = log(normalization_factor)
            
            # Accumulate the STKDE value for the focus point by finding all points within bandwidth range in the KDTree.
            # To exclude the focus point we start the STKDE at -1.0 * weight. Epan = 1-u**2, with u=0.
            # We could exclude the focus point result from the KDTree, but using -weight is much faster and mathematically equal.
            stkde_value = -weight
            
            # Stack for iterative traversal: (node_idx, axis)
            max_stack_size = ceil(log2(n / leaf_size)) + 2
            stack = np.empty((max_stack_size, 2), dtype=np.uint32)
            stack_ptr = 0
            stack[stack_ptr, 0] = 0  # root node
            stack[stack_ptr, 1] = 0  # x-axis
            stack_ptr += 1
            
            # Start traversal
            while stack_ptr > 0:
                stack_ptr -= 1 
                node_idx = stack[stack_ptr, 0]
                axis = stack[stack_ptr, 1]
                node = nodes[node_idx]
                
                # Check if this is a leaf node
                if node['leaf_idx'] != -1:
                    leaf = leaves[node['leaf_idx']]
                    count = leaf['count']
                    leaf_points = leaf['points']
                    
                    # Filter points within AABB
                    for i in range(count):
                        leaf_point = leaf_points[i]
                        leaf_point_x = leaf_point[0]
                        leaf_point_y = leaf_point[1]
                        leaf_point_t = leaf_point[2]
                        leaf_point_w = leaf_point[3]
                        
                        # Leaf point is within bounding box so contributes to KDE for the original point in the outermost loop
                        if xmin < leaf_point_x < xmax and ymin < leaf_point_y < ymax and tmin < leaf_point_t < tmax:
                            # Calculate 3D Epanechnikov kernel
                            x_epan = (1 - ((x_value - leaf_point_x) / x_bandwidth) ** 2)
                            y_epan = (1 - ((y_value - leaf_point_y) / y_bandwidth) ** 2)
                            t_epan = (1 - ((t_value - leaf_point_t) / t_bandwidth) ** 2)
                            stkde_value += x_epan * y_epan * t_epan * leaf_point_w
                    continue
                
                # Internal node -> check which children to visit
                split_val = node['split_val']
                next_axis = (axis + 1) % 3
                
                # Check if AABB overlaps with left child
                if bounds_min[axis] <= split_val:
                    stack[stack_ptr, 0] = node['left_idx']
                    stack[stack_ptr, 1] = next_axis
                    stack_ptr += 1
                
                # Check if AABB overlaps with right child
                if bounds_max[axis] > split_val:
                    stack[stack_ptr, 0] = node['right_idx']
                    stack[stack_ptr, 1] = next_axis
                    stack_ptr += 1
            
            # Convert the STKDE value into log space, adding small Epsilon value to avoid log(0)
            # Multiply STKDE value with normalization factor, using ln(a) + ln(b) = ln(a*b), a ≠ 0 because of Epsilon
            log_stkde_results[index] = log(stkde_value + 1e-300) + log_normalization_factor
            
        # Calculate sum of natural logarithm for the entire array 
        # NOTE: Atomic add would avoid loop in np.sum but isn't available in Numba yet
        log_likelihood = np.sum(log_stkde_results)
        return log_likelihood
          
    ## Public methods
    def estimate(self, points:NDArray[np.float64]) -> Tuple[np.float64,np.float64,np.float64]:   
        """Approximate optimal bandwidths for 3D kernel density estimation.

        Args:
            points (NDArray[np.float64]): Array of shape (n, 3) for [x, y, t] coordinates, or (n, 4) for [x, y, t, weights]. Must be float64 dtype.

        Returns:
            Tuple[float,float,float]: Estimated optimal bandwidth values for (x, y, t) dimensions.
        """
        if not isinstance(points, np.ndarray):
            raise ValueError(f"Input must be a NumPy array, got {type(points)}.")
        if points.dtype != np.float64:
            raise ValueError(f"Array must be float64, got {points.dtype}.")
        n, d = points.shape
        has_weights = d == 4
        if points.ndim != 2 or d not in (3, 4) or n == 0:
            raise ValueError(f"Array must have shape (n, 3) or (n, 4), where n is not 0, got {points.shape}.")
        if has_weights:
            if np.any((points[:, 3] <= 0) | ~np.isfinite(points[:, 3])):
                raise ValueError("Points array contains invalid weights, either 0, infinity, negative, or NaN. Please remove invalid weights.")
        
        match self.method:
            case 'scott':
                bandwidths = self._scott(points)
            case 'silverman':
                bandwidths = self._silverman(points)
            case 'cross-validation':
                bandwidths = self._cross_validation(points)
            case _:
                raise ValueError(f"Unsupported method supplied: {self.method}.")
        
        assert all(bw > 0 for bw in bandwidths), f"Bandwidth calculation returned non-positive values: {bandwidths}."
        return bandwidths