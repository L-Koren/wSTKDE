# MIT License

# Copyright (c) 2024 - 2026 L. Koren

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
from src.bandwidth_selection import BandwidthSelector

# Standard library imports
from pathlib import Path
import warnings
from math import ceil
from typing import Optional, Literal, Tuple

# Third-party imports
import pandas as pd
import geopandas as gpd
from shapely import bounds, get_type_id, GeometryType
import numpy as np
from numpy.typing import NDArray
from numba import njit, types as nbt
import pyvista as pv



### STKDE Class
class STKDE:
    """Calculates (weighted) Spatio-Temporal Kernel Density Estimate using custom algorithm. General methodology based on Hu, et al. (2018).
    
    Args:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame containing the points to calculate the STKDE for.
        time_col (str):
            Name of the column in the GeoDataFrame that contains time values.
        grid_size (Tuple[int, int, int]):
            Size of the estimation grid for each axis [x,y,t].
        bandwidths (Tuple[float, float, float] | Literal["scott", "silverman", "cross-validation"], optional): 
            Bandwidth to use for each axis [x,y,t], or method to estimate bandwidths with. 
            For explanation on methods see BandwidthSelector documentation.
            Defaults to "scott".
        weight_col (Optional[str]):
            Optional name of the column in the GeoDataFrame that contains weights for the points. Defaults to None.
    
    ## Attributes:
        bandwidths: Bandwidths that were used for the STKDE.
        grid: Grid that was used for the STKDE. Stored as unique centerpoint coordinate per dimension [x,y,t].
        result: 3D array of STKDE results for every voxel in the supplied grid.
    """
    __slots__ = ("bandwidths", "grid", "result")
    
    # Numba type signatures
    _POINTS_ARRAY = nbt.Array(dtype=nbt.float64, ndim=2, layout='C', readonly=True, aligned=True)
    _FLOAT64_ARRAY_1D = nbt.Array(dtype=nbt.float64, ndim=1, layout='C', readonly=False, aligned=True)
    _BOUNDING_BOX = nbt.UniTuple(dtype=nbt.float64, count=6)
    _GRID = nbt.UniTuple(dtype=_FLOAT64_ARRAY_1D.copy(readonly=True), count=3)
    _BANDWIDTHS = nbt.UniTuple(dtype=nbt.float64, count=3)
    _STKDE_VALUES = nbt.Array(dtype=nbt.float64, ndim=3, layout='C', readonly=False, aligned=True)
    
    ## Methods
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        time_col: str,
        grid_size: Tuple[int, int, int], 
        bandwidths: Tuple[float, float, float] | Literal["scott", "silverman", "cross-validation"] = "scott",
        weight_col: Optional[str] = None
    ):
        self._validate_inputs(gdf, time_col, grid_size, bandwidths, weight_col)
        points = self._extract_points(gdf, time_col, weight_col)
        
        # Create grid and bandwidths
        bbox = self._calculate_bounding_box(points)
        self.grid = self._generate_voxel_grid(grid_size, bbox)
        if isinstance(bandwidths, str):
            bws = BandwidthSelector(bandwidths)
            self.bandwidths = bws.estimate(points)
        else:
            hx, hy, ht = bandwidths
            bandwidth_values = (np.float64(hx), np.float64(hy), np.float64(ht))
            self.bandwidths = bandwidth_values
        self._check_bandwidth_grid_ratio()
            
        # Calculate STKDE
        if weight_col:
            stkde_values = self._calculate_stkde_weighted(points, self.grid, self.bandwidths)
        else:
            stkde_values = self._calculate_stkde(points, self.grid, self.bandwidths)
        self.result = stkde_values
        
    
    ## Private methods.
    def _validate_inputs(self, gdf, time_col, grid_size, bandwidths, weight_col) -> None:
        """Helper function to validate the init inputs."""
        
        # Validate GeoDataFrame (gdf)
        if gdf is None or len(gdf) == 0:
            raise ValueError(
                "GeoDataFrame is empty.")
        if len(gdf) < 100:
            warnings.warn(
                "GeoDataFrame has less than 100 points, results may not be reliable."
            )
        if time_col not in gdf.columns:
            raise KeyError(
                f"Provided time column '{time_col}' not found in GeoDataFrame's columns: {gdf.columns}."
            )
        if not pd.api.types.is_numeric_dtype(gdf[time_col]):
            raise TypeError(f"Time column '{time_col}' must be of numeric type. Please convert column to float or int.")
        if gdf[time_col].isna().any():
            raise ValueError(f"Found NA values in time column '{time_col}', remove these records or fill the values.")
        
        # Validate grid_size
        if len(grid_size) != 3:
            raise ValueError(
                "Grid size must be provided in the format (x_voxels, y_voxels, t_voxels)."
            )
        if not all(isinstance(v, int) and v > 0 for v in grid_size):
            raise ValueError(
                f"Cannot generate a grid with dimensions, expecting three positive integers, recieved: {grid_size}."
            )
        
        # Validate bandwidths
        if not isinstance(bandwidths, str):
            if len(bandwidths) != 3:
                raise ValueError(
                    "Numeric bandwidths must be provided in the format (x_bandwidth, y_bandwidth, t_bandwidth)."
                )
            if not all(isinstance(b, (int, float)) and b > 0 for b in bandwidths):
                raise ValueError(
                    f"All numeric bandwidths must be positive floats, recieved: {bandwidths}."
                )
            
        # Validate weights
        if weight_col:
            if weight_col not in gdf.columns:
                raise KeyError(
                    f"Weight column {weight_col} not found in GeoDataFrame."
                )
            if not (gdf[weight_col].notna() & (gdf[weight_col] > 0)).all():
                raise ValueError(
                    f"Weight column {weight_col} contains invalid weights"
                    "please make sure that all values are positive numbers."
                )
    
    def _check_bandwidth_grid_ratio(self, threshold:float = 2.0):
        """Check if bandwidths are at least 2 times grid size in each dimension.""" 
        invalid_ratios = []
        for dim, grid, bw in zip("xyt", self.grid, self.bandwidths, strict=True):
            grid_size = grid[1] - grid[0]
            ratio = bw / grid_size
            if ratio < threshold:
                invalid_ratios.append(f"    Dimension {dim}: Bandwidth = {bw:.4f}, Grid = {grid_size:.4f}, Ratio = {ratio:.4f}")
        if invalid_ratios:
            raise ValueError(
                "The ratio between bandwidth and grid size for the following dimensions is too low (< 2.0):\n"
                + "\n".join(invalid_ratios)
                + "\nIncrease the grid resolution, or increase the size of the bandwidths."
            )
            
    def _extract_points(self, gdf:gpd.GeoDataFrame, time_col:str, weight_col:str|None) -> NDArray[np.float64]:
        """Extract the points from the GeoDataFrame to a Numpy array.

        Args:
            gdf (gpd.GeoDataFrame):
                GeoDataFrame containing the points to calculate the STKDE for.
            time_col (str):
                Name of the column in the GeoDataFrame that contains time values
            weight_col (str | None): 
                Name of the column in the GeoDataFrame that contains weights for the points, None if no weights.

        Returns:
            NDArray[np.float64]: Numpy array of size (n, 3) or (n, 4) containing all point coordinates and optionally weights.
        """
        
        # Validate geometry
        geom_arr = gdf.geometry.values
        if not np.all((get_type_id(geom_arr) == GeometryType.POINT)):
            raise ValueError("Non-point geometries found in the geodataframe.")
        
        # Create points array
        bounds_arr = bounds(geom_arr)[:, :2] # You could use shapely.get_coordinates(), but in testing this was much faster (3x)
        if weight_col:
            points = np.column_stack((bounds_arr, gdf[time_col].to_numpy(np.float64), gdf[weight_col].to_numpy(np.float64)))
            if np.any((points[:, 3] <= 0) | ~np.isfinite(points[:, 3])):
                raise ValueError("Points array contains invalid weights, either 0, infinity, negative, or NaN. Please remove invalid weights.")
        else:
            points = np.column_stack((bounds_arr, gdf[time_col].to_numpy(np.float64)))
        return points
        
    
    # Numba methods use @staticmethod to allow for compilation.
    @staticmethod
    @njit((_BOUNDING_BOX)(_POINTS_ARRAY), fastmath=True, cache=True)
    def _calculate_bounding_box(points:NDArray[np.float64]) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:
        """Calculates 3D bounding box for array.

        Args:
            points (NDArray[np.float64]):
                Numpy array of size (n, 3) or (n, 4) containing all point coordinates and optionally weights.

        Returns:
            Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]: 
                Tuple of xmin, xmax, ymin, ymax, tmin, and tmax.
        """
        
        xmin = xmax = points[0, 0]
        ymin = ymax = points[0, 1]
        tmin = tmax = points[0, 2]
        
        for i in range(1, points.shape[0]):
            x = points[i, 0]
            y = points[i, 1]
            t = points[i, 2]
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            tmin = min(tmin, t)
            tmax = max(tmax, t)
        return xmin, xmax, ymin, ymax, tmin, tmax
    
    @staticmethod
    @njit((_GRID)(nbt.UniTuple(nbt.uint32, 3), _BOUNDING_BOX), cache=True)
    def _generate_voxel_grid(
        grid_size:Tuple[int, int, int],
        bounding_box:Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Generate voxel grid coordinates that extend half a voxel beyond the bounding box to ensure full grid coverage.
    
        Args:
            grid_size (Tuple[np.uint32, np.uint32, np.uint32]):
                Tuple of number of voxels grid should contain in x, y, and t dimension.
            bounding_box (Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]):
                Tuple of xmin, xmax, ymin, ymax, tmin, and tmax.

        Returns:
            Tuple[NDArray, NDArray, NDArray]: Linearly spaced arrays for the x, y, and t dimensions of the voxel grid.
        """

        # Calculate bounding box size
        xmin, xmax, ymin, ymax, tmin, tmax = bounding_box
        x_length = xmax - xmin
        y_length = ymax - ymin
        t_length = tmax - tmin
        
        # Calculate n-1 voxel sizes for each dimension
        x_voxels, y_voxels, t_voxels = grid_size
        x_voxel_size = x_length / (x_voxels - 1)
        y_voxel_size = y_length / (y_voxels - 1)
        t_voxel_size = t_length / (t_voxels - 1)
        
        # Half voxel size
        x_half_voxel_size = x_voxel_size / 2
        y_half_voxel_size = y_voxel_size / 2
        t_half_voxel_size = t_voxel_size / 2
        
        # First point is min - half of voxel size
        first_voxel_center_x = xmin - x_half_voxel_size
        first_voxel_center_y = ymin - y_half_voxel_size
        first_voxel_center_t = tmin - t_half_voxel_size
        
        # Max voxel center
        max_voxel_center_x = xmax + x_half_voxel_size
        max_voxel_center_y = ymax + y_half_voxel_size
        max_voxel_center_t = tmax + t_half_voxel_size
        
        # Create grid for each dimension
        xgrid = np.linspace(first_voxel_center_x, max_voxel_center_x, x_voxels)
        ygrid = np.linspace(first_voxel_center_y, max_voxel_center_y, y_voxels)
        tgrid = np.linspace(first_voxel_center_t, max_voxel_center_t, t_voxels)
        return xgrid, ygrid, tgrid
    
    @staticmethod
    @njit(_STKDE_VALUES(_POINTS_ARRAY, _GRID, _BANDWIDTHS), cache=True)
    def _calculate_stkde(
        points: NDArray[np.float64],
        grid:Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        bandwidths: Tuple[np.float64, np.float64, np.float64]
    ) -> NDArray[np.float64]:
        """Calculates Spatio-Temporal Kernel Density Estimation (STKDE).
        STKDE is calculated using a very fast custom algorithm, general method is based on Hu, et al. (2018).

        Args:
            points (NDArray[np.float64]): 
                Array of shape (n, 3) for [x, y, t] coordinates. Must be float64 dtype.
            grid (Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]): 
                Tuple of three arrays for [x, y, t] voxel centerpoint coordinates.
            bandwidths (Tuple[np.float64, np.float64, np.float64]): 
                Bandwidths for the x, y, and t dimensions.

        Returns:
            NDArray[np.float64]: 3D Array of STKDE values.
        """
        # Extract required values, Numba enforces types so no need for type checks
        n, d = points.shape
        if d != 3:
            raise ValueError("Input array should have 3 columns for x, y and t coordinates.")
        x_voxel_centers, y_voxel_centers, t_voxel_centers = grid
        x_bandwidth, y_bandwidth, t_bandwidth = bandwidths

        # Extract the number of layers
        number_of_x_layers = x_voxel_centers.shape[0]
        number_of_y_layers = y_voxel_centers.shape[0]
        number_of_t_layers = t_voxel_centers.shape[0]

        # Extract the minimum values
        x_minimum_value = x_voxel_centers[0]
        y_minimum_value = y_voxel_centers[0]
        t_minimum_value = t_voxel_centers[0]

        # Extract the voxel sizes, or the distance equal to an index 'i' in each direction
        x_distance = x_voxel_centers[1] - x_minimum_value
        y_distance = y_voxel_centers[1] - y_minimum_value
        t_distance = t_voxel_centers[1] - t_minimum_value
        # Precompute reciprocal distances between voxels, multiplication is slightly faster than division in the loop
        x_distance_recip = 1 / x_distance
        y_distance_recip = 1 / y_distance
        t_distance_recip = 1 / t_distance

        # Express bandwidths in terms of index
        x_bandwidth_i = x_bandwidth / x_distance
        y_bandwidth_i = y_bandwidth / y_distance
        t_bandwidth_i = t_bandwidth / t_distance
        
        # 3D array filled with zeroes
        stkde_values = np.zeros((number_of_x_layers, number_of_y_layers, number_of_t_layers), dtype=np.float64)
        
        # Create constantly overwritten buffers to write intermediate Epanechnikov values to.
        max_voxels_x = ceil(2 * x_bandwidth_i)
        max_voxels_y = ceil(2 * y_bandwidth_i) 
        x_epanechnikov_values = np.empty(max_voxels_x, dtype=np.float64)
        y_epanechnikov_values = np.empty(max_voxels_y, dtype=np.float64)

        # Loop through points finding voxels in range of them,
        # then calculate the point's kernel contributions and store these in the stkde_values array
        for i in range(n):

            # Extract x,y,t value of the current point
            x_value, y_value, t_value = points[i]

            # Convert value to be expressed in terms of grid indexes
            x_val_i = (x_value - x_minimum_value) * x_distance_recip
            y_val_i = (y_value - y_minimum_value) * y_distance_recip
            t_val_i = (t_value - t_minimum_value) * t_distance_recip
            
            # Determine which voxels are in bandwidth range of the point
            # Calculate the indexes of the voxels in range per dimension
            lb_x = max(ceil(x_val_i - x_bandwidth_i), 0)
            ub_x = min(ceil(x_val_i + x_bandwidth_i), number_of_x_layers)  
            if ub_x < 1:
                continue
            lb_y = max(ceil(y_val_i - y_bandwidth_i), 0)
            ub_y = min(ceil(y_val_i + y_bandwidth_i), number_of_y_layers)  
            if ub_y < 1:
                continue
            lb_t = max(ceil(t_val_i - t_bandwidth_i), 0)
            ub_t = min(ceil(t_val_i + t_bandwidth_i), number_of_t_layers)  
            if ub_t < 1:
                continue

            # Precompute Epanechnikov kernel values per dimension, skip the t dimension as we calculate it later
            for j, x_voxel_value in enumerate(x_voxel_centers[lb_x:ub_x]):
                x_epanechnikov_values[j] = 1 - ((x_voxel_value - x_value) / x_bandwidth)**2
            for j, y_voxel_value in enumerate(y_voxel_centers[lb_y:ub_y]):
                y_epanechnikov_values[j] = 1 - ((y_voxel_value - y_value) / y_bandwidth)**2

            # Calculate cartesian product of the Epanechnikov kernels for the point and sum with the values in the stkde values array
            # NOTE: This loop ordering is theoretically suboptimal for memory access but because the stacked loops are typically small, this is faster
            for j, t_voxel_value in enumerate(t_voxel_centers[lb_t:ub_t]):
                t_index = lb_t + j
                t_epanechnikov_value = 1 - ((t_voxel_value - t_value) / t_bandwidth)**2
                for k in range(ub_y - lb_y):
                    ty_epanechnikov_value = t_epanechnikov_value * y_epanechnikov_values[k]
                    y_index = lb_y + k
                    for l in range(ub_x - lb_x):
                        # Add the combined Epanechnikov value to the appropiate voxels
                        stkde_values[lb_x + l, y_index, t_index] += ty_epanechnikov_value * x_epanechnikov_values[l]

        # Finally multiply all values with the normalization factor
        normalization_factor = 1 / (n * x_bandwidth * y_bandwidth * t_bandwidth) * 0.421875 # 0.75**3 (once for each kernel)
        for x in range(number_of_x_layers):
            for y in range(number_of_y_layers):
                for t in range(number_of_t_layers):
                    stkde_values[x,y,t] *= normalization_factor 
        return stkde_values

    @staticmethod
    @njit(_STKDE_VALUES(_POINTS_ARRAY, _GRID, _BANDWIDTHS), cache=True)
    def _calculate_stkde_weighted(
        points: NDArray[np.float64],
        grid:Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        bandwidths: Tuple[np.float64, np.float64, np.float64]
    ) -> NDArray[np.float64]:
        """Calculates weighted Spatio-Temporal Kernel Density Estimation (STKDE).
        STKDE is calculated using a very fast custom algorithm, general unweighted method is based on Hu, et al. (2018).

        Args:
            points (NDArray[np.float64]): 
                Array of shape (n, 4) for [x, y, t, weight] coordinates. Must be float64 dtype.
            grid (Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]): 
                Tuple of three arrays for [x, y, t] voxel centerpoint coordinates.
            bandwidths (Tuple[np.float64, np.float64, np.float64]): 
                Bandwidths for the x, y, and t dimensions.

        Returns:
            NDArray[np.float64]: 3D Array of STKDE values.
        """
        # Extract required values, Numba enforces types so no need for type checks
        n, d = points.shape
        if d != 4:
            raise ValueError("Input array should have 4 columns for x, y, t coordinates and weights.")
        x_voxel_centers, y_voxel_centers, t_voxel_centers = grid
        x_bandwidth, y_bandwidth, t_bandwidth = bandwidths

        # Extract the number of layers
        number_of_x_layers = x_voxel_centers.shape[0]
        number_of_y_layers = y_voxel_centers.shape[0]
        number_of_t_layers = t_voxel_centers.shape[0]

        # Extract the minimum values
        x_minimum_value = x_voxel_centers[0]
        y_minimum_value = y_voxel_centers[0]
        t_minimum_value = t_voxel_centers[0]

        # Extract the voxel sizes, or the distance equal to an index 'i' in each direction
        x_distance = x_voxel_centers[1] - x_minimum_value
        y_distance = y_voxel_centers[1] - y_minimum_value
        t_distance = t_voxel_centers[1] - t_minimum_value
        # Precompute reciprocal distances between voxels, multiplication is slightly faster than division in the loop
        x_distance_recip = 1 / x_distance
        y_distance_recip = 1 / y_distance
        t_distance_recip = 1 / t_distance

        # Express bandwidths in terms of index
        x_bandwidth_i = x_bandwidth / x_distance
        y_bandwidth_i = y_bandwidth / y_distance
        t_bandwidth_i = t_bandwidth / t_distance
        
        # 3D array filled with zeroes
        stkde_values = np.zeros((number_of_x_layers, number_of_y_layers, number_of_t_layers), dtype=np.float64)
        
        # Create constantly overwritten buffers to write intermediate Epanechnikov values to.
        max_voxels_x = ceil(2 * x_bandwidth_i)
        max_voxels_y = ceil(2 * y_bandwidth_i) 
        x_epanechnikov_values = np.empty(max_voxels_x, dtype=np.float64)
        y_epanechnikov_values = np.empty(max_voxels_y, dtype=np.float64)
        
        # Variables for calculating weight adjusted n value -> neff (Kish, 1965)
        sum_weights = 0.0
        sum_squared_weights = 0.0

        # Loop through points finding voxels in range of them,
        # then calculate the point's kernel contributions and store these in the stkde_values array
        for i in range(n):

            # Extract x,y,t value of the current point and the weight
            x_value, y_value, t_value, weight = points[i]
            sum_weights += weight
            sum_squared_weights += weight**2

            # Convert value to be expressed in terms of grid indexes
            x_val_i = (x_value - x_minimum_value) * x_distance_recip
            y_val_i = (y_value - y_minimum_value) * y_distance_recip
            t_val_i = (t_value - t_minimum_value) * t_distance_recip
            
            # Determine which voxels are in bandwidth range of the point
            # Calculate the indexes of the voxels in range per dimension
            lb_x = max(ceil(x_val_i - x_bandwidth_i), 0)
            ub_x = min(ceil(x_val_i + x_bandwidth_i), number_of_x_layers)  
            if ub_x < 1:
                continue
            lb_y = max(ceil(y_val_i - y_bandwidth_i), 0)
            ub_y = min(ceil(y_val_i + y_bandwidth_i), number_of_y_layers)  
            if ub_y < 1:
                continue
            lb_t = max(ceil(t_val_i - t_bandwidth_i), 0)
            ub_t = min(ceil(t_val_i + t_bandwidth_i), number_of_t_layers)  
            if ub_t < 1:
                continue

            # Precompute Epanechnikov kernel values per dimension, skip the t dimension as we calculate it later
            for j, x_voxel_value in enumerate(x_voxel_centers[lb_x:ub_x]):
                x_epanechnikov_values[j] = 1 - ((x_voxel_value - x_value) / x_bandwidth)**2
            for j, y_voxel_value in enumerate(y_voxel_centers[lb_y:ub_y]):
                y_epanechnikov_values[j] = 1 - ((y_voxel_value - y_value) / y_bandwidth)**2

            # Calculate cartesian product of the Epanechnikov kernels for the point and sum with the values in the stkde values array
            # NOTE: This loop ordering is theoretically suboptimal for memory access
            # but because the stacked loops are typically small, this is faster
            for j, t_voxel_value in enumerate(t_voxel_centers[lb_t:ub_t]):
                t_index = lb_t + j
                t_epanechnikov_value = (1 - ((t_voxel_value - t_value) / t_bandwidth)**2)  * weight # Multiply with weight
                for k in range(ub_y - lb_y):
                    ty_epanechnikov_value = t_epanechnikov_value * y_epanechnikov_values[k]
                    y_index = lb_y + k
                    for l in range(ub_x - lb_x):
                        # Add the combined Epanechnikov value to the appropiate voxels
                        stkde_values[lb_x + l, y_index, t_index] += ty_epanechnikov_value * x_epanechnikov_values[l]

        # Finally multiply all values with the neff normalization factor
        neff = (sum_weights**2) / sum_squared_weights
        normalization_factor = 1 / (neff * x_bandwidth * y_bandwidth * t_bandwidth) * 0.421875 # 0.75**3 (once for each kernel)
        if not (normalization_factor > 0 and np.isfinite(normalization_factor)):
            raise ValueError("Please check input weights, it is likely that one of the weights is 0, NaN, or infinite.")

        for x in range(number_of_x_layers):
            for y in range(number_of_y_layers):
                for t in range(number_of_t_layers):
                    stkde_values[x,y,t] *= normalization_factor 
        return stkde_values

    def output_as_vtk(self, output_file:str|Path) -> None:
        """Write STKDE values to a Visualization Toolkit (VTK) ImageData file (.vtk/.vti).

        Args:
            output_file (str | Path): Location and name of where the file will be stored, NOTE: Overwrites existing files.
        """
        # Extract data from self
        xgrid, ygrid, tgrid = self.grid
        min_x, min_y, min_t = xgrid[0], ygrid[0], tgrid[0]
        
        # Get distance between voxels 
        x_distance = xgrid[1] - min_x
        y_distance = ygrid[1] - min_y
        t_distance = tgrid[1] - min_t
        
        # Number of voxels
        x_num_voxels = xgrid.shape[0]
        y_num_voxels = ygrid.shape[0]
        t_num_voxels = tgrid.shape[0]
        
        # Set the grid dimensions (+1 because the dimensions are the corners of the voxels)
        dimensions = (x_num_voxels + 1, y_num_voxels + 1, t_num_voxels + 1)
        # Set the grid origin and spacing
        spacing = (x_distance, y_distance, t_distance)
        origin = (min_x - x_distance / 2, min_y - y_distance / 2, min_t - t_distance / 2)

        # Create the vtk grid
        vtk_grid = pv.ImageData(dimensions=dimensions, spacing=spacing, origin=origin)
        
        # Add the stkde data
        # Image grid requires input to be in 'kji' order. See: https://vtk.org/doc/nightly/html/classvtkStructuredGrid.html
        vtk_grid.cell_data["stkde_values"] = self.result.T.flat # type: ignore
        
        # Output to file
        output_file = Path(output_file)
        if not output_file.suffix:
            output_file = output_file.with_suffix(".vti")
        elif output_file.suffix not in {".vti", ".vtk"}:
            raise ValueError(f"Invalid file extension: {output_file.suffix}. Must be .vti or .vtk.")
        if not output_file.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {output_file.parent}.")
        vtk_grid.save(output_file)