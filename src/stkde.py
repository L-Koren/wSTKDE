# MIT License

# Copyright (c) 2024 L. Koren

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


# Standard library imports
from pathlib import Path
import warnings
import time
from math import log, ceil
from typing import Optional, Tuple, Literal

# Third-party imports
# Dataframe manipulation
import pandas as pd
import geopandas as gpd

# Scientific computation
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from numba import njit, prange
from numba.types import float64, int64, Array, boolean, uint32, UniTuple #type:ignore

# Output to file
import pyvista as pv



### Functions
@njit(UniTuple(float64, 6)(Array(float64, 2, 'C', True, aligned=True)), fastmath=True, cache=True)
def _find_extrema_xyt(arr:NDArray[float64]) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:
    """Find the extrema in the first three columns of an array, x,y,t in this case.
        Parallelizing this function is slower up to about 5 million points, after that it is faster, therefore not worth it to parallelize.
        One could further consider employing a different approach like the one proposed by MSeifert here: https://stackoverflow.com/a/41733144,
        this was so close in testing that it was deemed not worth it to implement at the cost of readability.

    Args:
        arr (NDArray[float64]): Array containing the x, y, t coordinates of points, can have any number of columns.

    Returns:
        Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]: Tuple with the minimum and maximum values of x, y, and t,
        returned as xmin, xmax, ymin, ymax, tmin, tmax 
    """
    xmin = xmax = arr[0, 0]
    ymin = ymax = arr[0, 1]
    tmin = tmax = arr[0, 2]
    
    for i in range(1, arr.shape[0]):
        x = arr[i, 0]
        y = arr[i, 1]
        t = arr[i, 2]
        
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        tmin = min(tmin, t)
        tmax = max(tmax, t)

    return xmin, xmax, ymin, ymax, tmin, tmax


@njit(float64(Array(float64, 2, 'C', True, aligned=True), float64, float64, float64, float64, float64, float64, float64), cache=True)
def calculate_stkde_for_voxel(coords_array:NDArray[np.float64], x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64, 
                           x_voxel_coordinate:np.float64, y_voxel_coordinate:np.float64, t_voxel_coordinate:np.float64, normalization_factor:np.float64) -> np.float64:
    """Calculates STKDE value, using Epanechnikov kernel for a single voxel using all points in the array. O(n) complexity.

    Args:
        coords_array (NDArray[np.float64]): Array of coordinates with x, y, t values
        x_bandwidth (np.float64): Bandwidth for the x-axis
        y_bandwidth (np.float64): Bandwidth for the y-axis
        t_bandwidth (np.float64): Bandwidth for the t-axis
        x_voxel_coordinate (np.float64): Coordinate of the voxel in the x dimension
        y_voxel_coordinate (np.float64): Coordinate of the voxel in the y dimension
        t_voxel_coordinate (np.float64): Coordinate of the voxel in the t dimension
        normalization_factor (np.float64): Normalization factor for the STKDE, calculated as 1 / (n * h_x * h_y * h_t) * 0.75^3 

    Returns:
        np.float64: STKDE value for the voxel
    """
    
    # Calculate min and max values
    min_x = x_voxel_coordinate - x_bandwidth
    max_x = x_voxel_coordinate + x_bandwidth
    min_y = y_voxel_coordinate - y_bandwidth
    max_y = y_voxel_coordinate + y_bandwidth
    min_t = t_voxel_coordinate - t_bandwidth
    max_t = t_voxel_coordinate + t_bandwidth
    
    # Initialize array sum
    array_sum = 0.0
    
    # Loop through all points to only use the points within all bandwidth ranges.
    for i in range(coords_array.shape[0]):
        x, y, t = coords_array[i]
        if min_x < x < max_x and min_y < y < max_y and min_t < t < max_t:
            
            # Calculate Epanechnikov kernel
            x_epan = (1 - ((x_voxel_coordinate - x) / x_bandwidth) ** 2)
            y_epan = (1 - ((y_voxel_coordinate - y) / y_bandwidth) ** 2)
            t_epan = (1 - ((t_voxel_coordinate - t) / t_bandwidth) ** 2)

            # Multiply the kernels
            array_sum += x_epan * y_epan * t_epan

    # Calculate voxel stkde value
    stkde_value = array_sum * normalization_factor
    return stkde_value


@njit(float64(Array(float64, 2, 'C', True, aligned=True), float64, float64, float64, float64, float64, float64, float64), cache=True)
def calculate_stkde_for_voxel_weights(coords_array:NDArray[np.float64], x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64, 
                           x_voxel_coordinate:np.float64, y_voxel_coordinate:np.float64, t_voxel_coordinate:np.float64, normalization_factor:np.float64) -> np.float64:
    """Calculates weighted STKDE value, using Epanechnikov kernel for a single voxel using all points in the array. O(n) complexity.

    Args:
        coords_array (NDArray[np.float64]): Array of coordinates with x, y, t, weight values
        x_bandwidth (np.float64): Bandwidth for the x-axis
        y_bandwidth (np.float64): Bandwidth for the y-axis
        t_bandwidth (np.float64): Bandwidth for the t-axis
        x_voxel_coordinate (np.float64): Coordinate of the voxel in the x dimension
        y_voxel_coordinate (np.float64): Coordinate of the voxel in the y dimension
        t_voxel_coordinate (np.float64): Coordinate of the voxel in the t dimension
        normalization_factor (np.float64): Normalization factor for the STKDE, calculated as 1 / (neff * h_x * h_y * h_t) * 0.75^3 

    Returns:
        np.float64: Weighted STKDE value for the voxel
    """
    
    # Calculate min and max values
    min_x = x_voxel_coordinate - x_bandwidth
    max_x = x_voxel_coordinate + x_bandwidth
    min_y = y_voxel_coordinate - y_bandwidth
    max_y = y_voxel_coordinate + y_bandwidth
    min_t = t_voxel_coordinate - t_bandwidth
    max_t = t_voxel_coordinate + t_bandwidth
    
    # Initialize array sum
    array_sum = 0.0
    
    # Loop through all points to only use the points within all bandwidth ranges.
    for i in range(coords_array.shape[0]):
        x, y, t, weight = coords_array[i]
        if min_x < x < max_x and min_y < y < max_y and min_t < t < max_t:
            
            # Calculate Epanechnikov kernel
            x_epan = (1 - ((x_voxel_coordinate - x) / x_bandwidth) ** 2)
            y_epan = (1 - ((y_voxel_coordinate - y) / y_bandwidth) ** 2)
            t_epan = (1 - ((t_voxel_coordinate - t) / t_bandwidth) ** 2)

            # Multiply the kernels, and the weight
            array_sum += x_epan * y_epan * t_epan * weight

    # Calculate voxel stkde value
    stkde_value = array_sum * normalization_factor
    return stkde_value


@njit(float64(Array(float64, 2, 'C', True, aligned=True), float64, float64, float64), parallel=True, cache=True)
def _log_likelihood(existing_points:NDArray[np.float64], x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64) -> np.float64:
    """Calculates the log likelihood (comparative to predicted STKDE values) in parallel for every point in the existing_points array. See Hu et al. (2018) for more.

    Args:
        existing_points (NDArray[np.float64]): Array of existing points with x, y, t coordinates
        x_bandwidth (np.float64): Bandwidth for the x-axis
        y_bandwidth (np.float64): Bandwidth for the y-axis
        t_bandwidth (np.float64): Bandwidth for the t-axis

    Returns:
        np.float64: Log likelihood value (maximum is better)
    """

    # Get number of points
    n = np.uint32(existing_points.shape[0])
    
    # Create empty array to store log values 
    log_stkde_results = np.empty(n, dtype=np.float64)
    
    # Calculate normalization factor (with n-1)
    normalization_factor = 1 / ((n-1) * x_bandwidth * y_bandwidth * t_bandwidth) * 0.421875
    
    # Loop through all points in the existing_points array in parallel
    for index in prange(n):
        
        # Extract the point
        x_value, y_value, t_value = existing_points[index]
        
        # Calculate the STKDE value for the voxel (NOTE: This loops through all the points, and is therefore slow)
        stkde_value = calculate_stkde_for_voxel(existing_points, x_bandwidth, y_bandwidth, t_bandwidth, x_value, y_value, t_value, normalization_factor)

        # Calculate the correct STKDE value, by removing the STKDE value of the point itself, which is 1 * 1 * 1 * normalization factor
        # It is possible to remove the point from the array, but this will be far slower than just calculating it this way
        stkde_value -= normalization_factor
        
        # Add the result to the array (log is a confusing name -> returns ln)
        # NOTE: Possible optimization: ln(a) + ln(b) = ln(a*b), but this is not implemented here
        log_stkde_results[index] = log(stkde_value)
        
    # Calculate sum of natural logarithm for the entire array 
    # NOTE If atomic add was supported by Numba using it in the loop instead of looping through the array by using np.sum would likely be faster
    log_likelihood = np.sum(log_stkde_results)
        
    # Return log likelihood (maximum is better)
    return log_likelihood


@njit(float64(Array(float64, 2, 'C', True, aligned=True), float64, float64, float64), parallel=True, cache=True)
def _log_likelihood_weights(existing_points:NDArray[np.float64], x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64) -> np.float64:
    """Calculates the weighted log likelihood (comparative to predicted STKDE values) in parallel for every point in the existing_points array. See Hu et al. (2018) for more.
    Unfortunately, this function has a complexity of O(n^2), perhaps a 3d range tree would be a better approach.

    Args:
        existing_points (NDArray[np.float64]): Array of existing points with x, y, t coordinates and weights
        x_bandwidth (np.float64): Bandwidth for the x-axis
        y_bandwidth (np.float64): Bandwidth for the y-axis
        t_bandwidth (np.float64): Bandwidth for the t-axis

    Returns:
        np.float64: Weighted log likelihood value (maximum is better)
    """

    # Get number of points
    n = np.uint32(existing_points.shape[0])
    
    # Create empty array to store log values 
    log_stkde_results = np.empty(n, dtype=np.float64)
    
    # Calculate sum of weights, and sum of squared weights, to calculate neff (Kish, 1965) in the parallel loop, we replace n with neff in the normalization factor
    sum_weights = 0.0
    sum_squared_weights = 0.0
    for i in range(n):
        val = existing_points[i, 3]
        sum_weights += val
        sum_squared_weights += val ** 2
        
    # Precompute a part of the normalization factor: (0.421875)/(neff * x_bandwidth * y_bandwidth * t_bandwidth) = 1/neff * (0.421875)/(x_bandwidth * y_bandwidth * t_bandwidth)
    not_neff_normalization_factor = (0.421875)/(x_bandwidth * y_bandwidth * t_bandwidth)
    
    # Loop through all points in the existing_points array in parallel
    for index in prange(n):
        
        # Extract the point
        x_value, y_value, t_value, weight = existing_points[index]
        
        # Calculate neff for this point, by subtracting the weight of the point itself
        neff = ((sum_weights - weight)**2)/(sum_squared_weights - (weight**2))
        
        # Calculate normalization factor for this point
        normalization_factor = 1/neff * not_neff_normalization_factor
        
        # Calculate the STKDE value for the voxel (NOTE: This loops through all the points, and is therefore slow)
        stkde_value = calculate_stkde_for_voxel_weights(existing_points, x_bandwidth, y_bandwidth, t_bandwidth, x_value, y_value, t_value, normalization_factor)
            
        # Calculate the correct STKDE value, by removing the STKDE value of the point itself, which is 1 * 1 * 1 * normalization factor * point weight
        # It is possible to remove the point from the array, but this will be far slower than just calculating it this way
        stkde_value -= normalization_factor * weight
        
        # Add the result to the array (log is a confusing name -> returns ln)
        # NOTE: Possible optimization: ln(a) + ln(b) = ln(a*b), but this is not implemented here
        log_stkde_results[index] = log(stkde_value)
  
    # Calculate sum of natural logarithm for the entire array 
    # NOTE If atomic add was supported by Numba using it in the loop instead of looping through the array by using np.sum would likely be faster
    log_likelihood = np.sum(log_stkde_results)
        
    # Return log likelihood (maximum is better)
    return log_likelihood


def find_optimal_bandwidth(existing_points: NDArray[np.float64], max_x_bandwidth: np.float64, max_y_bandwidth: np.float64, max_t_bandwidth: np.float64, 
                           min_x_bandwidth: np.float64 = np.float64(1.0), min_y_bandwidth: np.float64 = np.float64(1.0), min_t_bandwidth: np.float64 = np.float64(1.0), 
                           replace_inf_value:np.float64 = np.float64(1e9), verbose:bool = False) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.uint16]:
    """Finds the optimal bandwidths for the STKDE using the Hu et al. (2018) method. The optimal bandwidths are found by maximizing the log likelihood of the STKDE.
    Optimization is performed using the L-BFGS-B algorithm. This algorithm can be done in parallel see: https://pypi.org/project/optimparallel/, but this was not done here. 

    Args:
        existing_points (NDArray[np.float64]): Array of existing points with x, y, t coordinates and possible weights
        max_x_bandwidth (np.float64): Maximum bandwidth for the x-axis, upper bound for optimization algorithm
        max_y_bandwidth (np.float64): Maximum bandwidth for the y-axis, upper bound for optimization algorithm
        max_t_bandwidth (np.float64): Maximum bandwidth for the t-axis, upper bound for optimization algorithm
        min_x_bandwidth (np.float64, optional): Minimum bandwidth for the x-axis, lower bound for the optimization algorithm. Defaults to 1.0.
        min_y_bandwidth (np.float64, optional): Minimum bandwidth for the y-axis, lower bound for the optimization algorithm. Defaults to 1.0.
        min_t_bandwidth (np.float64, optional): Minimum bandwidth for the t-axis, lower bound for the optimization algorithm. Defaults to 1.0.
        replace_inf_value (np.float64, optional): In case of -infinity returned for the log likelihood, we replace the value with with a large number so the optimization algorithm can continue. Defaults to 1e9.
        verbose (bool, optional): Print messages upon completion of optimization. Defaults to False.

    Raises:
        ValueError: If any weights are 0, infinity, negative, or NaN.
        ValueError: If existing points array has incorrect shape, must be either (n,3) or (n,4).
        ValueError: If optimization failed.

    Returns:
        Tuple[np.float64, np.float64, np.float64, np.float64, np.uint16]: Tuple of: optimal bandwidths for x, y, and t, maximum value of the log likelihood, and number of combinations tested
    """

    # Get number of points
    n = existing_points.shape[0]
    
    # Warn if n > 15000
    if n > 15000:
        warnings.warn(f"Number of points relatively high: {n}, computing optimal bandwidths is not yet optimized properly\n"
                      f" for large numbers of points. Will work just fine, but ideally a 3d range tree solution is used.", stacklevel=2)
    
    # If width of array is 4 use weights
    if np.shape(existing_points)[1] == 4:
        
        # Check if any weights are 0, infinity, negative or NaN. NOTE this is quite inefficient, but saves a headache if the user makes a mistake
        if np.any(np.isin(existing_points[:, 3], [0, np.inf, -np.inf])) or np.any(np.isnan(existing_points[:, 3])) or np.any(existing_points[:, 3] < 0):
            raise ValueError("One of the weights is invalid: either 0, infinity, negative, or NaN. Please remove or recalculate weights.")
        
        # Define function to minimize
        def minimize_function(params):
            hx, hy, ht = params
            result = -_log_likelihood_weights(existing_points=existing_points, x_bandwidth=hx, y_bandwidth=hy, t_bandwidth=ht)

            # If resulting value is -infinite, set to 1e9
            if np.isinf(result):
                result = replace_inf_value
            return result
        
    # No weights 
    elif np.shape(existing_points)[1] == 3:
        # Define function to minimize
        def minimize_function(params):
            hx, hy, ht = params
            result = -_log_likelihood(existing_points=existing_points, x_bandwidth=hx, y_bandwidth=hy, t_bandwidth=ht)

            # If resulting value is -infinite, set to 1e9
            if np.isinf(result):
                result = replace_inf_value
            return result
    else:
        raise ValueError("Existing points array has incorrect shape, must be (n,3) or (n,4) for x, y, t or x, y, t, weight respectively.")

    # Initial guess for optimal bandwidths (midpoint of min and max values)
    initial_guess = [(min_x_bandwidth + max_x_bandwidth) / 2, (min_y_bandwidth + max_y_bandwidth) / 2,
                     (min_t_bandwidth + max_t_bandwidth) / 2]

    # Bounds for bandwidths
    bounds = [(min_x_bandwidth, max_x_bandwidth), (min_y_bandwidth, max_y_bandwidth),
              (min_t_bandwidth, max_t_bandwidth)]
    
    # Minimize the negative log-likelihood
    startoptimize = time.time()
    result = minimize(minimize_function, initial_guess,
                      bounds=bounds, method="L-BFGS-B") # For explanation as to why L-BFGS-B is used, see my thesis
    
    try:
        combinations_tested = np.uint16(result.nit)
    except:
        raise ValueError("Optimization failed")

    # Print optimization time
    if verbose:
        print(f"Optimization took {time.time()-startoptimize:=.2f} seconds for {combinations_tested} combinations tested")
    
    # Retrieve the optimal bandwidths and maximum value
    optimal_hx, optimal_hy, optimal_ht = result.x
    max_value = -result.fun
    
    return optimal_hx, optimal_hy, optimal_ht, max_value, combinations_tested


@njit(UniTuple(Array(float64, 1, 'C', False, aligned=True), 3)(int64, float64, float64, int64, float64, float64, int64, float64, float64), cache=True)
def _generate_voxel_grid_constant_method(x_number_of_voxels:int, min_x_points:np.float64, max_x_points:np.float64,
                                     y_number_of_voxels:int, min_y_points:np.float64, max_y_points:np.float64,
                                     t_number_of_voxels:int, min_t_points:np.float64, max_t_points:np.float64) -> Tuple[NDArray, NDArray, NDArray]:
    """Generates three linear arrays for the x, y, and t dimensions of the voxel grid using the side method.
    Side method creates an encompassing grid around the points, with the first point being the minimum - half voxel size and the last point being the maximum + half voxel size.
    The number of voxels on each side are directly from user input.

    Args:
        x_number_of_voxels (int): Number of voxels in the x dimension.
        min_x_points (np.float64): Minimum x value of the points.
        max_x_points (np.float64): Maximum x value of the points.
        y_number_of_voxels (int): Number of voxels in the y dimension.
        min_y_points (np.float64): Minimum y value of the points.
        max_y_points (np.float64): Maximum y value of the points.
        t_number_of_voxels (int): Number of voxels in the t dimension.
        min_t_points (np.float64): Minimum t value of the points.
        max_t_points (np.float64): Maximum t value of the points.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Linearly spaced arrays for the x, y, and t dimensions of the voxel grid.
    """
    # Calculate points' lengths
    x_points_length = max_x_points - min_x_points
    y_points_length = max_y_points - min_y_points
    t_points_length = max_t_points - min_t_points
    
    # Calculate n-1 voxel sizes for each dimension
    x_voxel_size = x_points_length / (x_number_of_voxels - 1)
    y_voxel_size = y_points_length / (y_number_of_voxels - 1)
    t_voxel_size = t_points_length / (t_number_of_voxels - 1)
    
    # Half voxel size
    x_half_voxel_size = x_voxel_size / 2
    y_half_voxel_size = y_voxel_size / 2
    t_half_voxel_size = t_voxel_size / 2
    
    # First point is min - half of voxel size
    first_voxel_center_x = min_x_points - x_half_voxel_size
    first_voxel_center_y = min_y_points - y_half_voxel_size
    first_voxel_center_t = min_t_points - t_half_voxel_size
    
    # Max voxel center
    max_voxel_center_x = max_x_points + x_half_voxel_size
    max_voxel_center_y = max_y_points + y_half_voxel_size
    max_voxel_center_t = max_t_points + t_half_voxel_size
    
    # Create grid for each dimension
    xgrid = np.linspace(first_voxel_center_x, max_voxel_center_x, x_number_of_voxels)
    ygrid = np.linspace(first_voxel_center_y, max_voxel_center_y, y_number_of_voxels)
    tgrid = np.linspace(first_voxel_center_t, max_voxel_center_t, t_number_of_voxels)
    return xgrid, ygrid, tgrid


@njit(UniTuple(Array(float64, 1, 'C', False, aligned=True), 3)(float64, int64, float64, float64, float64, int64, float64, float64, float64, int64, float64, float64), cache=True)
def _generate_voxel_grid_adaptive_method(x_bandwidth:np.float64, x_voxels:int, min_x_points:np.float64, max_x_points:np.float64,
                                          y_bandwidth:np.float64, y_voxels:int, min_y_points:np.float64, max_y_points:np.float64,
                                          t_bandwidth:np.float64, t_voxels:int, min_t_points:np.float64, max_t_points:np.float64) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate three linear arrays for the x, y, and t dimensions of the voxel grid using the bandwidth method.
    Bandwidth method creates an encompassing grid around the points, the size of the grid is determined by the bandwidth and number of voxels.

    Args:
        x_bandwidth (np.float64): Bandwidth for the x-axis.
        x_voxels (int): Number of voxels per bandwidth in the x-axis.
        min_x_points (np.float64): Minimum x value of the points.
        max_x_points (np.float64): Maximum x value of the points.
        y_bandwidth (np.float64): Bandwidth for the y-axis.
        y_voxels (int): Number of voxels per bandwidth in the y-axis.
        min_y_points (np.float64): Minimum y value of the points.
        max_y_points (np.float64): Maximum y value of the points.
        t_bandwidth (np.float64): Bandwidth for the t-axis.
        t_voxels (int): Number of voxels per bandwidth in the t-axis.
        min_t_points (np.float64): Minimum t value of the points.
        max_t_points (np.float64): Maximum t value of the points.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Linearly spaced arrays for the x, y, and t dimensions of the voxel grid.
    """
    # Calculate points' lengths
    x_points_length = max_x_points - min_x_points
    y_points_length = max_y_points - min_y_points
    t_points_length = max_t_points - min_t_points
    
    # Voxel size in each dimension for bandwidth method
    x_voxel_size = x_bandwidth / x_voxels 
    y_voxel_size = y_bandwidth / y_voxels
    t_voxel_size = t_bandwidth / t_voxels

    # Precompute division of point cloud dist by voxel size (number of voxels, but it could be a float value)
    x_division = x_points_length / x_voxel_size
    y_division = y_points_length / y_voxel_size
    t_division = t_points_length / t_voxel_size

    # If the divisions are integers +2 else ceil + 1 (plus 1 accounts for the fact that the first and last voxel are missing half their width)
    if x_division == int(x_division):
        x_num_voxels = int(x_division) + 2
    else:
        x_num_voxels = ceil(x_division) + 1

    if y_division == int(y_division):
        y_num_voxels = int(y_division) + 2
    else:
        y_num_voxels = ceil(y_division) + 1

    if t_division == int(t_division):
        t_num_voxels = int(t_division) + 2
    else:
        t_num_voxels = ceil(t_division) + 1

    # Calculate sides length with voxels
    x_side_length = x_num_voxels * x_voxel_size
    y_side_length = y_num_voxels * y_voxel_size
    t_side_length = t_num_voxels * t_voxel_size

    # Calculate delta between side length and points side length
    x_side_difference = x_side_length - x_points_length
    y_side_difference = y_side_length - y_points_length
    t_side_difference = t_side_length - t_points_length

    # Calculate start of voxel bounds for grid
    x_min_grid = min_x_points - x_side_difference / 2
    y_min_grid = min_y_points - y_side_difference / 2
    t_min_grid = min_t_points - t_side_difference / 2

    # First voxel center point (half of step size to get to center of first voxel)
    first_voxel_center_x = x_min_grid + x_voxel_size / 2
    first_voxel_center_y = y_min_grid + y_voxel_size / 2
    first_voxel_center_t = t_min_grid + t_voxel_size / 2

    # Calculate end points for grid (center of final voxel)
    last_voxel_center_x = first_voxel_center_x + (x_num_voxels - 1) * x_voxel_size
    last_voxel_center_y = first_voxel_center_y + (y_num_voxels - 1) * y_voxel_size
    last_voxel_center_t = first_voxel_center_t + (t_num_voxels - 1) * t_voxel_size

    # Compute range of points (center points of voxels)
    xgrid = np.linspace(first_voxel_center_x, last_voxel_center_x, x_num_voxels)
    ygrid = np.linspace(first_voxel_center_y, last_voxel_center_y, y_num_voxels)
    tgrid = np.linspace(first_voxel_center_t, last_voxel_center_t, t_num_voxels)
    return xgrid, ygrid, tgrid


@njit(Array(float64, 1, 'C', True, True)(Array(float64, 2, 'C', True, True), float64, float64, float64,
                                                          Array(float64, 1, 'C', True, True),
                                                          Array(float64, 1, 'C', True, True),
                                                          Array(float64, 1, 'C', True, True),
                                                          boolean), parallel=True, cache=True)
def calculate_stkde_for_voxel_grid(coords_array: NDArray[np.float64], x_bandwidth: np.float64, y_bandwidth: np.float64, t_bandwidth: np.float64,
                                     x_voxel_centers_array: NDArray[np.float64], y_voxel_centers_array: NDArray[np.float64], t_voxel_centers_array: NDArray[np.float64],
                                     return_ijk_order: bool) -> NDArray[np.float64]:
    """Calculates the Spatio-Temporal Kernel Density Estimation (STKDE), using the Hu, et al. (2018) method.
    Uses custom very fast algorithm -> STOPKDE (Spatio-Temporal One-Pass Kernel Density Estimation).
    Algorithm is not implemented in parallel in this version, but has the potential to scale very efficiently across multiple threads.
    See: https://github.com/numba/numba/issues/2988, for updates on CPU atomics support in Numba (required for parallel implementation).
    Args:
        coords_array (NDArray[np.float64]): Array of combined coordinates (x, y, t) where x, y, and t are the coordinates of the points
        x_bandwidth (np.float64): Bandwidth for the x-axis
        y_bandwidth (np.float64): Bandwidth for the y-axis
        t_bandwidth (np.float64): Bandwidth for the t-axis
        x_voxel_centers_array (NDArray[np.float64]): Array of x voxel centers, created using np.linspace
        y_voxel_centers_array (NDArray[np.float64]): Array of y voxel centers, created using np.linspace
        t_voxel_centers_array (NDArray[np.float64]): Array of t voxel centers, created using np.linspace.
        return_ijk_order (bool): If True, returns the STKDE values in 'ijk' order, if False returns in 'kji' order, NOTE: kji order returns slower.

    Raises:
        ValueError: If number of columns in coords_array is not equal to 3

    Returns:
        NDArray[np.float64]: Flattened array of STKDE values for each voxel. Returns in 'ijk' order or 'kji' order.
    """
    # If number of columns does not equal 3 raise an error
    if coords_array.shape[1] != 3:
        raise ValueError(
            "Input array should have 3 columns for x, y and t coordinates")

    # Extract the number of points
    num_points = coords_array.shape[0]

    # Extract the number of layers
    number_of_x_layers = x_voxel_centers_array.shape[0]
    number_of_y_layers = y_voxel_centers_array.shape[0]
    number_of_t_layers = t_voxel_centers_array.shape[0]

    # Extract the minimum values
    x_minimum_value = x_voxel_centers_array[0]
    y_minimum_value = y_voxel_centers_array[0]
    t_minimum_value = t_voxel_centers_array[0]

    # Extract the distances, or the distance equal to an index 'i' in each direction
    x_distance = x_voxel_centers_array[1] - x_minimum_value
    y_distance = y_voxel_centers_array[1] - y_minimum_value
    t_distance = t_voxel_centers_array[1] - t_minimum_value

    # Express bandwidths in terms of index
    x_bandwidth_i = x_bandwidth / x_distance
    y_bandwidth_i = y_bandwidth / y_distance
    t_bandwidth_i = t_bandwidth / t_distance
    neg_x_bandwidth_i = -x_bandwidth_i
    neg_y_bandwidth_i = -y_bandwidth_i
    neg_t_bandwidth_i = -t_bandwidth_i

    # Precompute reciprocal distances between voxels, multiplication is slightly faster than division in the loop
    x_distance_recip = 1 / x_distance
    y_distance_recip = 1 / y_distance
    t_distance_recip = 1 / t_distance

    # Create a stkde values 3d array. We fill this array with zeroes so it is possible to add to the array.
    stkde_values = np.zeros((number_of_x_layers, number_of_y_layers, number_of_t_layers), dtype=np.float64)

    # Normalization factor 1 / (n * x_bandwidth * y_bandwidth * t_bandwidth), * 0.75^3 (* once for each kernel)
    normalization_factor = 1 / (num_points * x_bandwidth * y_bandwidth * t_bandwidth) * 0.421875

    # Loop through all the points, and discover which voxels are in range per point.
    # For each unique coordinate we calculate the Epanechnikov value and add it to the voxel.
    # This loop is not in parallel as Numba does not support atomic adds to a 3d array in njit mode, it does in CUDA mode, and other languages do as well.
    # Doing this in parallel is very close to an apparent optimal solution to this version of STKDE.
    # One could also consider varying the order of the internal loops based on the voxel sizes and the bandwidths, but this is not implemented here.
    for i in range(num_points):

        # Extract x,y value of the current point
        x_value, y_value, t_value = coords_array[i]

        # Convert value to be expressed in terms of indexes
        x_val_i = (x_value - x_minimum_value) * x_distance_recip
        y_val_i = (y_value - y_minimum_value) * y_distance_recip
        t_val_i = (t_value - t_minimum_value) * t_distance_recip

        # Determine which voxels a point influences, done by converting the point coordinates to voxel coordinate indexes.
        # This code allows for points to be outside of the grid, this should generally not happen, but in case it does this code should handle appropriately.
        # For any CUDA implementation the branching should be removed entirely, as this is not GPU appropiate.
        # NOTE: Possible: calculate ub without the slower ceil by adding a precomp. integer. Manual testing of ceil: 100k = 2ms, so probably not worth it.
        
        # Check if x_value is inside of normal bounds index > 0. We do not inline here to avoid unpacking overhead.
        if x_val_i > 0:  # Normal point logic
            # Lower bound, max to avoid negative values
            lb_x = max(ceil(x_val_i - x_bandwidth_i), 0)
            # Upper bound, rounded up because of exclusive indexing
            ub_x = min(ceil(x_val_i + x_bandwidth_i), number_of_x_layers)
        elif x_val_i <= neg_x_bandwidth_i:  # Point is further away than a bandwidth from smallest voxel value, return no voxels
            continue  # Skip the rest of the loop, as no voxels are influenced
        else:  # Remaining negative i values (0 > x_val_i > -x_bandwidth_i)
            lb_x = 0  # Lower bound is always the smallest voxel value
            # Upper bound is calculated as normal
            ub_x = min(ceil(x_val_i + x_bandwidth_i), number_of_x_layers)

        # Repeat of the above logic for y
        if y_val_i > 0:
            lb_y = max(ceil(y_val_i - y_bandwidth_i), 0)
            ub_y = min(ceil(y_val_i + y_bandwidth_i), number_of_y_layers)
        elif y_val_i <= neg_y_bandwidth_i:
            continue
        else:
            lb_y = 0
            ub_y = min(ceil(y_val_i + y_bandwidth_i), number_of_y_layers)

        # Repeat of the above logic for t
        if t_val_i > 0:
            lb_t = max(ceil(t_val_i - t_bandwidth_i), 0)
            ub_t = min(ceil(t_val_i + t_bandwidth_i), number_of_t_layers)
        elif t_val_i <= neg_t_bandwidth_i:
            continue
        else:
            lb_t = 0
            ub_t = min(ceil(t_val_i + t_bandwidth_i), number_of_t_layers)

        # Precompute Epanechnikov kernel values per dimension, skip the first dimension as we calculate it in the nested loops later
        # Possible TODO: Reciprocal could be used instead of dividing by bandwidth, will lead to some floating point drift though
        # Second possible TODO: A custom ordering of the loops would presumably be faster, but this is not implemented here.
        y_epanechnikov_values = np.empty(ub_y - lb_y, dtype=np.float64)
        for j, y_voxel_value in enumerate(y_voxel_centers_array[lb_y:ub_y]):
            y_epanechnikov_values[j] = 1 - ((y_voxel_value - y_value) / y_bandwidth)**2

        t_epanechnikov_values = np.empty(ub_t - lb_t, dtype=np.float64)
        for j, t_voxel_value in enumerate(t_voxel_centers_array[lb_t:ub_t]):
            t_epanechnikov_values[j] = 1 - ((t_voxel_value - t_value) / t_bandwidth)**2

        # Calculate cartesian product of the Epanechnikov kernels for this point and sum with the values in the stkde values array
        for j, x_voxel_value in enumerate(x_voxel_centers_array[lb_x:ub_x]):
            x_index = lb_x + j
            x_epanechnikov_value = 1 - ((x_voxel_value - x_value) / x_bandwidth)**2
            for k in range(ub_y - lb_y):
                xy_epanechnikov_value = x_epanechnikov_value * y_epanechnikov_values[k]
                y_index = lb_y + k
                for l in range(ub_t - lb_t):
                    # Add the combined Epanechnikov value to the appropiate voxels
                    stkde_values[x_index, y_index, lb_t + l] += xy_epanechnikov_value * t_epanechnikov_values[l]

    # Flatten, multiply with normalization factor and return. O(number of voxels) time complexity
    # This was tested to be the fastest way to flatten the array, even for 125 million voxels this takes at worst 0.2 seconds.
    # It is possible to have the array be flattenend from the beginning, but this makes accessing the right voxel indexes harder, code less readable, and CUDA impossible.
    # Additionally, we need to loop through all values to multiply with the normalization factor, so we might as well flatten at that point.
    
    # NOTE: For kji order the return is a lot slower than for ijk order. This is because of the way the array is accessed in memory.
    # This could be fixed, but would require rewriting the entire function above, and would make ijk slower as a result. 
    if return_ijk_order:
        number_of_voxels_per_x_layer = number_of_y_layers * number_of_t_layers
        stkde_grid = np.empty(number_of_x_layers * number_of_voxels_per_x_layer, dtype=np.float64)
        for i in prange(number_of_x_layers):
            inner_loop_index = i * number_of_voxels_per_x_layer
            for j in range(number_of_y_layers):
                for k in range(number_of_t_layers):
                    stkde_grid[inner_loop_index] = stkde_values[i, j, k] * normalization_factor
                    inner_loop_index += 1
                    
    else: # Output as kji order
        number_of_voxels_per_t_layer = number_of_y_layers * number_of_x_layers
        stkde_grid = np.empty(number_of_t_layers * number_of_voxels_per_t_layer, dtype=np.float64)
        for k in prange(number_of_t_layers):
            inner_loop_index = k * number_of_voxels_per_t_layer
            for j in range(number_of_y_layers):
                for i in range(number_of_x_layers):
                    stkde_grid[inner_loop_index] = stkde_values[i, j, k] * normalization_factor
                    inner_loop_index += 1

    return stkde_grid


@njit(Array(float64, 1, 'C', True, True)(Array(float64, 2, 'C', True, True), float64, float64, float64,
                                         Array(float64, 1, 'C', True, True),
                                         Array(float64, 1, 'C', True, True),
                                         Array(float64, 1, 'C', True, True),
                                         boolean), parallel=True, cache=True)
def calculate_stkde_for_voxel_grid_weighted(coords_array: NDArray[np.float64], x_bandwidth: np.float64, y_bandwidth: np.float64, t_bandwidth: np.float64,
                                     x_voxel_centers_array: NDArray[np.float64], y_voxel_centers_array: NDArray[np.float64], t_voxel_centers_array: NDArray[np.float64],
                                     return_ijk_order: bool) -> NDArray[np.float64]:
    """Calculates the Spatio-Temporal Kernel Density Estimation (STKDE), using the Hu, et al. (2018) method. Allows for weighted points.
    Uses custom very fast algorithm -> STOPKDE (Spatio-Temporal One-Pass Kernel Density Estimation).
    Algorithm is not implemented in parallel in this version, but has the potential to scale very efficiently across multiple threads.
    See: https://github.com/numba/numba/issues/2988, for updates on CPU atomics support in Numba (required for parallel implementation).
    Args:
        coords_array (NDArray[np.float64]): Array of combined coordinates (x, y, t, w) where x, y, t are the coordinates of the points, and w is the weight
        x_bandwidth (np.float64): Bandwidth for the x-axis
        y_bandwidth (np.float64): Bandwidth for the y-axis
        t_bandwidth (np.float64): Bandwidth for the t-axis
        x_voxel_centers_array (NDArray[np.float64]): Array of x voxel centers, created using np.linspace
        y_voxel_centers_array (NDArray[np.float64]): Array of y voxel centers, created using np.linspace
        t_voxel_centers_array (NDArray[np.float64]): Array of t voxel centers, created using np.linspace
        return_ijk_order (bool): If True, returns the STKDE values in 'ijk' order, if False returns in 'kji' order, NOTE: kji order returns slower.

    Raises:
        ValueError: If number of columns in coords_array is not equal to 4
        ValueError: If the normalization factor is not greater than 0, or infinite, this indicates that a weight in the input is zero, NaN, or negative.
        
    Returns:
        NDArray[np.float64]: Flattened array of STKDE values for each voxel. Returns in 'ijk' order or 'kji' order.
    """
    # If number of columns does not equal 4 raise an error
    if coords_array.shape[1] != 4:
        raise ValueError(
            "Input array should have 4 columns for x, y, t coordinates and weights")

    # Extract the number of points
    num_points = coords_array.shape[0]

    # Extract the number of layers
    number_of_x_layers = x_voxel_centers_array.shape[0]
    number_of_y_layers = y_voxel_centers_array.shape[0]
    number_of_t_layers = t_voxel_centers_array.shape[0]

    # Extract the minimum values
    x_minimum_value = x_voxel_centers_array[0]
    y_minimum_value = y_voxel_centers_array[0]
    t_minimum_value = t_voxel_centers_array[0]

    # Extract the distances, or the distance equal to an index 'i' in each direction
    x_distance = x_voxel_centers_array[1] - x_minimum_value
    y_distance = y_voxel_centers_array[1] - y_minimum_value
    t_distance = t_voxel_centers_array[1] - t_minimum_value

    # Express bandwidths in terms of index
    x_bandwidth_i = x_bandwidth / x_distance
    y_bandwidth_i = y_bandwidth / y_distance
    t_bandwidth_i = t_bandwidth / t_distance
    neg_x_bandwidth_i = -x_bandwidth_i
    neg_y_bandwidth_i = -y_bandwidth_i
    neg_t_bandwidth_i = -t_bandwidth_i

    # Precompute reciprocal distances between voxels, multiplication is slightly faster than division in the loop
    x_distance_recip = 1 / x_distance
    y_distance_recip = 1 / y_distance
    t_distance_recip = 1 / t_distance

    # Create a stkde values 3d array. We fill this array with zeroes so it is possible to add to the array.
    stkde_values = np.zeros((number_of_x_layers, number_of_y_layers, number_of_t_layers), dtype=np.float64)

    # Initialize variables for sum of weights and sum of squared weights. 
    # Used for calculating normalization factor, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    sum_weights = 0.0
    sum_squared_weights = 0.0

    # Loop, only differences with non-weighted version are commented. 
    for i in range(num_points):
        
        # Extract weight, and add to sum of weights and sum of squared weights
        x_value, y_value, t_value, weight = coords_array[i]
        sum_weights += weight
        sum_squared_weights += weight**2

        x_val_i = (x_value - x_minimum_value) * x_distance_recip
        y_val_i = (y_value - y_minimum_value) * y_distance_recip
        t_val_i = (t_value - t_minimum_value) * t_distance_recip

        if x_val_i > 0:
            lb_x = max(ceil(x_val_i - x_bandwidth_i), 0)
            ub_x = min(ceil(x_val_i + x_bandwidth_i), number_of_x_layers)
        elif x_val_i <= neg_x_bandwidth_i:
            continue
        else:
            lb_x = 0
            ub_x = min(ceil(x_val_i + x_bandwidth_i), number_of_x_layers)

        if y_val_i > 0:
            lb_y = max(ceil(y_val_i - y_bandwidth_i), 0)
            ub_y = min(ceil(y_val_i + y_bandwidth_i), number_of_y_layers)
        elif y_val_i <= neg_y_bandwidth_i:
            continue
        else:
            lb_y = 0
            ub_y = min(ceil(y_val_i + y_bandwidth_i), number_of_y_layers)

        if t_val_i > 0:
            lb_t = max(ceil(t_val_i - t_bandwidth_i), 0)
            ub_t = min(ceil(t_val_i + t_bandwidth_i), number_of_t_layers)
        elif t_val_i <= neg_t_bandwidth_i:
            continue
        else:
            lb_t = 0
            ub_t = min(ceil(t_val_i + t_bandwidth_i), number_of_t_layers)

        y_epanechnikov_values = np.empty(ub_y - lb_y, dtype=np.float64)
        for j, y_voxel_value in enumerate(y_voxel_centers_array[lb_y:ub_y]):
            y_epanechnikov_values[j] = 1 - ((y_voxel_value - y_value) / y_bandwidth)**2

        t_epanechnikov_values = np.empty(ub_t - lb_t, dtype=np.float64)
        for j, t_voxel_value in enumerate(t_voxel_centers_array[lb_t:ub_t]):
            t_epanechnikov_values[j] = 1 - ((t_voxel_value - t_value) / t_bandwidth)**2

        for j, x_voxel_value in enumerate(x_voxel_centers_array[lb_x:ub_x]):
            x_index = lb_x + j
            # Add multiplication with weight to x Epanechnikov value
            x_epanechnikov_value = (1 - ((x_voxel_value - x_value) / x_bandwidth)**2) * weight
            for k in range(ub_y - lb_y):
                xy_epanechnikov_value = x_epanechnikov_value * y_epanechnikov_values[k]
                y_index = lb_y + k
                for l in range(ub_t - lb_t):
                    # Add the combined Epanechnikov value to the appropiate voxels
                    stkde_values[x_index, y_index, lb_t + l] += xy_epanechnikov_value * t_epanechnikov_values[l]

    # Normalization factor calculation, replace n with n effective
    neff = (sum_weights**2) / sum_squared_weights
    normalization_factor = 1 / (neff * x_bandwidth * y_bandwidth * t_bandwidth) * 0.421875

    # Raise a value error if normalization factor is 0, NaN, negative or infinite
    if not normalization_factor > 0 or not np.isfinite(normalization_factor):
        raise ValueError(
            "Please check input weights, it is likely that one of the weights is 0, NaN, or infinite")

    if return_ijk_order:
        number_of_voxels_per_x_layer = number_of_y_layers * number_of_t_layers
        stkde_grid = np.empty(number_of_x_layers * number_of_voxels_per_x_layer, dtype=np.float64)
        for i in prange(number_of_x_layers):
            inner_loop_index = i * number_of_voxels_per_x_layer
            for j in range(number_of_y_layers):
                for k in range(number_of_t_layers):
                    stkde_grid[inner_loop_index] = stkde_values[i, j, k] * normalization_factor
                    inner_loop_index += 1
                    
    else:
        number_of_voxels_per_t_layer = number_of_y_layers * number_of_x_layers
        stkde_grid = np.empty(number_of_t_layers * number_of_voxels_per_t_layer, dtype=np.float64)
        for k in prange(number_of_t_layers):
            inner_loop_index = k * number_of_voxels_per_t_layer
            for j in range(number_of_y_layers):
                for i in range(number_of_x_layers):
                    stkde_grid[inner_loop_index] = stkde_values[i, j, k] * normalization_factor
                    inner_loop_index += 1

    return stkde_grid


def write_stkde_to_vtk(stkde_arr:NDArray[np.float64], xgrid:NDArray[np.float64], ygrid:NDArray[np.float64], tgrid:NDArray[np.float64], output_file_name:str|Path) -> None:
    """Write STKDE values to a .vtk ImageData file.

    Args:
        stkde_arr (NDArray[np.float64]): Array containing the STKDE values for each voxel, requires input to be in 'kji' order. See: https://vtk.org/doc/nightly/html/classvtkStructuredGrid.html
        xgrid (NDArray[np.float64]): Array containing the x coordinates of the voxel grid.
        ygrid (NDArray[np.float64]): Array containing the y coordinates of the voxel grid.
        tgrid (NDArray[np.float64]): Array containing the t coordinates of the voxel grid.
        output_file_name (str | Path): Output file name/location for the .vtk file.
    """
    # Create grid
    grid = pv.ImageData()
    
    # Minimum values
    min_x, min_y, min_t = xgrid[0], ygrid[0], tgrid[0]
    
    # Get distance between voxels 
    x_distance = xgrid[1] - min_x
    y_distance = ygrid[1] - min_y
    t_distance = tgrid[1] - min_t
    
    # Number of voxels
    x_num_voxels = xgrid.shape[0]
    y_num_voxels = ygrid.shape[0]
    t_num_voxels = tgrid.shape[0]
    
    # Set the grid dimensions (+1 because the grid.dimensions are the corners of the voxels, this is the same as the number of voxels + 1)
    grid.dimensions = (x_num_voxels + 1, y_num_voxels + 1, t_num_voxels + 1)
    
    # Set the grid origin and spacing
    grid.origin = (min_x - x_distance / 2, min_y - y_distance / 2, min_t - t_distance / 2)
    grid.spacing = (x_distance, y_distance, t_distance)
    # Input should be in 'kji' order, there is no way to check this, so please be aware of this when using this function.
    grid.cell_data["stkde_values"] = stkde_arr
    
    grid.save(Path(output_file_name))


def stkde(gdf: gpd.GeoDataFrame, time_col: str, crs: str, number_of_voxels: Tuple[int, int, int], voxel_method: Literal['constant', 'adaptive'] = 'constant', 
    bandwidths: Optional[Tuple[float, float, float]] = None, calculate_optimal_bandwidths: bool = False, weight_col: Optional[str] = None, 
    output_file: Optional[str|Path] = None, verbose: bool = True) -> NDArray[np.float64]:
    """
    Calculates the Spatio-Temporal Kernel Density Estimation (STKDE) using the method by Hu, et al. (2018) for a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the points to calculate the STKDE.
        time_col (str): Name of the column in the GeoDataFrame that contains time values.
        crs (str): Coordinate Reference System (CRS) for the output STKDE grid. Reprojects the GeoDataFrame if necessary.
        number_of_voxels (Tuple[int, int, int]): Number of voxels for each axis (x, y, t), or per bandwidth depending on the 'voxel_method'.
        voxel_method (Literal['constant', 'adaptive'], optional): Method to calculate the voxel grid. Choose between a fixed number of voxels 
            ('constant') or a bandwidth-dependent number of voxels ('adaptive'). Defaults to 'constant'.
        bandwidths ([Tuple[float, float, float]], optional): Bandwidths for the STKDE in the x, y, and t dimensions (hx, hy, ht). 
            If not provided and 'calculate_optimal_bandwidths' is False, raises a ValueError. Defaults to None.
        calculate_optimal_bandwidths (bool, optional): Whether to calculate optimal bandwidths. If True, bandwidths will be estimated. 
            Defaults to False.
        weight_col (Optional[str], optional): Name of the column in the GeoDataFrame that contains weights for each point. 
            If provided a weighted STKDE will be computed. Defaults to None.
        output_file (Optional[str|Path], optional): Path to save the output VTK file. If None, the output will not be saved. Defaults to None.
        verbose (bool, optional): If True, the function will print progress updates during the computation. Defaults to True.

    Raises:
        ValueError: If bandwidths are not provided and 'calculate_optimal_bandwidths' is False.
        ValueError: If bandwidths are not in the correct format or contain non-positive values.
        ValueError: If the GeoDataFrame is empty or does not have a valid CRS.
        KeyError: If the time column or a specified weight column is not found in the GeoDataFrame.
        AssertionError: If there are negative, 0, or NaN values in the weight column.
        ValueError: If the 'number_of_voxels' parameter is incorrectly formatted or contains non-positive values.
        ValueError: If CRS conversion fails.
        TypeError: If the time column datatype is not numeric.
        AssertionError: If there are invalid geometries or missing time column values in the input GeoDataFrame.
        TypeError: If the output_file is not a string or Path object.

    Returns:
        NDArray[np.float64]: 1D array of STKDE values, representing the density estimate for each voxel in the flattened voxel grid. 
        The array is sorted in 'ijk' order, where 'i' is the x-axis, 'j' is the y-axis, and 'k' is the t-axis.
    """
    # Validate inputs ---------------------------------------------------------------------------------------------------------------------
    # Check bandwidths based on calculate_optimal_bandwidths flag
    if calculate_optimal_bandwidths:
        if bandwidths:
            warnings.warn("Bandwidths provided will be ignored as calculate_optimal_bandwidths is set to True.")
    else:
        if not bandwidths:
            raise ValueError("Bandwidths must be provided if calculate_optimal_bandwidths is set to False.")
        
        # Check if bandwidths is a tuple of length 3 and contains positive values
        if len(bandwidths) != 3:
            raise ValueError("Bandwidths must be provided in the format (x_bandwidth, y_bandwidth, t_bandwidth).")
        if not all(b > 0 for b in bandwidths):
            raise ValueError("All bandwidths must be positive values.")
        
        # Split bandwidths and convert to float64
        x_bandwidth, y_bandwidth, t_bandwidth = map(np.float64, bandwidths)
    
    # Validate GeoDataFrame (gdf)
    if gdf is None or len(gdf) == 0:
        raise ValueError("GeoDataFrame is empty.")
    if gdf.crs is None:
        raise ValueError("GeoDataFrame does not have a valid CRS.")
    if len(gdf) < 100:
        warnings.warn("GeoDataFrame has less than 100 points, results may not be reliable.")
        
    # Check if time_col exists in gdf
    if time_col not in gdf.columns:
        raise KeyError(f"Time column {time_col} not found in GeoDataFrame.")
    
    # Check if weight_col exists in gdf if provided and set weights_present accordingly
    if weight_col:
        if weight_col not in gdf.columns:
            raise KeyError(f"Weight column {weight_col} not found in GeoDataFrame.")
        # Check if all weights are valid (not negative, 0, or NaN)
        assert not (gdf[weight_col].le(0) | gdf[weight_col].isna()).any(), \
            f"There are values that are negative, 0, or NaN in the weight column"
        weights_present = np.bool_(True)
    else:
        weights_present = np.bool_(False)
    
    # Validate number_of_voxels
    if len(number_of_voxels) != 3:
        raise ValueError("Number of voxels must be provided in the format (x_voxels, y_voxels, t_voxels).")
    if not all(v > 0 for v in number_of_voxels):
        raise ValueError("Numbers of voxels must be positive.")
    x_voxels, y_voxels, t_voxels = number_of_voxels
    
    # Reproject the GeoDataFrame if necessary
    if gdf.crs != crs:
        try:
            gdf = gdf.to_crs(crs)  # type: ignore
        except Exception as e:
            raise ValueError(f"CRS conversion failed. Please check format of supplied CRS: {crs}. Error: {e}")
    
    # Check if time_col is numeric
    if not pd.api.types.is_numeric_dtype(gdf[time_col]):
        raise TypeError("Time column must be of numeric type. Please convert to float or int.")
    
    # Check if any geometry or value in the time column is empty
    assert not (gdf['geometry'].is_empty.any() or gdf[time_col].isna().any()), \
        "Invalid geometries or missing time column values found in the input GeoDataFrame"

    # Generate the output file path if required
    if output_file:
        # Ensure output_file is a Path object if it's a string
        if isinstance(output_file, str):
            output_file = Path(output_file)
        else:
            raise TypeError("Output file must be a string or Path object.")
            
        # Validate or adjust the file suffix
        if output_file.suffix not in ['.vtk', '.vti']:
            output_file = output_file.with_suffix('.vti')  # Default to .vti if no valid suffix is found

        # Create the full output path relative to the current working directory
        output_file_path = Path.cwd() / output_file
    
    # Start functionality ---------------------------------------------------------------------------------------------------------------------
    if verbose: 
        start_time = time.time()

    # Extract all gdf values to numpy arrays
    x_coords_array = gdf.geometry.x.values.astype('float64')
    y_coords_array = gdf.geometry.y.values.astype('float64')
    time_coords_array = gdf[time_col].values.astype('float64')
    
    # Combine coordinates into a single array using numpy.column_stack
    if weights_present == True:
        weights_array = gdf[weight_col].values
        assert isinstance(weights_array, np.ndarray)
        combined_coords_array = np.column_stack(
            (x_coords_array, y_coords_array, time_coords_array, weights_array))
    else:
        combined_coords_array = np.column_stack(
            (x_coords_array, y_coords_array, time_coords_array))

    # Print progress (skipped if verbose is False)
    if verbose:
        time_1 = time.time()
        print(
            "(1/5) Completed preprocessing in {:.2f} seconds.".format(time_1 - start_time))

    # Step 1: Compute optimal bandwidths  ---------------------------------------------------------------------------------------------------------------------
    
    # Extract min and max values for each dimension from combined_coords_array
    min_x_points, max_x_points, min_y_points, max_y_points, min_t_points, max_t_points = _find_extrema_xyt(combined_coords_array)

    # If calculate_optimal_bandwidths is True, compute optimal bandwidths
    match calculate_optimal_bandwidths:
        case True:
            # Compute min and max bandwidth sizes to check for optimal widths
            max_x_bandwidth = (max_x_points - min_x_points) / 2
            max_y_bandwidth = (max_y_points - min_y_points) / 2
            max_t_bandwidth = (max_t_points - min_t_points) / 2

            # Run the function to find the optimal bandwidths (this function checks for weights automatically based on the dims of combined_coords_array)
            x_bandwidth, y_bandwidth, t_bandwidth, max_log, combinations_tested = find_optimal_bandwidth(combined_coords_array, max_x_bandwidth, max_y_bandwidth, max_t_bandwidth, verbose=verbose)
            
            if combinations_tested == 0:
                warnings.warn("No combinations were tested in the optimisation algorithm.\n Either check the input data for invalid values in geometry or time column,\
or check if the data is very spread out, which can cause optimal bandwidths to be larger than half the grid size which is a hardcoded limitation currently.")
        
        case False:
            combinations_tested = "no"  # Set to no if no bandwidths are tested

    # Print progress (skipped if verbose is False)
    if verbose:
        time_2 = time.time()
        print("(2/5) Completed computing bandwidths in {:.2f} seconds. Attempted {} bandwidth combinations".format(
            time_2 - time_1, combinations_tested))

    # Step 2: Create voxel grid  -----------------------------------------------------------------------------------------------------------------------------
    match voxel_method: 
        case "constant": # Default method for creating voxel grid, uses the number of voxels per side as gotten directly from user
            xgrid, ygrid, tgrid = _generate_voxel_grid_constant_method(x_voxels, min_x_points, max_x_points, y_voxels, min_y_points, max_y_points, t_voxels, min_t_points, max_t_points)
            
        case "adaptive": # Optional method for creating voxel grid, calculates the number of voxels relative to bandwidths (finer control but much more difficult to use properly)
            xgrid, ygrid, tgrid = _generate_voxel_grid_adaptive_method(x_bandwidth, x_voxels, min_x_points, max_x_points, 
                                                                        y_bandwidth, y_voxels, min_y_points, max_y_points, 
                                                                        t_bandwidth, t_voxels, min_t_points, max_t_points)

    # Print progress (skipped if verbose is False)
    if verbose:
        time_3 = time.time()
        print(
            "(3/5) Completed grid computation in {:.2f} seconds.".format(time_3 - time_2))
    
    # Step 3: Calculate STKDE for each voxel ---------------------------------------------------------------------------------------------------------------------
    # Return in kji order as the vtk file format expects this order
    if weights_present:
        stkde_values = calculate_stkde_for_voxel_grid_weighted(combined_coords_array, x_bandwidth, y_bandwidth, t_bandwidth, xgrid, ygrid, tgrid, False)
    else:
        stkde_values = calculate_stkde_for_voxel_grid(combined_coords_array, x_bandwidth, y_bandwidth, t_bandwidth, xgrid, ygrid, tgrid, False)
    
    # Print progress (skipped if verbose is False)
    if verbose:
        time_4 = time.time()
        print(
            "(4/5) Completed STKDE calculations in {:.2f} seconds.".format(time_4 - time_3))

    # Step 4: Store results ----------------------------------------------------------------------
    if output_file:
        # Write STKDE values to a .vtk file
        write_stkde_to_vtk(stkde_values, xgrid, ygrid, tgrid, output_file_path)

    # Print output information (only if verbose is True)
    if verbose:
        # Calculate distances between voxels to print
        x_distance = xgrid[1] - xgrid[0]
        y_distance = ygrid[1] - ygrid[0]
        t_distance = tgrid[1] - tgrid[0]
        
        compute_time = time.time() - start_time
        output_message = "(5/5) STKDE result has been saved to" if output_file else "(5/5) STKDE result has been returned."
        print(f"""{output_message} {output_file_path if output_file else ""}
Total compute time: {compute_time:.2f} seconds.
Total number of voxels calculated: {stkde_values.shape[0]}.
Voxel size (x:{x_distance:.2f}, y:{y_distance:.2f}, t:{t_distance:.2f}), using bandwidths: x={x_bandwidth}, y={y_bandwidth}, t={t_bandwidth}.""")

    # Return stkde_values 
    return stkde_values


if __name__ == "__main__":
    
    # Older versions of STKDE function provided as validation, additionally example use of calculate_stkde_for_voxel_grid is shown
    # Please note, these version are not fully optimized and are not recommended for use in production environments
    
    user_input = input("Do you want to run tests? Please be aware this will take upwards of 30s as multiple functions will be compiled (y/n): ").strip().lower()
    if user_input == 'y':
        # Run tests
        @njit(Array(float64, 1, 'C', True, aligned=True)(Array(float64, 2, 'C', True, aligned=True), Array(float64, 2, 'C', True, aligned=True),
                                                          (float64), (float64), (float64), (uint32), (boolean)), parallel=True)
        def calculate_stkde_for_voxel_grid_sorting(flattened_voxel_grid: np.ndarray, combined_coords_array: np.ndarray,
                                                   x_bandwidth: np.float64, y_bandwidth: np.float64, t_bandwidth: np.float64,
                                                   n: np.uint32, weights_present: np.bool_) -> np.ndarray:
            """Calculates the Spatio-Temporal Kernel Density Estimation (STKDE), using the Hu, et al. (2018) method for each voxel in a flattened voxel grid using a custom sliding window method. 
            Also implements a custom weighted method for the STKDE calculation.

            Args:
                flattened_voxel_grid (np.ndarray): Sorted grid of voxels in a flattened format (x, y, t) where x, y, and t are the coordinates of the voxel, please note: the grid should be sorted by the t-coordinate
                combined_coords_array (np.ndarray): Array of combined coordinates (x, y, t, (optional:w)) where x, y, and t are the coordinates of the point and w is the weight of the point
                x_bandwidth (float): Bandwidth for the x-axis
                y_bandwidth (float): Bandwidth for the y-axis
                t_bandwidth (float): Bandwidth for the t-axis
                n (int): Number of points in the combined_coords_array
                weights_present (bool): Boolean indicating if weights are present in the combined_coords_array

            Raises:
                ValueError: If either of the input arrays are not sorted (randomly sampled with 100 sequential points)
                ValueError: If the grid does not have equally sized temporal layers
                ValueError: If weights are present but the combined_coords_array does not have 4 columns

            Returns:
                np.ndarray: Array of STKDE values for each voxel in the flattened voxel grid, with the coordinates of the voxel and the STKDE value (x, y, t, stkde)
            """
            # Total number of voxels
            num_voxels = flattened_voxel_grid.shape[0]

            # Total number of points
            num_points = combined_coords_array.shape[0]

            # Raise error if the input arrays are not sorted
            # Sample the third column of the voxel grid with 100 random sequential points
            sample = flattened_voxel_grid[np.linspace(0, num_voxels-1, 100).astype(np.int32), 2]
            # Check if the sampled array is sorted
            is_sorted = np.all(sample[:-1] <= sample[1:])
            if not is_sorted:
                raise ValueError("The input array (voxel grid) is not sorted. Please sort it using np.argsort on the 't' column before using this function.")

            # Sample the third column of the points array with 100 random sequential points
            sample = combined_coords_array[np.linspace(0, num_points-1, 100).astype(np.int32), 2]
            # Check if the sampled array is sorted
            is_sorted = np.all(sample[:-1] <= sample[1:])
            if not is_sorted:
                raise ValueError("The input array (combined coordinates) is not sorted. Please sort it using np.argsort on the 't' column before using this function.")
            
            # Create empty numpy array to store STKDE values, set data type to float64 (faster)
            stkde_values = np.zeros(num_voxels, dtype=np.float64)

            # Lowest t value (remember array is sorted so the first value is the lowest possible)
            lowest_t = flattened_voxel_grid[0, 2]
            # Find the last index where the lowest_t can be placed while keeping the array sorted (i.e. find the index of the first voxel in a different layer)
            num_voxels_per_layer = np.searchsorted(flattened_voxel_grid[:,2], lowest_t, side='right')
            
            # Extract info about temporal layers
            temporal_layers_centers = flattened_voxel_grid[::num_voxels_per_layer, 2]
            number_of_temporal_layers = temporal_layers_centers.shape[0]
            
            # Check if irregular grid
            if num_voxels_per_layer != num_voxels / number_of_temporal_layers:
                raise ValueError(
                    "The supplied grid does not have equally sized temporal layers.")
            # Check if combined_coords_array has 4 columns if weights are present
            if weights_present == True and combined_coords_array.shape[1] != 4:
                raise ValueError(
                    "weights_present is True, but the provided combined_coords_array does not have 4 columns")

            # Extract info about x layers
            x_layers_centers = np.unique(flattened_voxel_grid[:num_voxels_per_layer, 0]) # These values are sorted by np.unique

            # Note: we cannot get the num of voxels per x layer, because they might not be equal because of clipping to area
            number_of_x_layers = x_layers_centers.shape[0]

            # Extract info about y layers
            y_layers_centers = np.unique(flattened_voxel_grid[:num_voxels_per_layer, 1]) # These values are sorted by np.unique
            
            # Note: we cannot get the num of voxels per y layer, because they might not be equal because of clipping to area
            number_of_y_layers = y_layers_centers.shape[0]

            # Calculate stkde normalization factor 
            normalization_factor = 1 / (n * x_bandwidth * y_bandwidth * t_bandwidth)

            # Temporal slide ------------------------------------------------------------------------------------------------------------------------------

            # Preallocate two arrays for the temporal indexes (single array was giving errors)
            indexes_min = np.zeros(number_of_temporal_layers, dtype=np.uintc)
            indexes_max = np.zeros(number_of_temporal_layers, dtype=np.uintc)

            # Use a parallel loop to find the indexes for the points within the temporal bandwidth per temporal voxel layer
            for layer_index in prange(number_of_temporal_layers):
                # Extract current layer t value
                t_center = temporal_layers_centers[layer_index]

                # Compute the indexes and store them in the array
                # Find first index of lowest value (start of slice is inclusive)
                indexes_min[layer_index] = np.searchsorted(
                    combined_coords_array[:, 2], t_center - t_bandwidth, side='right')
                # Find first index of higher value than center + bandwidth (final is exclusive)
                indexes_max[layer_index] = np.searchsorted(
                    combined_coords_array[:, 2], t_center + t_bandwidth, side='left')

            # Find all points within each X slice -------------------------------------------------------------------------------------------------------------

            # Sort combined coords array by x (only indices)
            # Contains original arrays indices but sorted by their x_values
            x_sorted_combined_coords_array = np.argsort(combined_coords_array[:, 0])

            # Initialize array where row represents an x layer and column all the points
            x_binary_arrays = np.zeros(
                (number_of_x_layers, num_points), dtype=np.bool_)

            # Use a parallel loop to extract the indexes for the points within the x bandwidth per x voxel layer
            for layer_index in prange(number_of_x_layers):

                # Get the center of the current x layer
                x_center = x_layers_centers[layer_index]

                # Find the start and end indices for x_sorted_combined_coords_array for which the values in the original array (combined_coords) are within the bandwidth
                start_index = np.searchsorted(
                    combined_coords_array[x_sorted_combined_coords_array, 0], x_center - x_bandwidth, side='right')
                end_index = np.searchsorted(
                    combined_coords_array[x_sorted_combined_coords_array, 0], x_center + x_bandwidth, side='left')

                # Create a binary array per x slice
                binary_array = np.zeros(num_points, dtype=np.bool_)

                # If the start and end indices are the same, there are no points in the x slice
                if start_index == end_index:
                    x_binary_arrays[layer_index] = binary_array
                else:
                    # If the end index is within or outside the range of points, set True values accordingly
                    x_within_bandwidth_indices = x_sorted_combined_coords_array[start_index:end_index]

                    binary_array[x_within_bandwidth_indices] = True

                    # Store the binary array for the x slice
                    x_binary_arrays[layer_index] = binary_array
            
            # Find all points within each Y slice -------------------------------------------------------------------------------------------------------------

            # Sort combined coords array by x (only indices)
            # Contains original arrays indices but sorted by their y_values
            y_sorted_combined_coords_array = np.argsort(combined_coords_array[:, 1])

            # Initialize array where row represents an x layer and column all the points
            y_binary_arrays = np.zeros(
                (number_of_y_layers, num_points), dtype=np.bool_)

            # Use a parallel loop to extract the indexes for the points within the y bandwidth per y voxel layer
            for layer_index in prange(number_of_y_layers):

                # Get the center of the current y layer
                y_center = y_layers_centers[layer_index]

                # Find the start and end indices for y_sorted_combined_coords_array for which the values in the original array (combined_coords) are within the bandwidth
                start_index = np.searchsorted(
                    combined_coords_array[y_sorted_combined_coords_array, 1], y_center - y_bandwidth, side='right')
                end_index = np.searchsorted(
                    combined_coords_array[y_sorted_combined_coords_array, 1], y_center + y_bandwidth, side='left')

                # Create a binary array per y slice
                binary_array = np.zeros(num_points, dtype=np.bool_)

                # If the start and end indices are the same, there are no points in the y slice
                if start_index == end_index:
                    y_binary_arrays[layer_index] = binary_array
                else:
                    # If the end index is within or outside the range of points, set True values accordingly
                    y_within_bandwidth_indices = y_sorted_combined_coords_array[start_index:end_index]
                    binary_array[y_within_bandwidth_indices] = True

                    # Store the binary array for the y slice
                    y_binary_arrays[layer_index] = binary_array

            # Loop through the entire grid in parallel ----------------------------------------------------------------------------------------------------
            for voxel_index in prange(num_voxels):

                # Extract coordinates of the voxel
                voxel = flattened_voxel_grid[voxel_index]
                x, y, t = voxel

                # Extract binary array for points within x bandwidth relevant to the voxel TODO: If you wish to use this function please replace this with the logic from the other function
                x_binary_array_index = np.searchsorted(x_layers_centers, x)

                # Extract binary array for points within y bandwidth relevant to the voxel
                y_binary_array_index = np.searchsorted(y_layers_centers, y)

                # Calculate which temporal layer the voxel belongs to
                voxel_layer_index = voxel_index // num_voxels_per_layer

                # Extract t indices for the current temporal layer
                lower_t_index = indexes_min[voxel_layer_index]
                upper_t_index = indexes_max[voxel_layer_index]

                # If the lower and upper temporal indexes are the same, there are no points in the temporal slice
                if lower_t_index == upper_t_index:
                    stkde_values[voxel_index] = 0.0
                    continue

                # Slice the arrays to the correct indices
                sliced_coords_array = combined_coords_array[lower_t_index:upper_t_index]
                
                # Slice x and y binary arrays to points within t bandwidth
                sliced_x_binary_array = x_binary_arrays[x_binary_array_index][lower_t_index:upper_t_index]
                sliced_y_binary_array = y_binary_arrays[y_binary_array_index][lower_t_index:upper_t_index]

                # Remove all points not within x AND y bandwidths
                combined_mask_xy = np.bitwise_and(sliced_x_binary_array, sliced_y_binary_array)
                
                # Mask the points within temporal bandwidth to keep points within all bandwidths
                points_within_range = sliced_coords_array[combined_mask_xy]

                # If no neighbors within ranges, set STKDE value to 0 and continue
                if points_within_range.shape[0] == 0:
                    stkde_values[voxel_index] = 0.0
                    continue

                # Extract values of the nearest neighbor points
                x_values, y_values, t_values = points_within_range[:,
                                                                0], points_within_range[:, 1], points_within_range[:, 2]

                # Calculate STKDE value for voxel --------------------------------------------------------------

                # Calculate Epanechnikov kernel values
                # 1 - x^2 * 3/4 for |x^2| < 1 and 0 otherwise. Because we have already filtered the points within the bandwidths we can skip the 0 condition
                kernel_values_x = (1 - ((x - x_values) / x_bandwidth)**2) * 0.75
                kernel_values_y = (1 - ((y - y_values) / y_bandwidth)**2) * 0.75
                kernel_values_t = (1 - ((t - t_values) / t_bandwidth)**2) * 0.75

                # Multiply arrays together
                product = kernel_values_x * kernel_values_y * kernel_values_t

                # If weights are present multiply with weights
                if weights_present == True:
                    w_values = points_within_range[:, 3]
                    product = product * w_values

                # Sum the array
                array_sum = np.sum(product)

                # Multiply with normalization factor (inside of loop is faster because of parallelization)
                stkde_values[voxel_index] = normalization_factor * array_sum

            return stkde_values
        
        # Very old functions are below, these are very slow and should not be used. Unfortunately they are still required for the older methods shown here to work.
        @njit(UniTuple((Array(float64, 1, 'C')), 3)(Array(float64, 1, 'C'), Array(float64, 1, 'C'), Array(float64, 1, 'C')))
        def epanechnikov_kernel_values_numba(x:np.ndarray, y:np.ndarray, t:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Calculates the Epanechnikov kernel values for given np.arrays x, y, and t which contain normalized distances from the voxel point to the existing points.
            Epanechnikov kernel is defined as K(x) = 3/4 * (1 - x^2) for |x^2| < 1 and K(x) = 0 otherwise

            Args:
                x (np.array): Input array for x
                y (np.array): Input array for y
                t (np.array): Input array for t

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing arrays of kernel values for x, y, and t
            """
            
            x_squared = x**2
            y_squared = y**2
            t_squared = t**2

            kernel_values_x = np.zeros(x.shape, dtype=np.float64)
            kernel_values_y = np.zeros(y.shape, dtype=np.float64)
            kernel_values_t = np.zeros(t.shape, dtype=np.float64)

            within_bandwidth_x = x_squared < 1
            within_bandwidth_y = y_squared < 1
            within_bandwidth_t = t_squared < 1

            kernel_values_x[within_bandwidth_x] = (
                3/4) * (1 - x_squared[within_bandwidth_x])
            kernel_values_y[within_bandwidth_y] = (
                3/4) * (1 - y_squared[within_bandwidth_y])
            kernel_values_t[within_bandwidth_t] = (
                3/4) * (1 - t_squared[within_bandwidth_t])

            return kernel_values_x, kernel_values_y, kernel_values_t

        @njit
        def stkde_voxel(x: np.float64, y: np.float64, t: np.float64, hx: np.float64, hy: np.float64, ht: np.float64, n:np.uint32,
                        x_existing: np.ndarray, y_existing: np.ndarray, t_existing: np.ndarray, w_existing: Optional[np.ndarray] = None) -> np.float64:
            """ Calculates the STKDE for a single voxel point using the Epanechnikov kernel.

            Args:
                x (np.float64): X coordinate of the voxel point
                y (np.float64): Y coordinate of the voxel point
                t (np.float64): T coordinate of the voxel point
                hx (np.float64): X bandwidth
                hy (np.float64): Y bandwidth
                ht (np.float64): T bandwidth
                n (Union[np.uint32, np.float64]): Number of existing points, or neff if weights are present
                x_existing (np.ndarray): X coordinates of the existing points
                y_existing (np.ndarray): Y coordinates of the existing points
                t_existing (np.ndarray): T coordinates of the existing points
                w_existing (Optional[np.ndarray], optional): Weights of the existing points. Defaults to None.

            Returns:
                np.float64: STKDE value for the voxel point
            """

            # Calculate the argument to be put into the Epanechnikov kernel value for each of the x, y, t values of the existing points
            epan_x_argument = (x - x_existing) / hx
            epan_y_argument = (y - y_existing) / hy
            epan_t_argument = (t - t_existing) / ht

            # Calculate the Epanechnikov kernel value for each of the x, y, t values of the existing points
            epan_x_output, epan_y_output, epan_t_output = epanechnikov_kernel_values_numba(
                epan_x_argument, epan_y_argument, epan_t_argument)

            # Loop through all returned epanechikov values and multiply them together
            array_sum = np.float64(0.0)
            for i in range(epan_x_output.shape[0]):
                point_value = epan_x_output[i] * epan_y_output[i] * epan_t_output[i]
                if w_existing is not None:
                    point_value = point_value * w_existing[i]
                array_sum += point_value

            # Calculate STKDE value for voxel
            stkde_value = (1 / (n * hx * hy * ht)) * array_sum

            # Return the STKDE value
            return stkde_value

        @njit(Array(float64, 2, 'C', False, aligned=True)(Array(float64, 2, 'C', True, aligned=True), float64, float64, float64, float64, float64, float64))
        def filter_coordinates_numba(existing_points: np.ndarray, x_point:np.float64, y_point:np.float64, t_point:np.float64,
                                    x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64) -> np.ndarray:
            """ Brute force range query solution for filtering coordinates within a bandwidth of a point. O(n) complexity, but is called n times so O(n^2) complexity.

            Args:
                existing_points (np.ndarray): (n,3) array of existing points with x, y, t coordinates
                x_point (np.float64): Coordinate of the point in the x dimension
                y_point (np.float64): Coordinate of the point in the y dimension
                t_point (np.float64): Coordinate of the point in the t dimension
                x_bandwidth (np.float64): X bandwidth
                y_bandwidth (np.float64): Y bandwidth
                t_bandwidth (np.float64): T bandwidth

            Returns:
                np.ndarray: Coordinates of points within the bandwidths of the point
            """
            
            # Calculate min and max values
            min_x = x_point - x_bandwidth
            max_x = x_point + x_bandwidth
            min_y = y_point - y_bandwidth
            max_y = y_point + y_bandwidth
            min_t = t_point - t_bandwidth
            max_t = t_point + t_bandwidth

            # Create empty output array
            filtered_coords = np.empty_like(existing_points)
            count = 0
            for i in range(len(existing_points)):
                x, y, t = existing_points[i]
                if min_x < x < max_x and min_y < y < max_y and min_t < t < max_t:
                    filtered_coords[count] = existing_points[i]
                    count += 1
            
            return filtered_coords[:count]

        @njit(Array(float64, 2, 'C', False, aligned=True)(Array(float64, 2, 'C', True, aligned=True), float64, float64, float64, float64, float64, float64))  # O(n) complexity performs the best at n < 7000
        def filter_coordinates_numba_weights(existing_points: np.ndarray, x_point:np.float64, y_point:np.float64, t_point:np.float64,
                                            x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64) -> np.ndarray:
            """ Brute force range query solution for filtering coordinates within a bandwidth of a point. O(n) complexity, but is called n times so O(n^2) complexity.

            Args:
                existing_points (np.ndarray): (n,3) array of existing points with x, y, t coordinates
                x_point (np.float64): Coordinate of the point in the x dimension
                y_point (np.float64): Coordinate of the point in the y dimension
                t_point (np.float64): Coordinate of the point in the t dimension
                x_bandwidth (np.float64): X bandwidth
                y_bandwidth (np.float64): Y bandwidth
                t_bandwidth (np.float64): T bandwidth

            Returns:
                np.ndarray: Coordinates of points within the bandwidths of the point
            """
            
            # Calculate min and max values
            min_x = x_point - x_bandwidth
            max_x = x_point + x_bandwidth
            min_y = y_point - y_bandwidth
            max_y = y_point + y_bandwidth
            min_t = t_point - t_bandwidth
            max_t = t_point + t_bandwidth

            # Create empty output array
            filtered_coords = np.empty_like(existing_points)
            count = 0
            for i in range(len(existing_points)):
                x, y, t, w = existing_points[i]
                if min_x < x < max_x and min_y < y < max_y and min_t < t < max_t:
                    filtered_coords[count] = existing_points[i]
                    count += 1
            return filtered_coords[:count]
        
        
        @njit(Array(float64, 1, 'C', False, aligned=True)(Array(float64, 2, 'C', True, aligned=True), Array(float64, 2, 'C', True, aligned=True), 
                                                                (float64), (float64), (float64), (uint32), (boolean)))
        def calculate_stkde_for_voxel_grid_brute_force(flattened_voxel_grid:np.ndarray, combined_coords_array: np.ndarray,
                                                       x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64, n:np.uint32, weights_present:np.bool_):
            # Extract shape of the flattened voxel grid (number of voxels)
            num_voxels = flattened_voxel_grid.shape[0]
            
            # Create empty numpy array to store STKDE values, set data type to float64 (faster)
            stkde_values = np.zeros(num_voxels, dtype=np.float64)

            if weights_present == False:
                # Loop through each voxel and calculate STKDE value 
                for index, voxel in enumerate(flattened_voxel_grid):
                    # Extract coordinates of the voxel
                    x, y, t = voxel

                    # Find points within range
                    points_within_range = filter_coordinates_numba(
                        combined_coords_array, x_point=x, y_point=y, t_point=t, x_bandwidth=x_bandwidth, y_bandwidth=y_bandwidth, t_bandwidth=t_bandwidth)
                    
                    # If no neighbors within ranges, set STKDE value to 0 and continue
                    if len(points_within_range) == 0:
                        stkde_values[index] = 0.0
                        continue

                    # Extract values of the nearest neighbor points
                    x_values, y_values, t_values = points_within_range[:,
                                                                    0], points_within_range[:, 1], points_within_range[:, 2]

                    # Calculate STKDE value for voxel
                    stkde_values[index] = stkde_voxel(x=x, y=y, t=t, hx=x_bandwidth, hy=y_bandwidth,
                                                    ht=t_bandwidth, n=n, x_existing=x_values, y_existing=y_values, t_existing=t_values)

            # If no weights method split into two because if statements * multiple millions is slow:
            else:
                # Loop through each voxel and calculate STKDE value (This could probably be faster, but joblib just made it slower in my attempts)
                for index, voxel in enumerate(flattened_voxel_grid):
                    # Extract coordinates of the voxel
                    x, y, t = voxel

                    # Find points within range
                    points_within_range = filter_coordinates_numba_weights(
                        combined_coords_array, x_point=x, y_point=y, t_point=t, x_bandwidth=x_bandwidth, y_bandwidth=y_bandwidth, t_bandwidth=t_bandwidth)

                    # If no neighbors within ranges, set STKDE value to 0 and continue
                    if len(points_within_range) == 0:
                        stkde_values[index] = 0.0
                        continue

                    # Extract values of the nearest neighbor points
                    x_values, y_values, t_values, w_values = points_within_range[:,
                                                                                0], points_within_range[:, 1], points_within_range[:, 2], points_within_range[:, 3]

                    # Calculate STKDE value for voxel
                    stkde_values[index] = stkde_voxel(x=x, y=y, t=t, hx=x_bandwidth, hy=y_bandwidth, ht=t_bandwidth,
                                                    n=n, x_existing=x_values, y_existing=y_values, t_existing=t_values, w_existing=w_values)
            return stkde_values
        
        @njit(Array(float64, 1, 'C', False, aligned=True)(Array(float64, 2, 'C', True, aligned=True), Array(float64, 2, 'C', True, aligned=True), 
                                                                (float64), (float64), (float64), (uint32), (boolean)), parallel = True)
        def calculate_stkde_for_voxel_grid_brute_force_parallel(flattened_voxel_grid:np.ndarray, combined_coords_array: np.ndarray, 
                                                                x_bandwidth:np.float64, y_bandwidth:np.float64, t_bandwidth:np.float64, n:np.uint32, weights_present:np.bool_):

            # Extract shape of the flattened voxel grid (number of voxels)
            num_voxels = flattened_voxel_grid.shape[0]
            
            # Create empty numpy array to store STKDE values, set data type to float64 (faster)
            stkde_values = np.zeros(num_voxels, dtype=np.float64)
            
            if weights_present == False:
                # Loop through each voxel and calculate STKDE value 
                for index in prange(num_voxels):
                    # Extract coordinates of the voxel
                    voxel = flattened_voxel_grid[index]
                    x, y, t = voxel

                    # Find points within range
                    points_within_range = filter_coordinates_numba(
                        combined_coords_array, x_point=x, y_point=y, t_point=t, x_bandwidth=x_bandwidth, y_bandwidth=y_bandwidth, t_bandwidth=t_bandwidth)

                    # If no neighbors within ranges, set STKDE value to 0 and continue
                    if len(points_within_range) == 0:
                        stkde_values[index] = 0.0
                        continue

                    # Extract values of the nearest neighbor points
                    x_values, y_values, t_values = points_within_range[:,
                                                                    0], points_within_range[:, 1], points_within_range[:, 2]

                    # Calculate STKDE value for voxel
                    stkde_values[index] = stkde_voxel(x=x, y=y, t=t, hx=x_bandwidth, hy=y_bandwidth,
                                                    ht=t_bandwidth, n=n, x_existing=x_values, y_existing=y_values, t_existing=t_values)

            # If no weights method split into two because if statements * multiple millions is slow:
            else:
                # Loop through each voxel and calculate STKDE value (This could probably be faster, but joblib just made it slower in my attempts)
                for index in prange(num_voxels):
                    # Extract coordinates of the voxel
                    voxel = flattened_voxel_grid[index]
                    x, y, t = voxel

                    # Find points within range
                    points_within_range = filter_coordinates_numba_weights(
                        combined_coords_array, x_point=x, y_point=y, t_point=t, x_bandwidth=x_bandwidth, y_bandwidth=y_bandwidth, t_bandwidth=t_bandwidth)

                    # If no neighbors within ranges, set STKDE value to 0 and continue
                    if len(points_within_range) == 0:
                        stkde_values[index] = 0.0
                        continue

                    # Extract values of the nearest neighbor points
                    x_values, y_values, t_values, w_values = points_within_range[:,
                                                                                0], points_within_range[:, 1], points_within_range[:, 2], points_within_range[:, 3]

                    # Calculate STKDE value for voxel
                    stkde_values[index] = stkde_voxel(x=x, y=y, t=t, hx=x_bandwidth, hy=y_bandwidth, ht=t_bandwidth,
                                                    n=n, x_existing=x_values, y_existing=y_values, t_existing=t_values, w_existing=w_values)

            return stkde_values
        
        @njit(Array(float64, 2, 'C', False, aligned=True)(Array(float64, 1, 'C', True, aligned=True), Array(float64, 1, 'C', True, aligned=True), Array(float64, 1, 'C', True, aligned=True)),
              parallel=True)
        def create_voxel_grid(x_linspace:NDArray[np.float64], y_linspace:NDArray[np.float64], t_linspace:NDArray[np.float64]) -> NDArray[np.float64]:
            """Create a voxel grid from the provided linspace arrays for x, y, and t. NOTE: grid is in order of t, x, y.

            Args:
                x_linspace (NDArray[np.float64]): Voxel centers in the x dimension.
                y_linspace (NDArray[np.float64]): Voxel centers in the y dimension.
                t_linspace (NDArray[np.float64]): Voxel centers in the t dimension.

            Returns:
                NDArray[np.float64]: Flattened voxel grid with shape (num_voxels, 3) where each row is (x, y, t).
            """            
            
            # Get number of voxel centres in each dimension
            x_num_voxels = x_linspace.shape[0]
            y_num_voxels = y_linspace.shape[0]
            t_num_voxels = t_linspace.shape[0]
            
            # Calculate total number of voxels
            xy_voxels = x_num_voxels * y_num_voxels
            total_voxels = xy_voxels * t_num_voxels

            # Initialize the flattened voxel grid array
            flattened_voxel_grid = np.empty((total_voxels, 3), dtype=np.float64)
            
            # Parallel loop, starting with t because we want the grid to be t sorted later
            for i in prange(t_num_voxels):
                t_val = t_linspace[i]
                t_index = i * xy_voxels
                for j in range(x_num_voxels):
                    x_val = x_linspace[j]
                    x_index = (j * y_num_voxels) + t_index
                    for k in range(y_num_voxels):
                        y_val = y_linspace[k]
                        index = x_index + k
                        flattened_voxel_grid[index, 0] = x_val
                        flattened_voxel_grid[index, 1] = y_val
                        flattened_voxel_grid[index, 2] = t_val
            return flattened_voxel_grid
        
        
        # Running all 4 versions of the calculate_stkde_for_voxel_grid function -----------------------------------------------------------------------------------------

        # Define the size of your arrays
        n = np.uint32(10000) # number of points in point cloud
        m = 50 # number of layers in meshgrid in each dimensions (m^3 = total number of voxels)
        x_bw = np.float64(5)
        y_bw = np.float64(5)
        t_bw = np.float64(2)
        
        # Generate a (m^3,3) array for the meshgrid
        x = np.linspace(0, 100, m)
        y = np.linspace(0, 100, m)
        t = np.linspace(0, 100, m)

        # Create grid
        grid = create_voxel_grid(x, y, t)

        # Generate a random (n,3) point cloud with values between 0 and 100
        # NOTE: Note we do not use weights because Neff functionality was added later on :)
        point_cloud = np.random.randint(low=1, high=99, size=(n, 3))
        # Convert to float64
        point_cloud = point_cloud.astype(np.float64)
        
        # Sorting is not required for any method but the sorting method, for consistency reuse across all methods.
        # Record time taken and add to total time taken for sorted method later
        start_time_sort = time.time()
        point_cloud = point_cloud[np.argsort(point_cloud[:, 2])] 
        time_taken_sort = time.time() - start_time_sort
        
        # Print information about calculations
        print("\nRunning tests...")
        print(f"Number of voxels: {len(grid)}")
        print(f"Number of points in point cloud: {len(point_cloud)}\n")

        print("Calculating Spatio-Temporal Kernel Density Estimation (STKDE) for brute force method:")
        time1 = time.time()
        loop_method = calculate_stkde_for_voxel_grid_brute_force(grid, point_cloud, x_bw, y_bw, t_bw, n, np.bool_(False))
        print(f"Time taken: {time.time()-time1:.2f}s")

        print("\nCalculating STKDE for brute force method (Parallel):")
        time2 = time.time()
        loop_p_method = calculate_stkde_for_voxel_grid_brute_force_parallel(grid, point_cloud, x_bw, y_bw, t_bw, n, np.bool_(False))
        print(f"Time taken: {time.time()-time2:.2f}s")

        print("\nCalculating STKDE using precomputed sorting method:")
        time3 = time.time()
        sorting_method = calculate_stkde_for_voxel_grid_sorting(grid, point_cloud, x_bw, y_bw, t_bw, n, np.bool_(False))
        print(f"Time taken: {(time.time()-time3) + time_taken_sort:.2f}s") # Add time to sort

        print(f"\nCalculating STKDE using STOPKDE:")
        time4 = time.time()
        stopkde = calculate_stkde_for_voxel_grid(point_cloud, np.float64(x_bw), np.float64(y_bw), np.float64(t_bw), x, y, t, True) # Return as ijk
        print(f"Time taken: {time.time()-time4:.2f}s")
        
        # Compare the arrays
        print("\nComparing the results of the older methods:")
        print(f"{'- Brute force is equal to sorting method: ':<56}{np.array_equal(loop_method, sorting_method)}")
        print(f"{'- Brute force is equal to brute force parallel method: ':<56}{np.array_equal(loop_method, loop_p_method)}")
        
        print("\nComparing the results of the older methods to the STOPKDE method:")
        # NOTE: Different output orders require reshaping here to see if the results match. 
        loop_method_transpose = loop_method.reshape((m, m, m)).transpose((1, 2, 0)).flatten()
        loop_p_method_transpose = loop_p_method.reshape((m, m, m)).transpose((1, 2, 0)).flatten()
        sorting_method_transpose = sorting_method.reshape((m, m, m)).transpose((1, 2, 0)).flatten()
        close1 = np.allclose(loop_method_transpose, stopkde)
        close2 = np.allclose(loop_p_method_transpose, stopkde)
        close3 = np.allclose(sorting_method_transpose, stopkde)
        print(f"{'- Brute force is close to STOPKDE: ':<45}{close1}.    Average distance: {np.abs(loop_method_transpose - stopkde).mean():.3e}, max distance: {np.abs(loop_method_transpose - stopkde).max():.3e}.")
        print(f"{'- Brute force parallel is close to STOPKDE: ':<45}{close2}.    Average distance: {np.abs(loop_p_method_transpose - stopkde).mean():.3e}, max distance: {np.abs(loop_p_method_transpose - stopkde).max():.3e}.")
        print(f"{'- Sorting method is close to STOPKDE: ':<45}{close3}.    Average distance: {np.abs(sorting_method_transpose - stopkde).mean():.3e}, max distance: {np.abs(sorting_method_transpose - stopkde).max():.3e}.")
        print("\n")
        if close1 and close2 and close3:
            print("STOPKDE method normalizes the results in a different (but mathematically equal) order, therefore results are near equal due to floating point multiplications.")
            print("All methods have returned equal or similar results, tests passed")
        else:
            print("STOPKDE has not returned values close to the other methods, a bug is likely occuring please report this issue.")
        print("\n")