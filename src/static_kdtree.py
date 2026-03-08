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



# Standard library imports
from math import ceil, log2
from typing import Tuple

# Third-party imports
import numpy as np
from numpy.typing import NDArray
from numba import njit



class StaticKDTree:
    """Class for static k-dimensional trees written in Numba for efficient AABB (axis-aligned bounding box) queries. 
    Provides method to construct KD-trees from points and perform AABB queries on these trees.
    """
    __slots__ = ("nodes", "leaves", "n_points", "dimensions", "leaf_size", "leaf_width")
    
    def __init__(self, points:NDArray[np.float64], weighted:bool=False, leaf_size:int=32):
        """Create a Static KDTree.

        Args:
            points (NDArray[np.float64]): Array of shape (n, k) where n is the number of points and k is the number of dimensions.
            weighted (bool, optional): If True, last (-1) array dimension is stored as weight instead of coordinate. Defaults to False.
            leaf_size (int, optional): Number of points stored in the KDTree leaves, multiples of 2 generally recommended. Defaults to 32.
        """
        # Extract dimensions and leaf sizes
        n_points, leaf_width = points.shape
        dimensions = leaf_width - 1 if weighted else leaf_width
        
        # Create the structured array datatypes and build the KDTree
        node_dtype, leaf_dtype, stack_dtype = self._create_dtypes(leaf_size, leaf_width)
        self.nodes, self.leaves = self._build(points, dimensions, leaf_size, node_dtype, leaf_dtype, stack_dtype)
        
        # Store other properties
        self.n_points = n_points
        self.dimensions = dimensions
        self.leaf_size = leaf_size
        self.leaf_width = leaf_width
    
    ## Private methods
    def _create_dtypes(self, leaf_size:int, leaf_width:int) -> Tuple[np.dtype[np.void], np.dtype[np.void], np.dtype[np.void]]:
        """Create NumPy structured datatypes needed for the KDTree. 
        NOTE: This is currently not possible inside @njit functions, so we do it here.

        Args:
            leaf_size (int): Number of points stored in the KDTree leaves.
            leaf_width (int): Number of values to store per point.

        Returns:
            Tuple[np.dtype[np.void], np.dtype[np.void], np.dtype[np.void]]: Structured datatypes for nodes, leaves, and stack.
        """       
        # Create node, leaf, and stack dtypes
        node_dtype = np.dtype([
            ('split_val', np.float64),
            ('left_idx', np.int32),
            ('right_idx', np.int32),
            ('leaf_idx', np.int32)
        ], align=True)
        leaf_dtype = np.dtype([
            ('count', np.uint32),
            ('points', np.float64, (leaf_size, leaf_width))
        ], align=True)
        stack_dtype = np.dtype([
            ('start', np.uint32),
            ('end', np.uint32),
            ('split_dim', np.uint32),
            ('node_idx', np.uint32)
        ])
        return node_dtype, leaf_dtype, stack_dtype
        
    # Numba methods use @staticmethod to allow for compilation.
    @staticmethod
    @njit(cache=True)
    def _build(
        points:NDArray[np.float64],
        dimensions:int,
        leaf_size:int,
        node_dtype:np.dtype[np.void],
        leaf_dtype:np.dtype[np.void],
        stack_dtype:np.dtype[np.void]
    ) -> Tuple[NDArray[np.void], NDArray[np.void]]:
        """Function to create the static KDTree, outputs array of nodes and array of leaves.

        Args:
            points (NDArray[np.float64]): Array of shape (n, k).
            dimensions (int): Number of dimensions.
            leaf_size (int): Number of points stored in the KDTree leaves.
            node_dtype (np.dtype[np.void]): NumPy structured datatype for nodes array.
            leaf_dtype (np.dtype[np.void]): NumPy structured datatype for leaves array.
            stack_dtype (np.dtype[np.void]): NumPy structured datatype for stack array.

        Returns:
            Tuple[NDArray[np.void], NDArray[np.void]]: Structured arrays containing KDTree internal and leaf nodes.
        """
        # Limit size of arrays
        n = points.shape[0]
        max_depth = ceil(log2(n / leaf_size)) + 1 # Conservative estimate
        max_leaves = 2 ** max_depth
        max_nodes = 2 * max_leaves - 1 # Also includes leaves, due to storing the points' values in the leaves array
        
        # Allocate node and leaf arrays
        nodes = np.empty((max_nodes,), dtype=node_dtype)
        leaves = np.empty((max_leaves,), dtype=leaf_dtype)
        
        # Create array of original indexes (avoid reshuffling dimensions * n 64 bit values)
        indices = np.arange(n, dtype=np.int32)

        # Initialise the stack to track the remaining nodes, limit stack size to depth: log₂(n/k) + 2, where k is leaf size, plus safety margin (2).
        max_stack_size = ceil(log2(n / leaf_size)) + 2
        stack = np.empty((max_stack_size,), dtype=stack_dtype)
        sp = 0  # Keep track of where we are in the stack

        # Initialise counters for nodes and leaves
        node_counter = 0
        leaf_counter = 0
        
        # Create root node task in stack
        stack[sp]['start'] = 0
        stack[sp]['end'] = n
        stack[sp]['split_dim'] = 0
        stack[sp]['node_idx'] = node_counter
        sp += 1 
        node_counter += 1

        # Loop until the stack pointer is 0, i,e, no more nodes remaining
        while sp > 0:
            
            # Get the last inserted node from the stack (left child)
            sp -= 1
            start = stack[sp]['start']
            end = stack[sp]['end']
            split_dim = stack[sp]['split_dim']
            node_idx = stack[sp]['node_idx']
            n = end - start # Remaining number of points

            # If the leaf size is larger create a leaf, else create an internal node with two children
            if n <= leaf_size:
                leaf_idx = leaf_counter
                leaf = leaves[leaf_idx]
                leaf['count'] = n
                
                # Store the points in the leaf
                for i in range(n):
                    leaf['points'][i, :] = points[indices[start + i]]
                nodes[node_idx]['leaf_idx'] = leaf_idx
                nodes[node_idx]['left_idx'] = -1
                nodes[node_idx]['right_idx'] = -1
                leaf_counter += 1
                continue
            
            # Get the median value for the split_dim, and reorder the other index values around this value.
            median_idx = start + n // 2
            range_indices = indices[start:end]
            range_points = points[range_indices, split_dim]
            part = np.argpartition(range_points, n // 2)
            indices[start:end] = range_indices[part]
            split_val = points[indices[median_idx], split_dim]

            # Create the node
            nodes[node_idx]['split_val'] = split_val
            nodes[node_idx]['leaf_idx'] = -1
            
            left_idx = node_counter
            node_counter += 1
            nodes[node_idx]['left_idx'] = left_idx
            
            right_idx = node_counter
            node_counter += 1
            nodes[node_idx]['right_idx'] = right_idx

            # Get the split_dim for the child nodes (avoid doing the modulo twice)
            child_dim = (split_dim + 1) % dimensions
            
            # Push right child
            stack[sp]['start'] = median_idx # Exclude the median point from going right
            stack[sp]['end'] = end
            stack[sp]['split_dim'] = child_dim
            stack[sp]['node_idx'] = right_idx
            sp += 1

            # Push left child (Ensures dfs because the stack is LIFO)
            stack[sp]['start'] = start
            stack[sp]['end'] = median_idx
            stack[sp]['split_dim'] = child_dim
            stack[sp]['node_idx'] = left_idx
            sp += 1
            
        return nodes[:node_counter], leaves[:leaf_counter]
    
    @staticmethod
    @njit(cache=True)
    def _query_aabb_3d(
        nodes:NDArray[np.void], 
        leaves:NDArray[np.void], 
        n_points:int,
        leaf_size:int, leaf_width:int,
        xmin:float, xmax:float,
        ymin:float, ymax:float,
        zmin:float, zmax:float
    ) -> NDArray[np.float64]:
        """Execute an axis-aligned bounding box query on a 3D KDTree.

        Args:
            nodes (NDArray): Structured array containing KDTree internal nodes.
            leaves (NDArray): Structured array containing KDTree leaf nodes.
            n_points (int): Number of points in the original dataset.
            leaf_size (int): Number of points per leaf.
            leaf_width (int): Dimensionality of the points in the leaf.
            xmin (float): Minimum x-coordinate of the bounding box.
            xmax (float): Maximum x-coordinate of the bounding box.
            ymin (float): Minimum y-coordinate of the bounding box.
            ymax (float): Maximum y-coordinate of the bounding box.
            zmin (float): Minimum z-coordinate of the bounding box.
            zmax (float): Maximum z-coordinate of the bounding box.

        Returns:
            NDArray[np.float64]: Array of shape (n, 3) or (n, 4) containing all points within the bounding box.
        """
        
        # Pre-allocate result array for worst case (all points)
        results = np.empty((n_points, leaf_width), dtype=np.float64)
        result_count = 0
        
        # Stack for iterative traversal: (node_idx, axis)
        max_stack_size = ceil(log2(n_points / leaf_size)) + 2
        stack = np.empty((max_stack_size, 2), dtype=np.int32)
        stack_ptr = 0
        stack[stack_ptr, 0] = 0  # root node
        stack[stack_ptr, 1] = 0  # x-axis
        stack_ptr += 1
        
        bounds_min = np.array([xmin, ymin, zmin])
        bounds_max = np.array([xmax, ymax, zmax])
        
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = stack[stack_ptr, 0]
            axis = stack[stack_ptr, 1]
            node = nodes[node_idx]
            
            # Check if this is a leaf node
            if node['leaf_idx'] != -1:
                leaf = leaves[node['leaf_idx']]
                count = leaf['count']
                points = leaf['points']
                
                # Filter points within AABB
                for i in range(count):
                    point = points[i]
                    point_x = point[0]
                    point_y = point[1]
                    point_z = point[2]
                    if xmin < point_x < xmax and ymin < point_y < ymax and zmin < point_z < zmax:
                        results[result_count] = point
                        result_count += 1
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
        return results[:result_count]
    
    ## Public methods
    def query_aabb_3d(
        self,
        xmin:float, xmax:float,
        ymin:float, ymax:float,
        zmin:float, zmax:float
    ) -> NDArray[np.float64]:
        """Execute an axis-aligned bounding box query on a 3D KDTree.

        Args:
            xmin (float): Minimum x-coordinate of the bounding box.
            xmax (float): Maximum x-coordinate of the bounding box.
            ymin (float): Minimum y-coordinate of the bounding box.
            ymax (float): Maximum y-coordinate of the bounding box.
            zmin (float): Minimum z-coordinate of the bounding box.
            zmax (float): Maximum z-coordinate of the bounding box.

        Returns:
            NDArray[np.float64]: Array of shape (n, 3) or (n, 4) containing all points within the bounding box.
        """
        if not self.dimensions == 3:
            raise ValueError(
                f"Cannot execute 3d query on {self.dimensions}d tree."
            )
        return self._query_aabb_3d(
            self.nodes,
            self.leaves,
            self.n_points,
            self.leaf_size,
            self.leaf_width,
            xmin, xmax,
            ymin, ymax,
            zmin, zmax
        )