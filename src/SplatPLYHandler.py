import struct
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class SplatPLYHandler:
    """
    A class for handling and manipulating PLY (Polygon File Format) and SPLAT PLY (3D Gaussian splatting) files.

    This class provides methods to:
    - Load and parse PLY files in binary little-endian format.
    - Save modified PLY files with updated elements and properties.
    - Normalize and standardize property names for consistency.
    - Apply transformations (rotation, translation) to vertex positions and normals.
    - Perform ICP-based point cloud alignment.
    - Modify vertex colors based on given RGB values.
    - Compare and merge two PLY files based on vertex positions and a distance threshold.

    The class works primarily with binary PLY files that contain elements such as vertices and their
    properties (positions, normals, colors, etc.), supporting operations like structured reading, writing,
    and transformation.

    Attributes:
        - format (str): The format of the PLY file (default: "binary_little_endian 1.0").
        - elements (dict): Dictionary storing element properties, formats, and binary data.
        - comments (list): List of comments found in the PLY file.
        - mappingTypes (dict): Mapping between PLY property types and their corresponding struct formats.
        - _inv_SH_C0 (float): Precomputed inverse of the first spherical harmonic coefficient used
                              for color modifications.
    """


    ####### init #######
    def __init__(self):
        """
        Initializes the SplatPLYHandler object. This function sets the default PLY format
        to 'binary_little_endian 1.0' and prepares the necessary data structures for handling 
        PLY file elements, their properties, and associated binary data.

        Initializes:
            - format: Default PLY format.
            - elements: Dictionary to store element properties (e.g., vertices, faces).
            - comments: List to store comments in the PLY file.
            - mapping: Mapping from PLY property types to struct formats for binary parsing.
            - _inv_SH_C0: Precomputed inverse of the first spherical harmonic coefficient.
        """

        self.format = "binary_little_endian 1.0"  # Default format for splat PLY
        self.elements = {}  # Dictionary to store elements and their properties
        self.comments = []  # List to store comments

        # Define the mapping between (PLY) property types and struct formats
        self.mappingTypes = {
            "char":   "c", # '<c' 1-byte character
            "uchar":  "B", # '<B' 1-byte unsigned char
            "short":  "h", # '<h' 2-byte short int
            "ushort": "H", # '<H' 2-byte unsigned short int
            "int":    "i", # '<i' 4-byte int
            "uint":   "I", # '<I' 4-byte unsigned int
            "float":  "f", # '<f' 4-byte single-precision float
            "double": "d", # '<d' 8-byte double-precision float
        }

        # Define the first spherical harmonic coefficient (degree 0, order 0) as constant to pre-compute the color.
        # This allow to avoid sending harmonics to the web worker or GPU, but removes view-dependent lighting effects like reflections.
        # If we were to use a degree > 0, we would need to recompute the color each time the camera moves, and send many more harmonics to the worker.
        # Degree 0: 1 harmonic needed (3 floats) per gaussian (basically just RGB, no view-dependant lighting)
        # Degree 1: 4 harmonics needed (12 floats) per gaussian
        # Degree 2: 9 harmonics needed (27 floats) per gaussian
        # Degree 3: 16 harmonics needed (48 floats) per gaussian
        self._inv_SH_C0 = 1 / 0.28209479177387814


    ####### load_ply #######
    def load_ply(self, filepath: str):
        """
        Loads a PLY file, parses its header, and reads its binary data into appropriate 
        structures for further manipulation.

        Args:
            filepath (str): Path to the PLY file to be loaded.

        Processes:
            - Parses the PLY header to extract format, comments, elements, and properties.
            - Reads the binary data for each element into numpy arrays.
            - Normalizes the properties for consistency.
        """

        with open(filepath, 'rb') as file:
            lines = []

            # Read header line by line until "end_header"
            while True:
                line = file.readline().decode('utf-8').strip()
                lines.append(line)
                if line == "end_header":
                    break

            # Parse the header to extract format, comments, and elements with their properties
            self._parse_header(lines)

            # Read binary data
            self._parse_binary_data(file)

            # Normalize properties (e.g. renaming and/or reordering)
            self._normalize_properties()


    ####### save_ply #######
    def save_ply(self, filepath: str):
        """
        Saves the current PLY data to a new PLY file.

        Args:
            filepath (str): Path to save the output PLY file. If the file already exists it
                            will be overwritten.
        
        Writes:
            - A new PLY header based on the updated properties.
            - Binary data for each element in the PLY file.
        """

        with open(filepath, 'wb') as file:

            # Write the header
            header_lines = ["ply", f"format {self.format}"]
            header_lines.extend(f"comment {comment}" for comment in self.comments)
            for element, properties in self.elements.items():
                header_lines.append(f"element {element} {properties['count']}")
                header_lines.extend(f"property {prop}" for prop in properties['properties'])
            header_lines.append("end_header")
            file.write("\n".join(header_lines).encode('utf-8') + b"\n")

            # Write binary data
            for element, properties in self.elements.items():
                element_format = '<' + ''.join(properties['formats'])
                if not element_format:
                    raise ValueError(f"No valid format found for element '{element}'")
                np.array(properties['data'], dtype=np.float32).tofile(file)


    ####### _parse_header #######
    def _parse_header(self, header_lines):
        """
        Parses the PLY file header to extract information about the elements, properties,
        and comments. It also prepares the internal data structures to store this information.

        Args:
            header_lines (list): List of lines from the PLY file header.
        """

        current_element = None
        for line in header_lines:
            if line.startswith("format"):
                self.format = line.split(" ", 1)[1]  # Store file format
            elif line.startswith("comment"):
                self.comments.append(line.split(" ", 1)[1])  # Store comments
            elif line.startswith("element"):
                parts = line.split()
                element_name = parts[1]
                element_count = int(parts[2])
                # Create a new entry for the new element
                self.elements[element_name] = {"count": element_count, "properties": [], "data": [], "formats": []}
                current_element = element_name
            elif line.startswith("property"):
                property_def = line.split(" ", 1)[1]
                self.elements[current_element]["properties"].append(property_def)

                # Update binary format for struct based on property type
                property_type = property_def.split()[0]
                fmt = self.mappingTypes.get(property_type, '')
                if fmt == '':
                    raise ValueError(f"Unsupported property type: {property_type}")
                self.elements[current_element]["formats"].append(fmt)


    ####### _parse_binary_data #######
    def _parse_binary_data(self, file):
        """
        Parses the binary data section of the PLY file and unpacks it according to the 
        previously extracted element formats. It stores the parsed data in the internal 
        structures.

        Args:
            file (file object): Opened PLY file for reading the binary data.
        """

        for element in self.elements:
            properties = self.elements[element]
            count = properties['count']
            element_format = '<' + ''.join(properties['formats'])
            element_size = struct.calcsize(element_format)
            binary_data = file.read(count * element_size)
            # Unpack binary data into numpy array
            properties['data'] = np.array(struct.unpack(f"<{count * len(properties['formats'])}f", binary_data)).reshape(count, -1)


    ####### _normalize_properties #######
    def _normalize_properties(self):
        """
        Normalizes the properties of the PLY elements (e.g., renaming and/or reordering properties).
        This is useful when elements may have different property names but represent the same data.

        Specifically, renames and reorders the following properties if needed:
        - normals: from (normal_x, normal_y, normal_z) to (nx, ny, nz).
        - colors: from (red/r, green/g, blue/b) to (f_dc_0, f_dc_1, f_dc_2).
        - opacity: from (alpha) to (opacity). This variable is not reordered.
        """

        # Function to reorder each columns
        # Note: it doesn't need to be efficient, since it's done only once and only for 3 elements
        def reorder(data, properties, formats, correct_indexes, wrong_indexes):
            permutation = []
            last_index = 0

            for i in range(len(properties)):
                if i in correct_indexes:
                    permutation.append(wrong_indexes[correct_indexes.index(i)])
                else:
                    permutation.append(last_index)
                    last_index += 1
                    while last_index in wrong_indexes:
                        last_index += 1
                    
            data[:] = data[:, permutation] # Reorder the data columns
            properties[:] = [properties[i] for i in permutation] # Reorder the properties
            formats[:] = [formats[i] for i in permutation] # Reorder the formats
            
        for element in self.elements:
            data = self.elements[element]['data']
            properties = self.elements[element]['properties']
            formats = self.elements[element]['formats']
            
            # Check if normals are present and ensure correct property naming
            if "float nx" not in properties:
                if "float normal_x" in properties: # For example because the file was edited using Blender
                    wrong_indexes = [properties.index("float normal_x"),
                                     properties.index("float normal_y"),
                                     properties.index("float normal_z")]
                    correct_indexes = [3, 4, 5]
                    
                    # Rename (normal_x, normal_y, normal_z) to (nx, ny, nz)
                    properties[wrong_indexes[0]] = "float nx"
                    properties[wrong_indexes[1]] = "float ny"
                    properties[wrong_indexes[2]] = "float nz"

                    # Reorder each columns if needed
                    if wrong_indexes != correct_indexes:
                        reorder(data, properties, formats, correct_indexes, wrong_indexes)
                else:
                    raise ValueError("Normals are missing or have different names which are not suppoted.")
                
            # Check if colors are present and ensure correct property naming
            if "float f_dc_0" not in properties:
                if "float red" in properties or "float r" in properties:
                    if "float red" in properties:
                        wrong_indexes = [properties.index("float red"),
                                         properties.index("float green"),
                                         properties.index("float blue")]
                    else:
                        wrong_indexes = [properties.index("float r"),
                                         properties.index("float g"),
                                         properties.index("float b")]
                    correct_indexes = [6, 7, 8]
                    
                    # Rename (red/r, green/g, blue/b) to (f_dc_0, f_dc_1, f_dc_2)
                    properties[wrong_indexes[0]] = "float f_dc_0"
                    properties[wrong_indexes[1]] = "float f_dc_1"
                    properties[wrong_indexes[2]] = "float f_dc_2"

                    # Reorder each columns if needed
                    if wrong_indexes != correct_indexes:
                        reorder(data, properties, formats, correct_indexes, wrong_indexes)
                else:
                    raise ValueError("Colors are missing or have different names which are not suppoted.")
            
            # Check if opacity is present and ensure correct property naming
            if "float opacity" not in properties:
                if "float alpha" in properties:
                    # Rename (alpha) to (opacity)
                    properties[properties.index("float alpha")] = "float opacity"
                else:
                    raise ValueError("Opacity is missing or has a different name which is not suppoted.")
            
            # Check other properties...


    ####### apply_transformation #######
    def apply_transformation(self,
                             matrix4d : np.ndarray = None,
                             rotation_matrix : np.ndarray = None,
                             translation_vector : np.array = None,
                             scale_factor : float = np.nan,
                             element : str = "vertex"):
        """
        Applies a transformation (rotation and/or translation) to the specified element in the PLY file.

        Args:
            - matrix4d (np.ndarray, optional): 4x4 transformation matrix for both rotation, translation and uniform scaling.
                                               In case rotation_matrix, translation_vector and/or scale_factor are provided
                                               then they will not be overridden by matrix4d.
            - rotation_matrix (np.ndarray, optional): 3x3 rotation matrix.
            - translation_vector (np.ndarray, optional): 3D translation vector.
            - scale_factor (float, optional): Uniform scaling factor.
            - element (str, optional): The element (e.g., "vertex", "faces") to apply the transformation to.
        
        Modifies:
            - Transforms positions (x, y, z) of the specified element and also normals (nx, ny, nz) and
              quaternions (rot_0, rot_1, rot_2, rot_3) if they are present.

        Raises:
            - ValueError: If none of the arguments matrix4d, rotation_matrix, translation_vector and scale_factor are provided.
            - ValueError: If the element value doesn't exist.
            - ValueError: If the scaling_factor value, when obtained by matrix4d, is non-uniform or equal to zero.
        """
        # Ensure that at least one of the arguments rotation_matrix, translation_vector and scale_factor must be provided
        if matrix4d is None and rotation_matrix is None and translation_vector is None and scale_factor is np.nan:
            raise ValueError("At least one of the arguments matrix4d, rotation_matrix, translation_vector and scale_factor must be provided")
        
        # Ensure there is a "vertex" element
        if element not in self.elements:
            raise ValueError(f"Element '{element}' not found in PLY file")
        
        # Check if matrix4d is provided and then extract the rotation, translation and scaling components
        if matrix4d is not None:

            if rotation_matrix is None: # Extract rotation_matrix (if it's None)
                rotation_matrix = matrix4d[:3, :3].copy()

            if translation_vector is None: # Extract translation_vector (if it's None)
                translation_vector = matrix4d[:3, 3].copy()

            if scale_factor is np.nan: # Extract scale_factor (if it's np.nan)
                extracted_scales = np.linalg.norm(rotation_matrix, axis=0)

                if np.allclose(extracted_scales, extracted_scales[0]):  # Check for uniform scale
                    scale_factor = extracted_scales[0]
                    if scale_factor <= 0.0 or np.isclose(scale_factor, 0.0): # Check for scale_factor greater than zero
                        raise ValueError(f"Not valid scaling detected: {scale_factor}")
                else:
                    raise ValueError(f"Non-uniform scaling detected: {extracted_scales}")
        
        # For better performance, avoid any computation that does not alter the data
        if np.allclose(matrix4d, np.identity(4)): # Avoid any transformation if matrix4d is an identity matrix
            matrix4d = None
        if np.allclose(rotation_matrix, np.identity(3)): # Avoid any rotation if rotation_matrix is an identity matrix
            rotation_matrix = None
        if np.allclose(translation_vector, np.zeros(3)): # Avoid any traslation if translation_vector is a zero vector
            translation_vector = None
        if np.isclose(scale_factor, 1.0): # Avoid any scaling if scale_factor is equal to 1.0
            scale_factor = np.nan

        data = self.elements[element]['data']
        properties = self.elements[element]['properties']
            
        # Apply rotation, translation and/or scaling to points (x, y, z)
        columns = [properties.index("float x"),
                   properties.index("float y"),
                   properties.index("float z")]
        if matrix4d is not None:
            points = data[:, columns]
            data[:, columns] = (np.hstack((points, np.ones((points.shape[0], 1)))) @ matrix4d.T)[:, :3]
        else:
            if scale_factor is not np.nan:
                data[:, columns] *= scale_factor

            if rotation_matrix is not None:
                data[:, columns] = data[:, columns] @ rotation_matrix.T
                
            if translation_vector is not None:
                data[:, columns] += translation_vector.reshape(1, 3)
            
        # Apply rotation to normals (nx, ny, nz)
        if rotation_matrix is not None and "float nx" in properties:
            columns = [properties.index("float nx"),
                       properties.index("float ny"),
                       properties.index("float nz")]
            data[:, columns] = data[:, columns] @ rotation_matrix.T

        # Apply scaling to scales (scale_0, scale_1, scale_2)
        if scale_factor is not np.nan and "float scale_0" in properties:
            columns = [properties.index("float scale_0"),
                       properties.index("float scale_1"),
                       properties.index("float scale_2")]
            data[:, columns] *= scale_factor
            
        # Apply rotation to quaternions (rot_0, rot_1, rot_2, rot_3)
        if rotation_matrix is not None and "float rot_0" in properties:
            columns = [properties.index("float rot_0"),
                       properties.index("float rot_1"),
                       properties.index("float rot_2"),
                       properties.index("float rot_3")]
            quaternions = data[:, columns]
            rotated_quaternions = (R.from_quat(quaternions) * R.from_matrix(rotation_matrix)).as_quat()
            data[:, columns] = rotated_quaternions


    ####### align_icp #######
    def align_icp(self,
                  other_handler : 'SplatPLYHandler',
                  voxel_sizes : list = [0.2, 0.1, 0.05],
                  max_iteration : int = 10000,
                  distance_threshold_icp : float = np.nan,
                  max_nn_normals : int = 30,
                  max_nn_fpfh : int = 100,
                  voxel_size_factor : float = 1.5,
                  normals_factor : float = 2.0,
                  fpfh_factor : float = 5.0,
                  downsample_voxels : bool = True,
                  nb_neighbors_outlier: int = 20,
                  std_ratio_outlier: float = 2.0,
                  transformation_delta_threshold: float = 1e-4
                  ) -> np.ndarray:
        """
        Method to compute the required transformation in order to align the vertices of this SplatPLYHandler (source)
        to another SplatPLYHandler (target) using ICP. This method builds Open3D point
        clouds from the "vertex" element in both handlers, computes features for an initial alignment using
        Fast Global Registration (FGR), then refines the alignment using ICP (point-to-plane).

        Args:
            - other_handler (SplatPLYHandler): The target handler to be aligned to this (source) handler.
            - voxel_sizes (list of float, optional): Voxel sizes used to compute iteratively normals and FPFH features.
                                                     (default is [0.2, 0.1, 0.05]).
            - max_iteration (int, optional): Maximum iterations for ICP (default is 10000).
            - distance_threshold_icp (float, optional): Maximum threshold for ICP. If np.nan then it's calculated as
                                                        voxel_size * voxel_size_factor.
            - max_nn_normals (int, optional): Max iterations for estimating normals. This is a pseudo-necessary
                                              step even if normals properties are avaiable (default is 30).
            - max_nn_fpfh (int, optional): Max iterations for calculate FPFH features (default is 100).
            - voxel_size_factor (float, optional): Factor used to determine the correspondence distance (default is 1.5).
            - normals_factor (float, optional): Factor used to aproximate normals (default is 2).
            - fpfh_factor (float, optional): Factor used to determine the FPFH features (default is 5).
            - downsample_voxels (bool, optional): If True, the voxels of both handler will be downsampled based on
                                                  voxel_size (default is True).
            - nb_neighbors_outlier (int, optional): Number of neighbors for statistical outlier removal (default is 20).
            - std_ratio_outlier (float, optional): Standard deviation ratio for outlier removal (default is 2.0).
            - transformation_delta_threshold (float, optional): If the change in transformation matrix is below this threshold,
                                                                the iteration stops early. Used only if voxel_sizes contains
                                                                more than one value (default is 1e-4).

        Returns:
            np.ndarray: The 4x4 transformation matrix computed by ICP.

        Raises:
            ValueError: If the element value doesn't exist in one of the two handlers.
        """
        # Ensure both handlers contain a "vertex" element
        if "vertex" not in self.elements or "vertex" not in other_handler.elements:
            raise ValueError("Both handlers must contain a 'vertex' element.")

        # Get the indices for x, y, z in the vertex properties
        properties1 = self.elements["vertex"]["properties"]
        properties2 = other_handler.elements["vertex"]["properties"]
        columns1 = [properties1.index("float x"),
                    properties1.index("float y"),
                    properties1.index("float z")]
        columns2 = [properties2.index("float x"),
                    properties2.index("float y"),
                    properties2.index("float z")]

        # Extract vertex positions from both handlers
        points1 = self.elements["vertex"]["data"][:, columns1]
        points2 = other_handler.elements["vertex"]["data"][:, columns2]

        # Homogenize points1 for transformation during the iteration(s)
        points1 = np.hstack((points1, np.ones((points1.shape[0], 1))))

        # Initialize the transformation matrix as identity
        current_transformation = np.identity(4)
        previous_transformation = np.copy(current_transformation)

        # Start the iteration process for ICP alignment
        for voxel_size in voxel_sizes:

            # Apply the current transformation to the source points for better initialization
            transformed_points1  = (points1 @ current_transformation.T)[:, :3]

            # Build Open3D point clouds for source and target
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(transformed_points1)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points2)

            # Downsample the point clouds
            if downsample_voxels:
                pcd1 = pcd1.voxel_down_sample(voxel_size)
                pcd2 = pcd2.voxel_down_sample(voxel_size)

            # Remove statistical outliers
            pcd1 = pcd1.remove_statistical_outlier(nb_neighbors=nb_neighbors_outlier, std_ratio=std_ratio_outlier)[0]
            pcd2 = pcd2.remove_statistical_outlier(nb_neighbors=nb_neighbors_outlier, std_ratio=std_ratio_outlier)[0]

            # Estimate normals for both point clouds (required for point-to-plane ICP)
            pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius = normals_factor * voxel_size,
                max_nn = max_nn_normals))
            pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius = normals_factor * voxel_size,
                max_nn = max_nn_normals))

            # Compute FPFH features for both point clouds using the method defined earlier
            fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(pcd1,
                                                                    o3d.geometry.KDTreeSearchParamHybrid(radius = fpfh_factor * voxel_size,
                                                                                                         max_nn = max_nn_fpfh))
            fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(pcd2,
                                                                    o3d.geometry.KDTreeSearchParamHybrid(radius = fpfh_factor * voxel_size,
                                                                                                         max_nn = max_nn_fpfh))

            # Fast Global Registration (FGR) for an initial alignment
            distance_threshold = voxel_size * voxel_size_factor
            result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                pcd1, pcd2, fpfh1, fpfh2,
                o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
            )

            # Refine alignment with ICP (point-to-plane)
            result_icp = o3d.pipelines.registration.registration_icp(
                pcd1, pcd2,
                (voxel_size * voxel_size_factor) if distance_threshold_icp is np.nan else distance_threshold_icp,
                result_fgr.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
            )

            # Update the current transformation
            current_transformation = result_icp.transformation @ current_transformation

            # Check for convergence: if the transformation change is below the threshold, break early.
            transformation_change = np.linalg.norm(current_transformation - previous_transformation)
            if transformation_change < transformation_delta_threshold:
                break

            previous_transformation = np.copy(current_transformation)

        return current_transformation


    ####### change_colors #######
    def change_colors(self,
                      rgb : list,
                      alpha : float = np.nan,
                      element : str = "vertex",
                      mode : int = 0,
                      indexes : list = []):
        """
        Changes the colors of the vertices in the PLY file based on the provided RGB 
        values and alpha (optional). Colors can be applied either globally to all vertices 
        or selectively based on the provided indexes.

        Args:
            - rgb (list of float): RGB color values as a list of three floats (0.0 to 1.0).
            - alpha (float, optional): Alpha value for transparency (0.0 to 1.0).
            - element (str, optional): The element to which the color change should be applied (default is "vertex").
            - mode (int, optional): Determines how the color should be applied (default is 0).
            - indexes (list of int, optional): A list of indices to specify which vertices to change.

        Raises:
            - ValueError: If the RGB values are out of range.
            - ValueError: If the alpha value is out of range.
            - ValueError: If the element value doesn't exist.
            - ValueError: If the indexes values are out of bound.
        """
        
        # Ensured RGB values (and Alpha value) are valid
        new_color = np.array(rgb)
        if np.any((new_color < 0.0) | (new_color > 1.0)):
            raise ValueError("RGB values must be between 0.0 and 1.0")
        if alpha is not np.nan and not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha value must be between 0.0 and 1.0")
        
        # Ensure there is a "vertex" element
        if element not in self.elements:
            raise ValueError(f"Element '{element}' not found in PLY file")
        
        # Ensure indexes does not contain an index out of bounds
        indexes = np.array(indexes)
        if indexes.size > 0 and self.elements[element]['count'] <= indexes.max():
            raise ValueError(f"Indexes cannot be greater than the number of element '{element}'")
            
        data = self.elements[element]['data']
        properties = self.elements[element]['properties']
            
        # Normalize f_dc components and rescale to new color
        color_columns = [properties.index("float f_dc_0"),
                         properties.index("float f_dc_1"),
                         properties.index("float f_dc_2")]
        
        # Mode 0 is the correct way to update the colors if we ignore the f_rest_* components (or harmonics)
        # Note: this is just an approximation
        if mode == 0:
            # To obtain the color the computation would be:
            #
            # color = [
            #     0.5 + SH_C0 * f_dc_0,
            #     0.5 + SH_C0 * f_dc_1,
            #     0.5 + SH_C0 * f_dc_2
            # ]
            #
            # So the inverse operation is the following
            new_color = (new_color - 0.5) * self._inv_SH_C0
            
            # Update all colors to the new color.
            # If color = data[:, color_columns] then there are two ways to update the colors:
            # - Method 1: Loop over all colors
            #
            # for i in range(colors.shape[0]):
            #     colors[i, :] = new_color
            #
            # - Method 2: Use broadcasting (more efficient)
            #
            # colors[:] = new_color
            #
            if indexes.size == 0:
                data[:, color_columns] = new_color
            else:
                data[indexes[:, None], color_columns] = new_color
        
        # Mode 1 update the colors in a "realistic" way
        elif mode == 1:
            # Normalize f_dc components and rescale to new color
            if indexes.size == 0:
                f_dc_norm = np.linalg.norm(data[:, color_columns], axis=1, keepdims=True)
                data[:, color_columns] = new_color * f_dc_norm
            else:
                f_dc_norm = np.linalg.norm(data[indexes[:, None], color_columns], axis=1, keepdims=True)
                data[indexes[:, None], color_columns] = new_color * f_dc_norm

        else:
            raise ValueError("Invalid mode. Choose between 0 and 1.")
        
        # Update the alpha channel (if provided)
        if alpha is not np.nan and "float opacity" in properties:
            alpha_column = properties.index("float opacity")
            if indexes.size == 0:
                data[:, alpha_column] = alpha
            else:
                data[indexes[:, None], alpha_column] = alpha


    ####### compare_and_merge #######
    def compare_and_merge(self,
                          other_handler : 'SplatPLYHandler',
                          element : str = "vertex",
                          threshold: float = 0.001,
                          rgb1 : list = [1.0, 0.0, 0.0],
                          rgb2 : list = [0.0, 1.0, 0.0],
                          alpha1 : float = np.nan,
                          alpha2 : float = np.nan,
                          mode=0
                          ) -> 'SplatPLYHandler':
        """
        Compares two SplatPLYHandler based on the positions of their vertices (or other elements) and
        merges them by keeping and coloring points that are different. 

        Args:
            - other_handler (SplatPLYHandler): Another instance of the SplatPLYHandler to compare with.
            - element (str, optional): The element to compare (default is "vertex").
            - threshold (float, optional): The distance threshold for considering points as mismatched.
            - rgb1 (list of float, optional): RGB color for missing/removed parts, values as a list of three
                                              floats from 0.0 to 1.0 (default is Red = [1.0, 0.0, 0.0]).
            - rgb2 (list of float, optional): RGB color for extra/added parts, values as a list of three
                                              floats from 0.0 to 1.0 (default is Green = [0.0, 1.0, 0.0]).
            - alpha1 (float, optional): Alpha value (0.0 to 1.0) for transparency of mismatched points
                                        (missing/removed parts) in the first PLY file.
            - alpha2 (float, optional): Alpha value (0.0 to 1.0) for transparency of mismatched points
                                        (extra/added parts) in the second PLY file.
            - mode (int, optional): Determines how to apply color changes (default is 0).
        
        Returns:
            SplatPLYHandler: A new instance of SplatPLYHandler with the merged data.

        Raises:
            - ValueError: If the rgb1 or rgb2 values are out of range.
            - ValueError: If the alpha1 or alpha2 value is out of range.
            - ValueError: If the element value doesn't exist in one of the two handlers.
            - ValueError: If the two handlers have different element properties.
        """
        # Ensured RGB values (and Alpha values) are valid
        if not (all(0.0 <= c <= 1.0 for c in rgb1) and all(0.0 <= c <= 1.0 for c in rgb2)):
            raise ValueError("The provided rgb1 or rgb2 values must be between 0.0 and 1.0")
        if any(a is not np.nan and not (0.0 <= a <= 1.0) for a in [alpha1, alpha2]):
            raise ValueError("The provided alpha1 or alpha2 value must be between 0.0 and 1.0")
        
        # Ensure both handlers contain a "vertex" element
        if "vertex" not in self.elements or "vertex" not in other_handler.elements:
            raise ValueError("Both handlers must contain a 'vertex' element.")
        
        # Ensure both handlers have identical properties for the specified element
        if self.elements[element]['properties'] != other_handler.elements[element]['properties']:
            raise ValueError(f"The two handlers must have the same properties for the '{element}' element")
        
        
        # Create the merged handler
        merged_handler = SplatPLYHandler()
        merged_handler.elements = self.elements.copy()

        # Retrieve data from the two handlers
        properties = self.elements[element]['properties']
        data1 = self.elements[element]['data']
        data2 = other_handler.elements[element]['data']

        columns = [properties.index("float x"),
                   properties.index("float y"),
                   properties.index("float z")]
        points1 = data1[:, columns]
        points2 = data2[:, columns]
        
        # Convert points in Open3D PointCloud format for spatial matching
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        
        # Build KDTree for points2
        tree2 = o3d.geometry.KDTreeFlann(pcd2)
        mismatch_indixes_p1 = [i for i in range(len(points1)) if tree2.search_knn_vector_3d(pcd1.points[i], 1)[2][0] > threshold]
        
        # Build KDTree for points1
        tree1 = o3d.geometry.KDTreeFlann(pcd1)
        mismatch_indixes_p2 = [i for i in range(len(points2)) if tree1.search_knn_vector_3d(pcd2.points[i], 1)[2][0] > threshold]

        # Merge the data (only add unmatched points from the second PLY in order to avoid duplicates)
        merged_data = np.concatenate((data1, data2[mismatch_indixes_p2, :]), axis=0)
        merged_handler.elements[element]['count'] = merged_data.shape[0]
        merged_handler.elements[element]['data'] = merged_data

        # Update colors of mismatched points (if needed)
        if len(mismatch_indixes_p1) > 0:
            merged_handler.change_colors(rgb = rgb1,
                                         alpha = alpha1,
                                         mode = mode,
                                         indexes = mismatch_indixes_p1)
        if len(mismatch_indixes_p2) > 0:
            merged_handler.change_colors(rgb = rgb2,
                                         alpha = alpha2,
                                         mode = mode,
                                         indexes = range(len(points1), merged_data.shape[0]))

        return merged_handler
