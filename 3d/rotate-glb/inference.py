from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional, Tuple
import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tempfile
import os
import io
from pygltflib import GLTF2, UNSIGNED_SHORT, UNSIGNED_INT, UNSIGNED_BYTE, BYTE
import base64

class Color(BaseModel):
    r: Optional[int] = Field(default=255, gte=0, lte=255, description="R") 
    g: Optional[int] = Field(default=255, gte=0, lte=255, description="G")
    b: Optional[int] = Field(default=255, gte=0, lte=255, description="B")

class AppInput(BaseAppInput):
    glb_file: File = Field(description="The input GLB/GLTF file to render")
    rotation_x: float = Field(default=0.0, description="Rotation around X-axis in degrees")
    rotation_y: float = Field(default=0.0, description="Rotation around Y-axis in degrees") 
    rotation_z: float = Field(default=0.0, description="Rotation around Z-axis in degrees")
    resolution_width: int = Field(default=1024, description="Output image width in pixels")
    resolution_height: int = Field(default=1024, description="Output image height in pixels")
    camera_distance: float = Field(default=3.0, description="Distance of camera from the object")
    camera_elevation: float = Field(default=30.0, description="Camera elevation angle in degrees")
    camera_azimuth: float = Field(default=45.0, description="Camera azimuth angle in degrees")
    background_color: Optional[Color] = Field(default=None, description="Background color as RGB tuple (0-255)")
    wireframe: bool = Field(default=False, description="Render as wireframe instead of solid")
    mesh_color: Optional[Color] = Field(default=None, description="Mesh color as RGB tuple (0-255)")
    

class AppOutput(BaseAppOutput):
    rendered_image: File = Field(description="The rendered image of the GLB model with applied rotation")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize matplotlib for headless rendering."""
        # Use Agg backend for headless rendering
        plt.switch_backend('Agg')

    def _degrees_to_radians(self, degrees):
        """Convert degrees to radians."""
        return np.radians(degrees)

    def _create_rotation_matrix(self, rx, ry, rz):
        """Create a rotation matrix from euler angles in degrees."""
        rx_rad = self._degrees_to_radians(rx)
        ry_rad = self._degrees_to_radians(ry)
        rz_rad = self._degrees_to_radians(rz)
        
        # Create rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx_rad), -np.sin(rx_rad)],
                       [0, np.sin(rx_rad), np.cos(rx_rad)]])
        
        Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                       [0, 1, 0],
                       [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
        
        Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                       [np.sin(rz_rad), np.cos(rz_rad), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix (order: Z, Y, X)
        return Rz @ Ry @ Rx

    def _render_mesh_matplotlib(self, mesh, input_data):
        """Render mesh using matplotlib 3D plotting, preserving mesh colors if present."""
        
        # Create figure with specified resolution
        dpi = 100
        fig_width = input_data.resolution_width / dpi
        fig_height = input_data.resolution_height / dpi
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        bg_color = np.array([input_data.background_color.r, input_data.background_color.g, input_data.background_color.b]) / 255.0
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Get mesh data
        vertices = mesh.vertices
        faces = mesh.faces
        
        face_colors = None
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            # Try per-face colors
            if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) == len(faces):
                # face_colors is (N, 4) RGBA in 0-255
                face_colors = mesh.visual.face_colors[:, :3] / 255.0  # ignore alpha for matplotlib
            # Try per-vertex colors
            elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(vertices):
                # For each face, average the vertex colors
                vcols = mesh.visual.vertex_colors[:, :3] / 255.0
                face_colors = np.array([vcols[face].mean(axis=0) for face in faces])
        
        if input_data.wireframe:
            # Render wireframe
            for face in faces:
                face_vertices = vertices[face]
                face_vertices = np.vstack([face_vertices, face_vertices[0]])
                ax.plot(face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2], 'k-', linewidth=0.5)
        else:
            # Render solid mesh using Poly3DCollection
            if face_colors is not None:
                poly3d = [[vertices[j] for j in face] for face in faces]
                collection = Poly3DCollection(poly3d, alpha=0.95)
                collection.set_facecolor(face_colors)
                collection.set_edgecolor('black')
                collection.set_linewidth(0.1)
                ax.add_collection3d(collection)
            else:
                mesh_color = np.array([input_data.mesh_color.r, input_data.mesh_color.g, input_data.mesh_color.b]) / 255.0
                poly3d = [[vertices[j] for j in face] for face in faces]
                collection = Poly3DCollection(poly3d, alpha=0.8)
                collection.set_facecolor(mesh_color)
                collection.set_edgecolor('black')
                collection.set_linewidth(0.1)
                ax.add_collection3d(collection)
        
        # Set camera view
        ax.view_init(elev=input_data.camera_elevation, azim=input_data.camera_azimuth)
        
        # Set equal aspect ratio and limits
        max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                              vertices[:, 1].max() - vertices[:, 1].min(),
                              vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        range_val = max_range * input_data.camera_distance / 2.0
        ax.set_xlim(mid_x - range_val, mid_x + range_val)
        ax.set_ylim(mid_y - range_val, mid_y + range_val)
        ax.set_zlim(mid_z - range_val, mid_z + range_val)
        
        # Hide axes for cleaner look
        ax.set_axis_off()
        
        # Remove padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        return fig

    def extract_mesh_and_colors_from_gltf(self, glb_path):
        gltf = GLTF2().load(glb_path)
        mesh = gltf.meshes[0]
        primitive = mesh.primitives[0]

        def get_buffer_data(bufferView, accessor):
            buffer = gltf.buffers[bufferView.buffer]
            if buffer.uri:
                if buffer.uri.startswith("data:"):
                    data = base64.b64decode(buffer.uri[len("data:"):])
                else:
                    with open(buffer.uri, "rb") as f:
                        data = f.read()
            elif hasattr(buffer, 'data') and buffer.data is not None:
                data = buffer.data
            else:
                # Read from the GLB file's binary chunk
                with open(glb_path, "rb") as f:
                    f.seek(0)
                    header = f.read(12)
                    json_length = int.from_bytes(f.read(4), "little")
                    f.read(4)  # skip JSON chunk type
                    f.read(json_length)  # skip JSON chunk
                    bin_length = int.from_bytes(f.read(4), "little")
                    f.read(4)  # skip BIN chunk type
                    bin_data = f.read(bin_length)
                    data = bin_data
            start = bufferView.byteOffset or 0
            end = start + (bufferView.byteLength or 0)
            return data[start:end]

        # Get positions
        pos_accessor = gltf.accessors[primitive.attributes.POSITION]
        pos_bufferView = gltf.bufferViews[pos_accessor.bufferView]
        pos_data = get_buffer_data(pos_bufferView, pos_accessor)
        positions = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 3)

        # Get indices
        idx_accessor = gltf.accessors[primitive.indices]
        idx_bufferView = gltf.bufferViews[idx_accessor.bufferView]
        idx_data = get_buffer_data(idx_bufferView, idx_accessor)
        if idx_accessor.componentType == UNSIGNED_SHORT:
            indices = np.frombuffer(idx_data, dtype=np.uint16)
        elif idx_accessor.componentType == UNSIGNED_INT:
            indices = np.frombuffer(idx_data, dtype=np.uint32)
        elif idx_accessor.componentType == UNSIGNED_BYTE:
            indices = np.frombuffer(idx_data, dtype=np.uint8)
        elif idx_accessor.componentType == BYTE:
            indices = np.frombuffer(idx_data, dtype=np.int8)
        else:
            raise ValueError("Unsupported index component type")
        faces = indices.reshape(-1, 3)

        # Get vertex colors if present
        colors = None
        if "COLOR_0" in primitive.attributes:
            col_accessor = gltf.accessors[primitive.attributes["COLOR_0"]]
            col_bufferView = gltf.bufferViews[col_accessor.bufferView]
            col_data = get_buffer_data(col_bufferView, col_accessor)
            colors = np.frombuffer(col_data, dtype=np.float32).reshape(-1, 3)  # RGB, 0-1

        return positions, faces, colors

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Render the GLB file with the specified rotation using pygltflib for mesh/color extraction."""
        
        # Check if input file exists
        if not input_data.glb_file.exists():
            raise RuntimeError(f"GLB file does not exist at path: {input_data.glb_file.path}")
        
        try:
            # Use pygltflib to extract mesh and color data
            positions, faces, colors = self.extract_mesh_and_colors_from_gltf(input_data.glb_file.path)

            # Apply rotation
            rotation_matrix = self._create_rotation_matrix(
                input_data.rotation_x, 
                input_data.rotation_y, 
                input_data.rotation_z
            )
            positions = (positions @ rotation_matrix.T)

            # Center and scale
            centroid = positions.mean(axis=0)
            positions -= centroid
            max_extent = np.max(positions.max(axis=0) - positions.min(axis=0))
            if max_extent > 0:
                positions *= (2.0 / max_extent)

            # Render using matplotlib
            fig = plt.figure(figsize=(input_data.resolution_width / 100, input_data.resolution_height / 100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            bg_color = np.array([input_data.background_color.r if input_data.background_color else 255,
                                 input_data.background_color.g if input_data.background_color else 255,
                                 input_data.background_color.b if input_data.background_color else 255]) / 255.0
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)

            poly3d = [positions[face] for face in faces]
            if colors is not None:
                face_colors = [colors[face].mean(axis=0) for face in faces]
                collection = Poly3DCollection(poly3d, facecolors=face_colors, edgecolor='k', linewidth=0.1, alpha=0.95)
            else:
                mesh_color = np.array([input_data.mesh_color.r if input_data.mesh_color else 128,
                                       input_data.mesh_color.g if input_data.mesh_color else 128,
                                       input_data.mesh_color.b if input_data.mesh_color else 128]) / 255.0
                collection = Poly3DCollection(poly3d, facecolor=mesh_color, edgecolor='k', linewidth=0.1, alpha=0.95)
            ax.add_collection3d(collection)
            ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
            ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
            ax.set_zlim(positions[:, 2].min(), positions[:, 2].max())
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            buf.seek(0)
            image = Image.open(buf)
            if image.size != (input_data.resolution_width, input_data.resolution_height):
                image = image.resize((input_data.resolution_width, input_data.resolution_height), Image.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output_path = temp_file.name
            image.save(output_path, "PNG")
            plt.close(fig)
            buf.close()
            return AppOutput(rendered_image=File(path=output_path))
        except Exception as e:
            raise RuntimeError(f"Error rendering GLB file with pygltflib: {str(e)}")

    async def unload(self):
        """Clean up resources."""
        pass