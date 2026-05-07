# Model: fal-ai/hyper3d/rodin/v2

## Endpoint
`fal-ai/hyper3d/rodin/v2`

## Category
image-to-3d

## Description
Rodin by Hyper3D generates realistic and production ready 3D models from text or images.

## Input Schema

### Optional Fields
- `prompt` (string, default: ): A textual prompt to guide model generation. Optional for Image-to-3D mode - if empty, AI will generate a prompt based on your images.
- `input_image_urls` (array): URL of images to use while generating the 3D model. Required for Image-to-3D mode. Up to 5 images allowed.
- `use_original_alpha` (boolean, default: False): When enabled, preserves the transparency channel from input images during 3D generation.
- `seed` (unknown): Seed value for randomization, ranging from 0 to 65535. Optional.
- `geometry_file_format` (string, default: glb): Format of the geometry file. Possible values: glb, usdz, fbx, obj, stl. Default is glb.
- `material` (string, default: All): Material type. PBR: Physically-based materials with realistic lighting. Shaded: Simple materials with baked lighting. All: Both types included.
- `quality_mesh_option` (string, default: 500K Triangle): Combined quality and mesh type selection. Quad = smooth surfaces, Triangle = detailed geometry. These corresponds to `mesh_mode` (if the option contains 'Triangle', mesh_mode is 'Raw', otherwise 'Quad') and `quality_override` (the numeric part of the option) parameters in Hyper3D API.
- `TAPose` (boolean, default: False): Generate characters in T-pose or A-pose format, making them easier to rig and animate in 3D software.
- `bbox_condition` (unknown): An array that specifies the bounding box dimensions [width, height, length].
- `addons` (unknown): The HighPack option will provide 4K resolution textures instead of the default 1K, as well as models with high-poly. It will cost **triple the billable units**.
- `preview_render` (boolean, default: False): Generate a preview render image of the 3D model along with the model files.

## Output Schema
- `model_mesh` (unknown): Generated 3D object file.
- `model_meshes` (array): Additional generated 3D object files returned by Hyper3D besides model_mesh.
- `seed` (integer): Seed value used for generation.
- `textures` (array): Generated textures for the 3D object.

## Notes
- [Add implementation notes here]
