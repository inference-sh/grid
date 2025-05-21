from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os
import subprocess
import sysconfig  # Modern replacement for distutils.sysconfig
import sys

# build custom rasterizer
# build with `python setup.py install`
# nvcc is needed

def get_ext_filename():
    # Get the platform-specific extension suffix (e.g., .cpython-310-x86_64-linux-gnu.so)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    return f"custom_rasterizer_kernel{ext_suffix}"

ext_file = get_ext_filename()

# Check if the compiled extension exists and is newer than the source files
source_files = [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    'lib/custom_rasterizer_kernel/rasterizer_gpu.cu'
]

needs_rebuild = True
if os.path.exists(ext_file):
    ext_mtime = os.path.getmtime(ext_file)
    needs_rebuild = any(
        not os.path.exists(src) or os.path.getmtime(src) > ext_mtime 
        for src in source_files
    )

if needs_rebuild:
    print(f"Extension {ext_file} needs to be rebuilt, rebuilding...")
    custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [
        'lib/custom_rasterizer_kernel/rasterizer.cpp',
        'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
        'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
    ])

    setup(
        packages=find_packages(),
        version='0.1',
        name='custom_rasterizer',
        include_package_data=True,
        package_dir={'': '.'},
        ext_modules=[
            custom_rasterizer_module,
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
else:
    print(f"Extension {ext_file} is up to date, skipping setup")
    sys.exit(0)  # Exit successfully without running setup
