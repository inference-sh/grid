#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""
Convert old-style inf.yml files (with variants) to new flat format.

Old format has:
  variants:
    default:
      resources: {...}
      python: "3.10"

New format has:
  kernel: python-3.10
  resources: {...}
  env: {}

Usage:
  # Preview conversion of a single file
  ./convert_inf_yml.py path/to/inf.yml --dry-run
  
  # Convert a single file in place
  ./convert_inf_yml.py path/to/inf.yml
  
  # Find all old-format files in a directory
  ./convert_inf_yml.py grid/native --find
  
  # Convert all old-format files (preview first)
  ./convert_inf_yml.py grid/native --all --dry-run
  
  # Convert all old-format files for real
  ./convert_inf_yml.py grid/native --all
"""

import sys
import yaml
from pathlib import Path


def convert_inf_yml(input_path: str, output_path: str = None, dry_run: bool = False) -> str:
    """Convert old inf.yml format to new flat format."""
    
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data:
        print(f"⚠️  Empty file: {input_path}")
        return ""
    
    # Check if already in new format
    if 'kernel' in data and 'variants' not in data:
        print(f"⏭️  Already converted: {input_path}")
        return ""
    
    # Must have variants with default
    if 'variants' not in data:
        print(f"⚠️  No variants found: {input_path}")
        return ""
    
    variants = data['variants']
    if 'default' not in variants:
        print(f"⚠️  No 'default' variant found: {input_path}")
        return ""
    
    default_variant = variants['default']
    
    # Build new structure - order matters for readability
    new_data = {}
    
    # Preserve namespace if it exists
    if 'namespace' in data:
        new_data['namespace'] = data['namespace']
    
    # Required fields
    new_data['name'] = data.get('name', '')
    new_data['description'] = data.get('description', '')
    new_data['category'] = data.get('category', '')
    
    # Convert python to kernel
    python_version = default_variant.get('python', '3.10')
    new_data['kernel'] = f"python-{python_version}"
    
    # Flatten resources from default variant
    resources = default_variant.get('resources', {})
    
    # Convert RAM from bytes to GB (if > 1000, assume bytes)
    ram_bytes = resources.get('ram', 8000000000)
    if isinstance(ram_bytes, (int, float)) and ram_bytes > 1000:
        ram_gb = int(ram_bytes // 1_000_000_000)
    else:
        ram_gb = ram_bytes
    
    gpu = resources.get('gpu', {})
    gpu_count = gpu.get('count', 0)
    
    # Determine GPU type for new format
    if gpu_count == 0:
        gpu_type = 'none'
        gpu_vram = 0
    else:
        gpu_type = gpu.get('type', 'any')
        gpu_vram = gpu.get('vram', 0)
        # Convert VRAM from bytes to GB (if > 1000, assume bytes)
        if isinstance(gpu_vram, (int, float)) and gpu_vram > 1000:
            gpu_vram = int(gpu_vram // 1_000_000_000)
    
    new_data['resources'] = {
        'gpu': {
            'count': gpu_count,
            'vram': gpu_vram,
            'type': gpu_type,
        },
        'ram': ram_gb,
    }
    
    # Preserve env from default variant or use empty dict
    new_data['env'] = default_variant.get('env', {})
    
    # Preserve secrets if they exist
    if 'secrets' in data:
        new_data['secrets'] = data['secrets']
    
    # Preserve images
    new_data['images'] = data.get('images', {'card': '', 'thumbnail': '', 'banner': ''})
    
    # Preserve metadata
    new_data['metadata'] = data.get('metadata', {})
    
    # Generate YAML output
    output = yaml.dump(new_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    if dry_run:
        print(f"\n📄 Would convert: {input_path}")
        print("-" * 60)
        print(output)
        print("-" * 60)
    else:
        out_file = output_path or input_path
        with open(out_file, 'w') as f:
            f.write(output)
        print(f"✅ Converted: {input_path} -> {out_file}")
    
    return output


def find_old_inf_files(base_dir: str) -> list:
    """Find all inf.yml files that have the old variants format."""
    old_files = []
    
    for inf_path in Path(base_dir).rglob('inf.yml'):
        try:
            with open(inf_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data and 'variants' in data and 'kernel' not in data:
                old_files.append(str(inf_path))
        except Exception as e:
            print(f"⚠️  Error reading {inf_path}: {e}")
    
    return old_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert old inf.yml format (with variants) to new flat format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview conversion of a single file
  %(prog)s path/to/inf.yml --dry-run
  
  # Convert a single file in place
  %(prog)s path/to/inf.yml
  
  # Find all old-format files in a directory
  %(prog)s grid/native --find
  
  # Convert all old-format files (preview first)
  %(prog)s grid/native --all --dry-run
  
  # Convert all old-format files for real
  %(prog)s grid/native --all
"""
    )
    parser.add_argument('path', nargs='?', help='Path to inf.yml file or directory to scan')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--find', '-f', action='store_true', help='Find all old-format inf.yml files in directory')
    parser.add_argument('--all', '-a', action='store_true', help='Convert all old-format files found in directory')
    
    args = parser.parse_args()
    
    if not args.path:
        parser.print_help()
        sys.exit(1)
    
    path = Path(args.path)
    
    if args.find:
        if not path.is_dir():
            print(f"Error: {path} is not a directory")
            sys.exit(1)
        
        old_files = find_old_inf_files(str(path))
        print(f"\n📂 Found {len(old_files)} old-format inf.yml files:\n")
        for f in sorted(old_files):
            print(f"  {f}")
        return
    
    if args.all:
        if not path.is_dir():
            print(f"Error: {path} is not a directory")
            sys.exit(1)
        
        old_files = find_old_inf_files(str(path))
        print(f"\n🔄 {'Would convert' if args.dry_run else 'Converting'} {len(old_files)} files...\n")
        
        for f in sorted(old_files):
            convert_inf_yml(f, dry_run=args.dry_run)
        return
    
    # Single file conversion
    if path.is_file():
        convert_inf_yml(str(path), dry_run=args.dry_run)
    else:
        print(f"Error: {path} is not a file. Use --find or --all for directories.")
        sys.exit(1)


if __name__ == '__main__':
    main()
