# Python Code Executor

Execute Python code with access to popular libraries for data processing, web scraping, and automation.

## Features

- **CPU-only environment** - No GPU dependencies
- **Comprehensive library support** - Popular libraries pre-installed
- **Web scraping tools** - requests, BeautifulSoup, Selenium, Scrapy, Playwright
- **Data processing** - NumPy, Pandas, Matplotlib, Seaborn, Plotly
- **Image & SVG generation** - Create plots, graphics, PDFs, and images
- **Video processing & creation** - MoviePy, PyAV, FFmpeg, VideoGear
- **3D model manipulation** - Trimesh, Open3D, mesh I/O, various 3D formats
- **Automatic file return** - Files saved during execution are automatically returned
- **Safe execution** - Timeout controls and isolated subprocess execution
- **Rich output capture** - stdout, stderr, execution time, and generated files

## Included Libraries

### Web Scraping & HTTP
- **requests** (>=2.31.0) - HTTP requests library
- **beautifulsoup4** (>=4.12.0) - HTML/XML parsing
- **lxml** (>=4.9.0) - XML/HTML processing
- **html5lib** (>=1.1) - HTML5 parser
- **selenium** (>=4.10.0) - Browser automation
- **playwright** (>=1.40.0) - Modern browser automation
- **scrapy** (>=2.11.0) - Web scraping framework
- **httpx** (>=0.24.0) - Async HTTP client
- **aiohttp** (>=3.8.0) - Async HTTP client/server

### Data Processing & Analysis
- **numpy** (>=1.24.0) - Numerical computing
- **pandas** (>=2.0.0) - Data analysis and manipulation
- **scipy** (>=1.11.0) - Scientific computing
- **matplotlib** (>=3.7.0) - Data visualization
- **seaborn** (>=0.12.0) - Statistical visualization
- **plotly** (>=5.14.0) - Interactive plotting

### Data Formats & Serialization
- **pyyaml** (>=6.0) - YAML parser
- **toml** (>=0.10.2) - TOML parser
- **jsonlines** (>=3.1.0) - JSON Lines format
- **xmltodict** (>=0.13.0) - XML to dict conversion
- **openpyxl** (>=3.1.0) - Excel file handling
- **xlrd** (>=2.0.1) - Excel reading
- **pypdf2** (>=3.0.0) - PDF manipulation

### Database & Storage
- **sqlalchemy** (>=2.0.0) - SQL toolkit and ORM
- **redis** (>=4.5.0) - Redis client
- **pymongo** (>=4.3.0) - MongoDB client

### Text Processing & NLP
- **nltk** (>=3.8.0) - Natural language toolkit
- **textblob** (>=0.17.0) - Text processing
- **regex** (>=2023.6.3) - Advanced regex

### Date/Time Utilities
- **python-dateutil** (>=2.8.2) - Date extensions
- **pytz** (>=2023.3) - Timezone library

### Authentication & APIs
- **python-dotenv** (>=1.0.0) - Environment variables
- **pyjwt** (>=2.8.0) - JSON Web Tokens
- **oauth2client** (>=4.1.3) - OAuth 2.0 client

### Image Processing (CPU only)
- **opencv-python-headless** (>=4.8.0) - Computer vision library
- **pillow** (>=10.0.0) - Image manipulation
- **scikit-image** (>=0.21.0) - Image processing algorithms
- **imageio** (>=2.31.0) - Image reading/writing
- **imageio-ffmpeg** (>=0.4.9) - Video handling

### Video Processing & Creation
- **moviepy** (1.0.3) - Video editing and composition (pinned for stability)
- **av** (>=11.0.0) - PyAV - Advanced video/audio processing
- **ffmpeg-python** (>=0.2.0) - Python bindings for FFmpeg
- **vidgear** (>=0.3.2) - High-performance video processing
- **pydub** (>=0.25.1) - Audio manipulation

### 3D Processing & Manipulation
- **trimesh** (>=4.0.0) - Loading and processing 3D meshes
- **open3d** (>=0.18.0) - 3D data processing and visualization
- **pywavefront** (>=1.3.3) - Wavefront OBJ file loader
- **PyMCubes** (>=0.1.4) - Marching cubes for 3D reconstruction
- **plyfile** (>=1.0.0) - PLY format support
- **pygltflib** (>=1.16.0) - GLTF/GLB model format
- **numpy-stl** (>=3.1.0) - STL file reading and writing
- **meshio** (>=5.3.0) - I/O for various mesh formats
- **pyvista** (>=0.43.0) - 3D plotting and mesh analysis

### SVG & Graphics Generation
- **svgwrite** (>=1.4.3) - SVG creation
- **cairosvg** (>=2.7.0) - SVG to PNG/PDF conversion
- **svgpathtools** (>=1.6.1) - SVG path manipulation
- **reportlab** (>=4.0.0) - PDF generation
- **pdf2image** (>=1.16.3) - PDF to image conversion

### Scientific Computing & ML
- **scikit-learn** (>=1.3.0) - Machine learning
- **statsmodels** (>=0.14.0) - Statistical modeling

### CLI & Utilities
- **tqdm** (>=4.65.0) - Progress bars
- **colorama** (>=0.4.6) - Terminal colors
- **tabulate** (>=0.9.0) - Pretty tables
- **rich** (>=13.4.0) - Rich text and formatting
- **click** (>=8.1.0) - CLI creation
- **argparse** (>=1.4.0) - Command-line parsing

### Testing
- **pytest** (>=7.4.0) - Testing framework

## Usage

### Input Schema

```json
{
  "code": "print('Hello World!')",
  "timeout": 30,
  "capture_output": true,
  "working_dir": null
}
```

### Parameters

- `code` (required): Python code to execute
- `timeout` (optional): Execution timeout in seconds (1-300, default: 30)
- `capture_output` (optional): Whether to capture output (default: true)
- `working_dir` (optional): Working directory for execution (default: temp dir)

### Output Schema

```json
{
  "stdout": "Output from code execution",
  "stderr": "Error output if any",
  "exit_code": 0,
  "execution_time": 1.23,
  "error": null,
  "files": []
}
```

- `files`: Optional list of output files generated by the code (empty by default)

## Returning Generated Files

The Python executor automatically detects and returns any files created during code execution. You don't need to manually handle file returns - just save files to the appropriate location.

### Where to Save Files

You can save files in two locations:

1. **Current working directory** - Files saved in the current directory or any subdirectory will be automatically detected
2. **`outputs/` subdirectory** - A pre-created `outputs/` directory is available for convenience (recommended)

### How It Works

1. Before your code runs, the executor takes a snapshot of existing files
2. Your code executes and creates/saves files
3. After execution, the executor detects all new files
4. New files are automatically added to the `files` output array

### File Saving Examples

#### Save a matplotlib plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')

# Save to outputs directory (recommended)
plt.savefig('outputs/sine_wave.png')
print("Plot saved!")
```

#### Generate and save an SVG file

```python
import svgwrite

# Create an SVG drawing
dwg = svgwrite.Drawing('outputs/circle.svg', size=('200px', '200px'))
dwg.add(dwg.circle(center=(100, 100), r=50, fill='blue', stroke='black'))

# Save the SVG
dwg.save()
print("SVG created!")
```

#### Create multiple output files

```python
from PIL import Image
import numpy as np

# Create multiple images with different sizes
for size in [100, 200, 300]:
    # Create a gradient image
    array = np.linspace(0, 255, size*size, dtype=np.uint8).reshape(size, size)
    img = Image.fromarray(array, mode='L')

    # Save to outputs directory
    img.save(f'outputs/gradient_{size}x{sizex}.png')
    print(f"Created {size}x{size} image")

print("All images created!")
```

#### Generate a PDF report

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Create a PDF
c = canvas.Canvas('outputs/report.pdf', pagesize=letter)
c.drawString(100, 750, "Sample PDF Report")
c.drawString(100, 700, "Generated by Python Code Executor")
c.save()

print("PDF report created!")
```

#### Save data files

```python
import pandas as pd
import json

# Create sample data
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Tokyo']
}

df = pd.DataFrame(data)

# Save as CSV
df.to_csv('outputs/data.csv', index=False)

# Save as JSON
with open('outputs/data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Save as Excel
df.to_excel('outputs/data.xlsx', index=False)

print("Data files saved!")
```

### Best Practices

- **Use the `outputs/` directory** - It's pre-created and keeps your files organized
- **Use descriptive filenames** - Makes it easier to identify files in the output
- **Avoid creating too many files** - Be mindful of the total size and count
- **Print confirmation messages** - Let users know what files were created

### Important Notes

- Hidden files (starting with `.`) are not returned
- Python cache files (`.pyc`) are excluded automatically
- The `working_dir` parameter allows you to specify a custom working directory
- All detected files are returned in the `files` array of the output

## Examples

### Simple Calculation

```python
import numpy as np
import pandas as pd

arr = np.array([1, 2, 3, 4, 5])
print(f"Sum: {arr.sum()}")
print(f"Mean: {arr.mean()}")

df = pd.DataFrame({"values": arr})
print(df.describe())
```

### Web Scraping

```python
import requests
from bs4 import BeautifulSoup

response = requests.get("https://example.com")
soup = BeautifulSoup(response.content, "html.parser")
title = soup.find("title").text
print(f"Page title: {title}")
```

### Data Processing with Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000, 60000, 70000]
}
df = pd.DataFrame(data)

print(df.describe())
print(f"\nAverage salary: ${df['salary'].mean():,.2f}")

# Create a bar chart
plt.figure(figsize=(8, 6))
plt.bar(df['name'], df['salary'])
plt.title('Salary by Person')
plt.xlabel('Name')
plt.ylabel('Salary ($)')
plt.tight_layout()

# Save the plot (will be automatically returned)
plt.savefig('outputs/salary_chart.png')
print("\nChart saved to outputs/salary_chart.png")
```

### Video Processing

```python
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import numpy as np

# Create a simple video with text overlay
# Note: For this example, we'll create a color clip
from moviepy.editor import ColorClip

# Create a 5-second red video
duration = 5
clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=duration)

# Add text overlay
txt_clip = TextClip("Hello World!", fontsize=70, color='white')
txt_clip = txt_clip.set_position('center').set_duration(duration)

# Composite the video
video = CompositeVideoClip([clip, txt_clip])

# Export the video
video.write_videofile('outputs/hello_video.mp4', fps=24)
print("Video created!")
```

### 3D Model Processing

```python
import trimesh
import numpy as np

# Create a simple 3D mesh (sphere)
sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

# Apply transformations
sphere.apply_transform(trimesh.transformations.rotation_matrix(
    angle=np.radians(45),
    direction=[0, 1, 0],
    point=sphere.centroid
))

# Export to various formats
sphere.export('outputs/sphere.obj')
sphere.export('outputs/sphere.stl')
sphere.export('outputs/sphere.ply')

print(f"Created sphere with {len(sphere.vertices)} vertices")
print(f"Exported to OBJ, STL, and PLY formats")
```

### Audio Manipulation

```python
from pydub import AudioGenerator
from pydub.generators import Sine

# Generate a 440 Hz sine wave (A note) for 3 seconds
sine_wave = Sine(440).to_audio_segment(duration=3000)

# Apply fade in/out effects
sine_wave = sine_wave.fade_in(1000).fade_out(1000)

# Export as MP3 and WAV
sine_wave.export('outputs/tone.mp3', format='mp3')
sine_wave.export('outputs/tone.wav', format='wav')

print("Audio files created!")
```

## Deployment

```bash
# Test locally
infsh run input.json

# Deploy to production
infsh deploy
```

## Variants

- **default**: 8GB RAM for standard scripts
- **high_memory**: 16GB RAM for larger datasets

## Notes

- This is a CPU-only app - no GPU/ML libraries included
- Code runs in isolated subprocess for safety
- Timeout enforced to prevent infinite loops
- All popular Python libraries pre-installed
- Non-interactive matplotlib backend (use `plt.savefig()` instead of `plt.show()`)
- Files saved to the working directory are automatically detected and returned
- Use the pre-created `outputs/` directory for saving generated files
- Hidden files and Python cache files are automatically excluded from results
- Video processing uses CPU-based encoding (may be slower than GPU)
- 3D models can be loaded, manipulated, and exported in various formats (OBJ, STL, PLY, GLTF, etc.)
- FFmpeg is available for video/audio processing
