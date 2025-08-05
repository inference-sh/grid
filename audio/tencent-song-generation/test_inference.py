#!/usr/bin/env python3
"""
Test script for the Song Generation Inference App

This script demonstrates how to use the inference app with various input configurations.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to the path to import the inference module
sys.path.append(os.path.dirname(__file__))

from inference import App, AppInput

async def test_basic_generation():
    """Test basic song generation with lyrics only."""
    print("=== Test 1: Basic Lyrics Generation ===")
    
    app = App()
    
    # Mock metadata for testing
    metadata = {"app_variant": "default"}
    
    # Initialize the app
    await app.setup(metadata)
    
    # Create test input
    input_data = AppInput(
        lyrics="[intro-short] ; [verse] 雪花舞动在无尽的天际.情缘如同雪花般轻轻逝去.希望与真挚.永不磨灭.你的忧虑.随风而逝 ; [chorus] 我怀抱着守护这片梦境.在这世界中寻找爱与虚幻.苦辣酸甜.我们一起品尝.在雪的光芒中.紧紧相拥 ; [outro-short]",
        descriptions="female, dark, pop, sad, piano and drums",
        generate_type="mixed",
        temperature=0.9,
        max_duration=10
    )
    
    try:
        # Run inference
        result = await app.run(input_data, metadata)
        
        print(f"Generation completed successfully!")
        print(f"Generated audio: {result.generated_audio.path}")
        print(f"Generation info:\n{result.generation_info}")
        
        # Check if files exist
        if os.path.exists(result.generated_audio.path):
            print(f"✓ Generated audio file exists: {os.path.getsize(result.generated_audio.path)} bytes")
        else:
            print("✗ Generated audio file not found")
            
    except Exception as e:
        print(f"Error during generation: {e}")
    
    finally:
        # Clean up
        await app.unload()

async def test_separate_generation():
    """Test separate vocal and BGM generation."""
    print("\n=== Test 2: Separate Generation ===")
    
    app = App()
    metadata = {"app_variant": "default"}
    
    await app.setup(metadata)
    
    input_data = AppInput(
        lyrics="[verse] 花朵绽放如诗篇.随风轻舞动.湖面波光映倒影.心随波浪荡漾中.感受着这美好时光.仿佛梦境般飘渺 ; [chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅.阳光洒满湖面.一切都如此美好",
        descriptions="female, bright, pop, happy, guitar and drums",
        generate_type="separate",
        temperature=0.8,
        max_duration=8
    )
    
    try:
        result = await app.run(input_data, metadata)
        
        print(f"Separate generation completed!")
        print(f"Main audio: {result.generated_audio.path}")
        
        if result.vocal_audio:
            print(f"Vocal audio: {result.vocal_audio.path}")
            if os.path.exists(result.vocal_audio.path):
                print(f"✓ Vocal file exists: {os.path.getsize(result.vocal_audio.path)} bytes")
        
        if result.bgm_audio:
            print(f"BGM audio: {result.bgm_audio.path}")
            if os.path.exists(result.bgm_audio.path):
                print(f"✓ BGM file exists: {os.path.getsize(result.bgm_audio.path)} bytes")
                
    except Exception as e:
        print(f"Error during separate generation: {e}")
    
    finally:
        await app.unload()

async def test_auto_prompt():
    """Test generation with auto prompt type."""
    print("\n=== Test 3: Auto Prompt Generation ===")
    
    app = App()
    metadata = {"app_variant": "default"}
    
    await app.setup(metadata)
    
    input_data = AppInput(
        lyrics="[verse] 月光洒满大地.星星在天空中闪烁.我们手牵手走过.这美好的夜晚.心中充满温暖.爱意如潮水般涌来 ; [chorus] 让我们一起歌唱.在这美丽的夜晚.让爱永远陪伴.直到永远",
        auto_prompt_type="Pop",
        generate_type="mixed",
        temperature=0.7,
        max_duration=12
    )
    
    try:
        result = await app.run(input_data, metadata)
        
        print(f"Auto prompt generation completed!")
        print(f"Generated audio: {result.generated_audio.path}")
        print(f"Generation info:\n{result.generation_info}")
        
    except Exception as e:
        print(f"Error during auto prompt generation: {e}")
    
    finally:
        await app.unload()

async def test_different_styles():
    """Test generation with different style descriptions."""
    print("\n=== Test 4: Different Styles ===")
    
    styles = [
        ("Jazz", "male, smooth, jazz, romantic, saxophone and piano"),
        ("Rock", "male, powerful, rock, energetic, electric guitar and drums"),
        ("Classical", "orchestral, classical, peaceful, strings and woodwinds"),
        ("Electronic", "electronic, dance, upbeat, synthesizer and beats")
    ]
    
    app = App()
    metadata = {"app_variant": "default"}
    
    await app.setup(metadata)
    
    base_lyrics = "[verse] 时光飞逝如流水.岁月如歌般悠扬.我们在这人生路上.寻找着属于自己的方向 ; [chorus] 让我们一起前行.在这美好的时光里.让梦想照亮前方.让希望永远不灭"
    
    for style_name, style_desc in styles:
        print(f"\n--- Testing {style_name} style ---")
        
        input_data = AppInput(
            lyrics=base_lyrics,
            descriptions=style_desc,
            generate_type="mixed",
            temperature=0.9,
            max_duration=6
        )
        
        try:
            result = await app.run(input_data, metadata)
            print(f"✓ {style_name} generation completed: {os.path.basename(result.generated_audio.path)}")
            
        except Exception as e:
            print(f"✗ {style_name} generation failed: {e}")
    
    await app.unload()

async def main():
    """Run all tests."""
    print("Song Generation Inference App - Test Suite")
    print("=" * 50)
    
    # Check CUDA availability
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("✗ CUDA not available - tests may fail")
    
    print()
    
    # Run tests
    await test_basic_generation()
    await test_separate_generation()
    await test_auto_prompt()
    await test_different_styles()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    asyncio.run(main()) 