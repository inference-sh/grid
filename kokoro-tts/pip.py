#!/usr/bin/env python
import sys

def main():
    """
    Simple function to log the command-line arguments this script was called with.
    """
    print("pip.py was called with the following arguments:")
    
    for i, arg in enumerate(sys.argv):
        print(f"  Argument {i}: {arg}")
    
    print(f"Total arguments: {len(sys.argv)}")

if __name__ == "__main__":
    main()
