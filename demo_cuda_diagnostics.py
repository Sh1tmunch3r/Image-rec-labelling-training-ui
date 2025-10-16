#!/usr/bin/env python3
"""
Demonstration script for CUDA diagnostics functionality
This script shows how the application detects and reports CUDA status
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from device_utils import log_cuda_diagnostics, get_cuda_diagnostics, get_device

def main():
    print("\n" + "=" * 70)
    print("CUDA Detection Demo - Image Labeling Studio Pro")
    print("=" * 70)
    print()
    
    # Show full diagnostics (as shown in startup logs)
    log_cuda_diagnostics()
    
    print("\n" + "=" * 70)
    print("Device Selection Testing")
    print("=" * 70)
    print()
    
    # Test all device preferences
    for preference in ['auto', 'force_gpu', 'force_cpu']:
        device, device_name, warning = get_device(preference)
        
        print(f"\nPreference: {preference}")
        print(f"  Device: {device}")
        print(f"  Device Name: {device_name}")
        if warning:
            print(f"  ⚠️  Warning: {warning}")
        else:
            print(f"  ✓  No warnings")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    diagnostics = get_cuda_diagnostics()
    
    if diagnostics['cuda_available']:
        print("\n✓ CUDA is available and working!")
        print(f"  GPU: {diagnostics['device_name']}")
        print(f"  Training will use GPU acceleration (5-10x faster)")
    else:
        print("\n⚠️  CUDA is not available")
        print(f"  PyTorch version: {diagnostics['torch_version']}")
        print(f"  CUDA version: {diagnostics['cuda_version']}")
        
        if diagnostics['cuda_version'] is None:
            print("\n  Issue: CPU-only PyTorch installation")
            print("  Solution: Reinstall PyTorch with CUDA support")
            print("  Visit: https://pytorch.org/get-started/locally/")
        else:
            print("\n  Issue: CUDA installed but GPU not detected")
            print("  Solutions:")
            print("    1. Check if you have an NVIDIA GPU (run 'nvidia-smi')")
            print("    2. Verify NVIDIA drivers are installed and up to date")
            print("    3. Check CUDA_VISIBLE_DEVICES environment variable")
        
        print("\n  Training will use CPU (slower but functional)")
    
    print("\n" + "=" * 70)
    print()

if __name__ == '__main__':
    main()
