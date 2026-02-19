#!/usr/bin/env python3
"""Test script to identify tuple index out of range issues in cond_conc.py"""

import numpy as np
import pandas as pd
from cond_conc import calculate_effective_mass, dos_parabolic

def test_calculate_effective_mass():
    """Test the calculate_effective_mass function with edge cases"""
    print("Testing calculate_effective_mass...")
    
    # Test with empty arrays
    try:
        energies = np.array([])
        dos = np.array([])
        result = calculate_effective_mass(energies, dos, "VBM")
        print("Empty arrays test passed")
    except Exception as e:
        print(f"Empty arrays test failed: {e}")
    
    # Test with minimal data
    try:
        energies = np.array([-1.0, 0.0, 1.0])
        dos = np.array([1.0, 0.5, 2.0])
        result = calculate_effective_mass(energies, dos, "VBM")
        print(f"Minimal data test passed: {result}")
    except Exception as e:
        print(f"Minimal data test failed: {e}")
        
    # Test with all zero DOS
    try:
        energies = np.array([-1.0, 0.0, 1.0])
        dos = np.array([0.0, 0.0, 0.0])
        result = calculate_effective_mass(energies, dos, "CBM")
        print(f"Zero DOS test passed: {result}")
    except Exception as e:
        print(f"Zero DOS test failed: {e}")

def test_where_usage():
    """Test numpy.where usage that could cause tuple index errors"""
    print("\nTesting numpy.where usage...")
    
    # Test empty where result
    try:
        arr = np.array([1, 2, 3])
        indices = np.where(arr > 10)[0]  # Should be empty
        if len(indices) > 0:
            first_idx = indices[0]
            print(f"First index: {first_idx}")
        else:
            print("No indices found (this is expected)")
    except IndexError as e:
        print(f"Where usage test failed: {e}")

if __name__ == "__main__":
    test_calculate_effective_mass()
    test_where_usage()
    print("\nAll tests completed.")