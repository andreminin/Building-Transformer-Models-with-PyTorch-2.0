import torch
import gc
import subprocess
import os
from typing import Dict, Optional, Tuple

def get_device() -> str:
    """
    Get the best available device for PyTorch with priority: CUDA > MPS > CPU
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    # Priority 1: CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        # Get CUDA device info
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "Unknown"
        cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda_device_count > 0 else 0
        
        print(f"CUDA detected: {cuda_device_count} device(s)")
        print(f"Device: {cuda_device_name}")
        print(f"GPU Memory: {cuda_memory:.1f} GB")
        return "cuda"
    
    # Priority 2: MPS (Apple Silicon GPUs) - only if CUDA is not available
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) detected")
        return "mps"
    
    # Priority 3: CPU (fallback)
    else:
        print("Using CPU (no GPU detected)")
        return "cpu"
        
def print_cuda_memory(verbose: bool = False) -> Optional[Dict[str, float]]:
    """
    Print current CUDA memory usage and return memory statistics
    
    Args:
        verbose: If True, print detailed information
    
    Returns:
        Dictionary with memory statistics or None if CUDA not available
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        if verbose:
            print("=== GPU Memory Usage ===")
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Reserved:  {reserved:.2f} GB")
            print(f"Free:      {free:.2f} GB")
            print(f"Total:     {total:.2f} GB")
        else:
            print(f"GPU Memory: {allocated:.2f}GB used, {free:.2f}GB free, {total:.2f}GB total")
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    else:
        print("CUDA not available")
        return None

def get_detailed_memory_info() -> Optional[Dict[str, float]]:
    """
    Get detailed CUDA memory information including max usage
    
    Returns:
        Dictionary with detailed memory statistics or None if CUDA not available
    """
    if torch.cuda.is_available():
        stats = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
            'max_reserved': torch.cuda.max_memory_reserved() / 1024**3,
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
        stats['free'] = stats['total'] - stats['allocated']
        return stats
    return None

def print_detailed_memory():
    """Print detailed memory information"""
    stats = get_detailed_memory_info()
    if stats:
        print("=== Detailed GPU Memory ===")
        for key, value in stats.items():
            print(f"{key:15}: {value:.2f} GB")

# Clear all memory
def full_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory cleared")
    
    # Try to kill other Python processes using GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid and pid != str(os.getpid()):
                subprocess.run(['kill', '-9', pid])
                print(f"Killed process {pid}")
    except:
        print("Could not kill other processes - may require sudo privileges")
        
def clear_gpu_memory(aggressive: bool = False):
    """
    Clear GPU memory and perform garbage collection
    
    Args:
        aggressive: If True, try to kill other Python processes using GPU
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory cleared")
    
    if aggressive:
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            pids = result.stdout.strip().split('\n')
            current_pid = str(os.getpid())
            for pid in pids:
                if pid and pid != current_pid:
                    subprocess.run(['kill', '-9', pid])
                    print(f"Killed process {pid}")
        except Exception as e:
            print(f"Could not kill other processes: {e}")
            
def calculate_optimal_batch_size_conservative(model, max_batch_size=16, safety_margin=0.7):
    """
    Conservative batch size calculation for transformer models
    """
    if not torch.cuda.is_available():
        return 2
    
    clear_gpu_memory()
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated() / 1024**3
    available_memory = total_memory * safety_margin - allocated
    
    print(f"Total GPU memory: {total_memory:.2f}GB")
    print(f"Currently allocated: {allocated:.2f}GB")
    print(f"Available for training: {available_memory:.2f}GB")
    
    # Conservative estimates for BERT models:
    # - Small batch (2-4): ~1-2GB memory
    # - Medium batch (8-16): ~4-8GB memory  
    # - Large batch (16+): ~8+ GB memory
    
    if available_memory > 10:
        batch_size = min(16, max_batch_size)  # Cap at 16 for stability
    elif available_memory > 6:
        batch_size = 8
    elif available_memory > 3:
        batch_size = 4
    elif available_memory > 1.5:
        batch_size = 2
    else:
        batch_size = 1
    
    print(f"Recommended batch size: {batch_size}")
    return batch_size
    
def calculate_optimal_batch_size_simple(model, sample_batch, max_batch_size=32, safety_margin=0.8):
    """Simple batch size calculation"""
    if not torch.cuda.is_available():
        return 2
    
    clear_gpu_memory()
    base_allocated = torch.cuda.memory_allocated() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    available_memory = total_memory * safety_margin - base_allocated
    
    if available_memory <= 0:
        return 1
    
    # Conservative estimate for BERT models
    memory_per_batch = 0.4  # GB per batch
    optimal_batch_size = max(1, min(max_batch_size, int(available_memory / memory_per_batch)))
    
    print(f"Available memory: {available_memory:.2f}GB")
    print(f"Using batch size: {optimal_batch_size}")
    
    return optimal_batch_size

def get_optimal_batch_size():
    if not torch.cuda.is_available():
        return 2
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated() / 1024**3
    available = total_memory * 0.7 - allocated  # Use 70% of available memory
    
    # Estimate based on BERT model size
    if available > 4:
        return 8
    elif available > 2:
        return 4
    elif available > 1:
        return 2
    else:
        return 1

def setup_memory_environment(expandable_segments: bool = True):
    """
    Set up PyTorch memory environment variables
    
    Args:
        expandable_segments: Whether to use expandable memory segments
    """
    if expandable_segments:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("Memory environment configured")

def get_device() -> str:
    """
    Get the best available device for PyTorch
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage of a function
    
    Usage:
        @monitor_memory_usage
        def your_function():
            # your code here
    """
    def wrapper(*args, **kwargs):
        print("Memory before function execution:")
        print_cuda_memory(verbose=True)
        
        result = func(*args, **kwargs)
        
        print("Memory after function execution:")
        print_cuda_memory(verbose=True)
        
        return result
    return wrapper

def get_optimal_bar_length():
    """Calculate optimal bar length based on terminal width"""
    try:
        width = os.get_terminal_size().columns
        # Allow space for description, counters, and metrics
        return max(30, width - 70)
    except:
        return 50