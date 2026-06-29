#!/usr/bin/env python3
"""
Deploy all two vision-language models using their respective virtual environments
"""

import os
import sys
import subprocess
import signal
import time

def kill_processes_on_port(port):
    """Kill any process using the specified port"""
    try:
        result = subprocess.run(
            f"lsof -ti :{port}", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"  Killed process {pid} on port {port}")
                except:
                    pass
            return True
    except:
        pass
    
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        return True
    except:
        pass
    
    return False

def main():
    print("="*60)
    print("Deploying Two Vision-Language Models")
    print("="*60)
    
    # Clean up ports
    print("\nCleaning up ports...")
    for port in [8010, 8012]:
        if kill_processes_on_port(port):
            print(f"  Cleared port {port}")
        else:
            print(f"  Port {port} is free")
    
    time.sleep(2)
    
    # Start each model in its own virtual environment
    processes = []
    
    # Kimi-VL
    print("\nStarting Kimi-VL on GPUs 2,3, port 8010...")
    kimi_process = subprocess.Popen(
        ["venvs/kimi/.venv/bin/python", "scripts/run_kimi.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(("Kimi-VL", kimi_process))
    
    # DeepSeek-VL
    print("Starting DeepSeek-VL on GPUs 4,5, port 8012...")
    deepseek_process = subprocess.Popen(
        ["venvs/deepseek/.venv/bin/python", "scripts/run_deepseek.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(("DeepSeek-VL", deepseek_process))
    
    print("\n" + "="*60)
    print("Models are starting...")
    print("="*60)
    
    # Monitor initial output
    print("\nMonitoring startup (first 30 seconds)...")
    start_time = time.time()
    while time.time() - start_time < 30:
        for name, proc in processes:
            if proc.poll() is None:  # Process is still running
                try:
                    line = proc.stdout.readline()
                    if line:
                        print(f"[{name}] {line.strip()}")
                except:
                    pass
        time.sleep(0.1)
    
    print("\n" + "="*60)
    print("All models should be deployed!")
    print("="*60)
    print("\nModel endpoints:")
    print("  - Kimi-VL: http://localhost:8010")
    print("  - DeepSeek-VL: http://localhost:8012")
    
    print("\nTest with:")
    print('  curl http://localhost:8010/health')
    print('  curl http://localhost:8012/health')
    
    print("\nPress Ctrl+C to stop all servers...")
    
    try:
        # Keep running and display output
        while True:
            for name, proc in processes:
                if proc.poll() is None:
                    try:
                        line = proc.stdout.readline()
                        if line:
                            print(f"[{name}] {line.strip()}")
                    except:
                        pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        for name, proc in processes:
            print(f"  Terminating {name}...")
            proc.terminate()
        
        # Wait for graceful shutdown
        time.sleep(2)
        
        # Force kill if still running
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Force killing {name}...")
                proc.kill()
        
        print("✓ All servers stopped")

if __name__ == "__main__":
    main()
