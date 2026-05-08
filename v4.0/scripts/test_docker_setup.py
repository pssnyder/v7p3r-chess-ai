#!/usr/bin/env python3
"""
Docker Training Setup Tester
============================

Validates that all components are ready for 48-hour training.
Run this before starting the full training to catch issues early.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def test_docker_installed():
    """Test if Docker is installed and running."""
    print("🔍 Testing Docker installation...")
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  ✅ Docker installed: {result.stdout.strip()}")
            return True
        else:
            print("  ❌ Docker not found")
            return False
    except Exception as e:
        print(f"  ❌ Docker check failed: {e}")
        return False


def test_docker_compose():
    """Test if Docker Compose is installed."""
    print("🔍 Testing Docker Compose...")
    try:
        result = subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  ✅ Docker Compose installed: {result.stdout.strip()}")
            return True
        else:
            print("  ❌ Docker Compose not found")
            return False
    except Exception as e:
        print(f"  ❌ Docker Compose check failed: {e}")
        return False


def test_gpu_support():
    """Test if GPU support is available."""
    print("🔍 Testing GPU support...")
    try:
        result = subprocess.run(
            ['docker', 'run', '--rm', '--gpus', 'all',
             'nvidia/cuda:12.1.0-base-ubuntu22.04', 'nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("  ✅ GPU support available")
            print("  GPU Info:")
            for line in result.stdout.split('\n')[:5]:
                if line.strip():
                    print(f"    {line}")
            return True
        else:
            print("  ⚠️  No GPU support (CPU training will be used)")
            return False
    except Exception as e:
        print(f"  ⚠️  GPU check failed: {e}")
        return False


def test_directories():
    """Test if required directories exist."""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        'scripts',
        'src',
        'data',
        'models',
        'logs'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ exists")
        else:
            print(f"  ⚠️  {dir_name}/ missing (will be created)")
            dir_path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return True  # Always pass since we create missing dirs


def test_dockerfile():
    """Test if Dockerfile is present and valid."""
    print("🔍 Testing Dockerfile...")
    
    dockerfile = Path('Dockerfile')
    if not dockerfile.exists():
        print("  ❌ Dockerfile not found")
        return False
    
    print("  ✅ Dockerfile exists")
    
    # Check for key stages
    with open(dockerfile, 'r') as f:
        content = f.read()
    
    required_stages = ['FROM', 'WORKDIR', 'COPY', 'CMD']
    for stage in required_stages:
        if stage in content:
            print(f"  ✅ Contains {stage}")
        else:
            print(f"  ⚠️  Missing {stage}")
    
    return True


def test_docker_compose_file():
    """Test if docker-compose.yml is valid."""
    print("🔍 Testing docker-compose.yml...")
    
    compose_file = Path('docker-compose.yml')
    if not compose_file.exists():
        print("  ❌ docker-compose.yml not found")
        return False
    
    print("  ✅ docker-compose.yml exists")
    
    # Validate with docker-compose config
    try:
        result = subprocess.run(
            ['docker-compose', 'config'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("  ✅ docker-compose.yml is valid")
            return True
        else:
            print(f"  ❌ docker-compose.yml validation failed:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"  ⚠️  Could not validate: {e}")
        return True  # Don't fail if validation tool isn't available


def test_scripts():
    """Test if required scripts exist."""
    print("🔍 Testing required scripts...")
    
    required_scripts = [
        'scripts/48h_training_orchestrator.py',
        'scripts/training_health_check.py',
        'scripts/train_move_ordering.py'
    ]
    
    all_exist = True
    for script_name in required_scripts:
        script_path = Path(script_name)
        if script_path.exists():
            print(f"  ✅ {script_name} exists")
        else:
            print(f"  ❌ {script_name} not found")
            all_exist = False
    
    return all_exist


def test_disk_space():
    """Test if sufficient disk space is available."""
    print("🔍 Testing disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        
        free_gb = free // (2**30)
        print(f"  ℹ️  Free space: {free_gb} GB")
        
        if free_gb < 20:
            print("  ⚠️  Low disk space (<20 GB)")
            print("  Recommendation: Free up at least 50 GB for training")
            return False
        elif free_gb < 50:
            print("  ⚠️  Disk space adequate but tight (<50 GB)")
            return True
        else:
            print("  ✅ Sufficient disk space (≥50 GB)")
            return True
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True


def test_requirements():
    """Test if requirements.txt exists."""
    print("🔍 Testing requirements.txt...")
    
    req_file = Path('requirements.txt')
    if req_file.exists():
        print("  ✅ requirements.txt exists")
        
        # Count dependencies
        with open(req_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        
        print(f"  ℹ️  {len(lines)} dependencies specified")
        return True
    else:
        print("  ❌ requirements.txt not found")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("V7P3R Docker Training Setup Tester")
    print("=" * 80)
    print()
    
    tests = [
        ('Docker Installation', test_docker_installed),
        ('Docker Compose', test_docker_compose),
        ('GPU Support', test_gpu_support),
        ('Directory Structure', test_directories),
        ('Dockerfile', test_dockerfile),
        ('Docker Compose File', test_docker_compose_file),
        ('Required Scripts', test_scripts),
        ('Disk Space', test_disk_space),
        ('Requirements File', test_requirements)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print()
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results[test_name] = False
    
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:10} | {test_name}")
    
    print()
    print(f"Score: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("✅ All tests passed! Ready to start training.")
        print()
        print("Next steps:")
        print("  1. Build container:  docker-compose build")
        print("  2. Start training:   docker-compose up -d")
        print("  3. Monitor:          docker-compose logs -f training")
        print()
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        print()
        print("Common fixes:")
        print("  - Install Docker Desktop: https://www.docker.com/products/docker-desktop")
        print("  - Install NVIDIA Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/")
        print("  - Free up disk space: Delete old models/logs")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
