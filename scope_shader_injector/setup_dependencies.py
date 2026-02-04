"""Helper script to check and suggest OpenGL dependencies."""

import os
import subprocess
import sys


def check_opengl_support() -> dict:
    """Check available OpenGL support on the system.
    
    Returns:
        Dict with status of various OpenGL backends.
    """
    results = {
        "platform": sys.platform,
        "display": None,
        "egl": None,
        "gl": None,
        "context_creation": None,
    }
    
    # Check display
    results["display"] = bool(
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    )
    
    if sys.platform.startswith('linux'):
        # Check EGL
        try:
            import ctypes.util
            results["egl"] = ctypes.util.find_library('EGL')
        except Exception:
            results["egl"] = None
            
        # Check GL
        try:
            import ctypes.util
            results["gl"] = ctypes.util.find_library('GL')
        except Exception:
            results["gl"] = None
    
    # Try to create a context
    try:
        import moderngl
        if sys.platform.startswith('linux'):
            try:
                ctx = moderngl.create_context(standalone=True, backend='egl')
                results["context_creation"] = "EGL"
                ctx.release()
            except Exception:
                try:
                    ctx = moderngl.create_context(standalone=True)
                    results["context_creation"] = "X11/GLX"
                    ctx.release()
                except Exception:
                    results["context_creation"] = None
        else:
            try:
                ctx = moderngl.create_context(standalone=True)
                results["context_creation"] = "default"
                ctx.release()
            except Exception:
                results["context_creation"] = None
    except ImportError:
        results["context_creation"] = "moderngl not installed"
    
    return results


def get_install_commands() -> list[str]:
    """Get commands to install OpenGL dependencies for the current platform."""
    if sys.platform.startswith('linux'):
        # Detect package manager
        if os.path.exists('/usr/bin/apt-get'):
            return [
                "sudo apt-get update",
                "sudo apt-get install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev",
            ]
        elif os.path.exists('/usr/bin/dnf'):
            return [
                "sudo dnf install -y mesa-libEGL-devel mesa-libGL-devel",
            ]
        elif os.path.exists('/usr/bin/yum'):
            return [
                "sudo yum install -y mesa-libEGL-devel mesa-libGL-devel",
            ]
        elif os.path.exists('/usr/bin/pacman'):
            return [
                "sudo pacman -S mesa",
            ]
    
    return []


def print_status():
    """Print OpenGL support status and recommendations."""
    print("=" * 60)
    print("Scope Shader Injector - OpenGL Dependency Check")
    print("=" * 60)
    print()
    
    status = check_opengl_support()
    
    print(f"Platform: {status['platform']}")
    print(f"Display available: {status['display']}")
    
    if sys.platform.startswith('linux'):
        print(f"EGL library: {status['egl'] or 'NOT FOUND'}")
        print(f"GL library: {status['gl'] or 'NOT FOUND'}")
    
    print(f"Context creation: {status['context_creation'] or 'FAILED'}")
    print()
    
    if status['context_creation']:
        print("✓ OpenGL context creation successful!")
        print("  The Shader Injector plugin should work correctly.")
    else:
        print("✗ OpenGL context creation failed!")
        print()
        print("Recommended installation commands:")
        print()
        
        commands = get_install_commands()
        if commands:
            for cmd in commands:
                print(f"  {cmd}")
        else:
            print("  Please install OpenGL/EGL development libraries for your platform.")
        
        print()
        print("After installing, restart the Scope server and reload the plugin.")
    
    print()
    print("=" * 60)
    
    return status['context_creation'] is not None


if __name__ == "__main__":
    success = print_status()
    sys.exit(0 if success else 1)
