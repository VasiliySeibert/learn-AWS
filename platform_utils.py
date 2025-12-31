"""
Platform detection utilities for cross-platform networking configuration.

This module provides automatic detection of the runtime environment (WSL2, macOS, Linux)
and returns appropriate networking configurations for Gradio and LM Studio.

Key Features:
- Automatic WSL2 detection
- Windows host IP resolution from WSL2
- Environment variable overrides for all settings
- Graceful fallbacks for edge cases
"""

import os
import re
from pathlib import Path


def is_wsl() -> bool:
    """
    Detect if running in Windows Subsystem for Linux (WSL).

    Detection method: Check if /proc/version contains "microsoft" or "Microsoft"
    This is reliable for both WSL1 and WSL2.

    Returns:
        bool: True if running in WSL, False otherwise

    Examples:
        >>> is_wsl()  # On WSL
        True
        >>> is_wsl()  # On macOS or native Linux
        False
    """
    try:
        proc_version = Path("/proc/version")
        if proc_version.exists():
            content = proc_version.read_text().lower()
            return "microsoft" in content
    except (OSError, IOError, PermissionError) as e:
        # If we can't read /proc/version, assume not WSL
        print(f"Warning: Could not read /proc/version: {e}")

    return False


def get_windows_host_ip() -> str | None:
    """
    Get the Windows host IP address from WSL.

    In WSL2, the Windows host is accessible via the nameserver IP listed in
    /etc/resolv.conf. This function parses that file to extract the IP.

    Returns:
        str | None: Windows host IP address (e.g., "10.255.255.254"), or None if:
            - Not running in WSL
            - Cannot read /etc/resolv.conf
            - Cannot parse nameserver IP

    Examples:
        >>> get_windows_host_ip()  # On WSL2
        '10.255.255.254'
        >>> get_windows_host_ip()  # On macOS
        None
    """
    if not is_wsl():
        return None

    try:
        resolv_conf = Path("/etc/resolv.conf")
        if not resolv_conf.exists():
            print("Warning: /etc/resolv.conf not found")
            return None

        content = resolv_conf.read_text()

        # Look for lines like: nameserver 10.255.255.254
        # Regex: match "nameserver" followed by whitespace and an IP address
        match = re.search(r'^nameserver\s+(\d+\.\d+\.\d+\.\d+)', content, re.MULTILINE)

        if match:
            return match.group(1)
        else:
            print("Warning: Could not find nameserver in /etc/resolv.conf")
            return None

    except (OSError, IOError, PermissionError) as e:
        print(f"Warning: Could not read /etc/resolv.conf: {e}")
        return None


def get_lm_studio_url() -> str:
    """
    Get the appropriate LM Studio API URL based on the environment.

    Resolution priority (highest to lowest):
    1. LM_STUDIO_URL environment variable (if set)
    2. Windows host IP in WSL2 (if detected)
    3. localhost (default fallback)

    Returns:
        str: Full LM Studio base URL (e.g., "http://10.255.255.254:1234/v1")

    Examples:
        >>> os.environ['LM_STUDIO_URL'] = 'http://192.168.1.100:1234/v1'
        >>> get_lm_studio_url()
        'http://192.168.1.100:1234/v1'

        >>> # On WSL2 with no env var
        >>> get_lm_studio_url()
        'http://10.255.255.254:1234/v1'

        >>> # On macOS with no env var
        >>> get_lm_studio_url()
        'http://localhost:1234/v1'
    """
    # Priority 1: Environment variable override
    env_url = os.environ.get("LM_STUDIO_URL")
    if env_url:
        if os.environ.get("DEBUG_NETWORKING"):
            print(f"[DEBUG] Using LM Studio URL from env var: {env_url}")
        return env_url

    # Priority 2: WSL detection
    if is_wsl():
        host_ip = get_windows_host_ip()
        if host_ip:
            url = f"http://{host_ip}:1234/v1"
            if os.environ.get("DEBUG_NETWORKING"):
                print(f"[DEBUG] WSL detected, using Windows host: {url}")
            return url
        else:
            # Fallback if IP detection fails in WSL
            print("Warning: WSL detected but couldn't get Windows host IP, using localhost")
            return "http://localhost:1234/v1"

    # Priority 3: Default (macOS/Linux)
    url = "http://localhost:1234/v1"
    if os.environ.get("DEBUG_NETWORKING"):
        print(f"[DEBUG] Using default LM Studio URL: {url}")
    return url


def get_gradio_server_name() -> str:
    """
    Get the appropriate Gradio server_name based on the environment.

    Resolution priority (highest to lowest):
    1. GRADIO_SERVER_NAME environment variable (if set)
    2. "0.0.0.0" in WSL2 (accessible from Windows)
    3. "127.0.0.1" on macOS/Linux (localhost only)

    Returns:
        str: Server name/IP to bind Gradio to

    Examples:
        >>> os.environ['GRADIO_SERVER_NAME'] = '192.168.1.100'
        >>> get_gradio_server_name()
        '192.168.1.100'

        >>> # On WSL2 with no env var
        >>> get_gradio_server_name()
        '0.0.0.0'

        >>> # On macOS with no env var
        >>> get_gradio_server_name()
        '127.0.0.1'
    """
    # Priority 1: Environment variable override
    env_server = os.environ.get("GRADIO_SERVER_NAME")
    if env_server:
        if os.environ.get("DEBUG_NETWORKING"):
            print(f"[DEBUG] Using Gradio server name from env var: {env_server}")
        return env_server

    # Priority 2: WSL detection - bind to all interfaces
    if is_wsl():
        server_name = "0.0.0.0"
        if os.environ.get("DEBUG_NETWORKING"):
            print(f"[DEBUG] WSL detected, binding Gradio to: {server_name}")
        return server_name

    # Priority 3: Default (macOS/Linux) - bind to localhost only
    server_name = "127.0.0.1"
    if os.environ.get("DEBUG_NETWORKING"):
        print(f"[DEBUG] Using default Gradio server name: {server_name}")
    return server_name


if __name__ == "__main__":
    """Test the platform detection functions."""
    print("=" * 60)
    print("Platform Detection Test")
    print("=" * 60)

    wsl = is_wsl()
    print(f"Running in WSL: {wsl}")

    if wsl:
        host_ip = get_windows_host_ip()
        print(f"Windows Host IP: {host_ip}")

    print(f"\nLM Studio URL: {get_lm_studio_url()}")
    print(f"Gradio Server Name: {get_gradio_server_name()}")

    print("\n" + "=" * 60)
    print("Environment Variable Override Test")
    print("=" * 60)
    print("Set LM_STUDIO_URL and GRADIO_SERVER_NAME to test overrides:")
    print("  export LM_STUDIO_URL='http://custom-host:1234/v1'")
    print("  export GRADIO_SERVER_NAME='0.0.0.0'")
    print("  export DEBUG_NETWORKING='1'")
