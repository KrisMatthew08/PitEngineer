import configparser
import os
import re
import struct

COMMON_AC_PATHS = [
    r"D:\SteamLibrary\steamapps\common\assettocorsa",
    r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa",
    r"C:\Program Files\Steam\steamapps\common\assettocorsa",
    r"E:\SteamLibrary\steamapps\common\assettocorsa",
]


def find_ac_dir() -> str | None:
    """Find the Assetto Corsa root directory by checking common locations."""
    for path in COMMON_AC_PATHS:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "content", "cars")):
            return path
    return None


def get_car_dir(car_id: str, ac_dir: str | None = None) -> str | None:
    """Get the path to a specific car's directory."""
    if ac_dir is None:
        ac_dir = find_ac_dir()
    if ac_dir is None:
        return None
    car_dir = os.path.join(ac_dir, "content", "cars", car_id)
    return car_dir if os.path.exists(car_dir) else None


def read_setup_ini(car_id: str, ac_dir: str | None = None) -> configparser.ConfigParser | None:
    """Read the setup.ini for a car, either from an unpacked data folder or data.acd."""
    car_dir = get_car_dir(car_id, ac_dir)
    if car_dir is None:
        return None

    # Try unpacked data/setup.ini first
    unpacked_path = os.path.join(car_dir, "data", "setup.ini")
    if os.path.exists(unpacked_path):
        return _parse_ini_file(unpacked_path)

    # If no unpacked data, try parsing data.acd (best-effort)
    acd_path = os.path.join(car_dir, "data.acd")
    if os.path.exists(acd_path):
        content = extract_file_from_acd(acd_path, car_id, "setup.ini")
        if content:
            return _parse_ini_string(content)

    return None


def _clean_ini_content(content: str) -> str:
    """Assetto Corsa INI files often use // for comments, which breaks configparser."""
    lines = []
    for line in content.splitlines():
        # Strip // comments
        if "//" in line:
            line = line.split("//", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _parse_ini_file(filepath: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(comment_prefixes=(";", "#"))
    parser.optionxform = str  # Preserve case
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            content = f.read()
    except Exception:
        with open(filepath, "r") as f:
            content = f.read()
            
    content = _clean_ini_content(content)
    try:
        parser.read_string(content)
    except Exception:
        pass
    return parser


def _parse_ini_string(content: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(comment_prefixes=(";", "#"))
    parser.optionxform = str  # Preserve case
    content = _clean_ini_content(content)
    try:
        parser.read_string(content)
    except Exception:
        pass
    return parser


def extract_file_from_acd(acd_path: str, car_id: str, target_file: str) -> str | None:
    """
    Attempt to extract a plaintext file from an ACD archive.
    Note: ACD files are encrypted. We implement a partial decryption where possible,
    but it may not work for all custom encrypted cars.
    """
    try:
        with open(acd_path, "rb") as f:
            data = f.read()
    except Exception:
        return None

    # Search for the entry header
    name_bytes = target_file.encode("utf-8")
    match = None
    for m in re.finditer(re.escape(name_bytes), data):
        pos = m.start()
        if pos >= 4:
            try:
                stored_len = struct.unpack_from("<i", data, pos - 4)[0]
                if stored_len == len(name_bytes):
                    name_off = pos - 4
                    data_len_off = name_off + 4 + len(name_bytes)
                    if data_len_off + 4 <= len(data):
                        char_count = struct.unpack_from("<i", data, data_len_off)[0]
                        data_off = data_len_off + 4
                        match = (char_count, data_off)
                        break
            except Exception:
                continue

    if not match:
        return None

    char_count, data_off = match
    if data_off + char_count * 4 > len(data):
        return None

    try:
        enc_vals = [struct.unpack_from("<i", data, data_off + i * 4)[0] & 0xFF for i in range(char_count)]
        
        # We need the key stream derived from the car_id to decrypt the file.
        # This is a complex proprietary algorithm in AC.
        # Since we may not have the exact full algorithm yet, if we can't perfectly 
        # decrypt it, we return None and let the caller fallback to heuristics.
        # But we can try the known first key logic if we can figure out the rest.
        # For now, we will return None to signify we can't confidently parse it, 
        # relying on the unpacked setup.ini for authoritative data, and existing heuristics 
        # for packed cars.
        
        # Example of partial key derivation (not complete):
        # s_lower = sum(ord(c) for c in car_id.lower())
        # k0 = s_lower % 256
        
        # Currently we do not have 100% ACD decryption logic, so we fail gracefully.
        return None
        
    except Exception:
        return None
