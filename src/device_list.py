from typing import Dict, List, Optional
import sounddevice as sd


def get_all_devices() -> List[Dict]:
    devices = sd.query_devices()
    result = []
    for idx, dev in enumerate(devices):
        result.append({
            "index": idx,
            "name": dev["name"],
            "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": int(dev["default_samplerate"]),
            "is_default_input": idx == sd.default.device[0],
            "is_default_output": idx == sd.default.device[1],
        })
    return result


def filter_devices(devices: List[Dict], name_filter: str) -> List[Dict]:
    needle = name_filter.lower()
    return [d for d in devices if needle in d["name"].lower()]


def find_input_device(name_filter: str) -> Optional[Dict]:
    matches = [
        d for d in filter_devices(get_all_devices(), name_filter)
        if d["max_input_channels"] > 0
    ]
    return matches[0] if matches else None
