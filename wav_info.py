import argparse
import wave
from pathlib import Path

import numpy as np

DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}  # sample width (bytes) -> dtype

def get_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        info = {
            "sample_rate": w.getframerate(),
            "bit_depth": w.getsampwidth() * 8,
            "channels": w.getnchannels(),
            "frames": w.getnframes(),
        }
    info["duration"] = info["frames"] / info["sample_rate"]
    return info


def print_info(info: dict, label: str) -> None:
    kind = {1: "mono", 2: "stereo"}.get(info["channels"], "multi-channel")
    print(f"--- {label} ---")
    print(f"  Sample rate (Fs) : {info['sample_rate']} Hz")
    print(f"  Bit depth        : {info['bit_depth']} bits")
    print(f"  Channels         : {info['channels']} ({kind})")
    print(f"  Frames           : {info['frames']}")
    print(f"  Duration         : {info['duration']:.3f} s")


def decode(raw: bytes, sampwidth: int) -> np.ndarray:
    """Decode raw PCM bytes to float64 samples (kept in native integer range)."""
    if sampwidth == 3:  # 24-bit: unpack little-endian byte triplets
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        return np.where(v & 0x800000, v - 0x1000000, v).astype(np.float64)
    return np.frombuffer(raw, dtype=DTYPES[sampwidth]).astype(np.float64)


def encode(samples: np.ndarray, sampwidth: int) -> bytes:
    if sampwidth == 3:
        v = np.clip(np.round(samples), -2**23, 2**23 - 1).astype(np.int32)
        b = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1)
        return b.astype(np.uint8).tobytes()
    dtype = DTYPES[sampwidth]
    lo, hi = np.iinfo(dtype).min, np.iinfo(dtype).max
    if sampwidth == 1:  # 8-bit WAV is unsigned
        lo, hi = 0, 255
    return np.clip(np.round(samples), lo, hi).astype(dtype).tobytes()


def to_mono(src: Path, dst: Path) -> None:
    with wave.open(str(src), "rb") as w:
        params = w.getparams()
        samples = decode(w.readframes(params.nframes), params.sampwidth)

    if params.nchannels > 1:
        samples = samples.reshape(-1, params.nchannels).mean(axis=1)

    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(encode(samples, params.sampwidth))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, nargs="?", default=None,
                        help="path to the .wav file (asked interactively if omitted)")
    parser.add_argument("--mono", action="store_true",
                        help="also write a mono copy next to the original")
    args = parser.parse_args()

    # Saisie manuelle du chemin si non fourni en argument
    if args.file is None:
        raw = input("Chemin du fichier .wav : ").strip().strip('"').strip("'")
        args.file = Path(raw)

    print_info(get_info(args.file), args.file.name)

    if args.mono:
        dst = args.file.with_name(args.file.stem + "_mono.wav")
        to_mono(args.file, dst)
        print_info(get_info(dst), dst.name)


if __name__ == "__main__":
    main()
