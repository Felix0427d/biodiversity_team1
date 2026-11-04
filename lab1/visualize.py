import wave
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from lab1.wav_info import decode

DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}

def read_mono(path: Path) -> tuple[np.ndarray, int]:
    """Lit un .wav et renvoie (echantillons mono normalises [-1, 1], Fs)."""
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        samples = decode(w.readframes(params.nframes), params.sampwidth)

    if params.nchannels > 1:  # stereo -> moyenne des canaux
        samples = samples.reshape(-1, params.nchannels).mean(axis=1)

    full_scale = 128.0 if params.sampwidth == 1 else float(2 ** (params.sampwidth * 8 - 1))
    return samples / full_scale, params.framerate

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, nargs="?", default=None,
                        help="path to the .wav file (asked interactively if omitted)")
    args = parser.parse_args()

    if args.file is None:
        raw = input("Chemin du fichier .wav : ").strip().strip('"').strip("'")
        args.file = Path(raw)

    samples, fs = read_mono(args.file)
    t = np.arange(len(samples)) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # --- Forme d'onde : amplitude en fonction du temps ---
    ax1.plot(t, samples, lw=0.3, color="steelblue")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Forme d'onde - {args.file.name}")
    ax1.grid(alpha=0.3)

    # --- Spectrogramme : frequence/amplitude en fonction du temps ---
    ax2.specgram(samples, NFFT=1024, Fs=fs, noverlap=512, cmap="viridis")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Frequence (Hz)")
    ax2.set_title("Spectrogramme (vue temps-frequence)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
