import argparse
import wave   
from pathlib import Path
import numpy as np

DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32} 

def get_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as w: #La fonction ouvre le fichier en lecture binaire et relève quatre valeurs
        info = {
            "sample_rate": w.getframerate(),
            "bit_depth": w.getsampwidth() * 8,
            "channels": w.getnchannels(),
            "frames": w.getnframes(),
        }
    info["duration"] = info["frames"] / info["sample_rate"]
    return info


def print_info(info: dict, label: str) -> None:
    kind = {1: "mono", 2: "stereo"}.get(info["channels"], "multi-channel") #Ce dictionnaire traduit le nombre de canaux en texte, à partir de 3 = multi-channel
    print(f"--- {label} ---")
    print(f"  Sample rate (Fs) : {info['sample_rate']} Hz")
    print(f"  Bit depth        : {info['bit_depth']} bits")
    print(f"  Channels         : {info['channels']} ({kind})")
    print(f"  Frames           : {info['frames']}")
    print(f"  Duration         : {info['duration']:.3f} s")


def decode(raw: bytes, sampwidth: int) -> np.ndarray:
    """Decode raw PCM bytes to float64 samples (kept in native integer range)."""
    """Decode transforme les octets d’un fichier WAV en nombres que Python peut utiliser.
    Un fichier WAV stocke le son sous forme d’octets bruts, qui ne veulent rien dire tels quels. 
    La fonction les regroupe selon la taille d’un échantillon (1, 2, 3 ou 4 octets) et les convertit 
    en vrais nombres : un son fort donne une grande valeur, un son faible une petite, et le sens (positif ou négatif) 
    indique si la membrane du haut-parleur va vers l’avant ou vers l’arrière.

    Le cas du 24 bits demande un traitement manuel parce que numpy ne sait pas lire des nombres de 3 octets. 
    Pour les autres tailles, numpy s’en charge directement."""
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
"""Elle fait trois choses :

Arrondir : un fichier WAV ne stocke que des nombres entiers, alors que la moyenne des canaux donne 
parfois des décimales (par exemple 100,5).
Limiter les valeurs (clip) : chaque taille d’échantillon a une plage autorisée, par exemple −32768 
à 32767 en 16 bits. Une valeur hors de cette plage est ramenée à la limite, car sinon elle 
reboucle de l’autre côté (32768 deviendrait −32768) et on entendrait un craquement.
Convertir en octets : numpy le fait directement pour 8, 16 et 32 bits. Pour le 24 bits, 
la fonction découpe chaque nombre en 3 octets à la main, du moins significatif au plus significatif."""


def to_mono(src: Path, dst: Path) -> None:
    """Crée une version mono du fichier WAV."""

    with wave.open(str(src), "rb") as w:
        # On récupère les paramètres du fichier original.
        params = w.getparams()

        # On lit les données audio et on les décode en valeurs numériques.
        samples = decode(
            w.readframes(params.nframes), params.sampwidth
        )

    if params.nchannels > 1:
        # On organise les échantillons par frame et par canal.
        # On calcule ensuite la moyenne des canaux pour obtenir un seul canal.
        samples = samples.reshape(-1, params.nchannels).mean(axis=1)

    # Création du nouveau fichier WAV en écriture.
    with wave.open(str(dst), "wb") as w:
        # Le nouveau fichier possède un seul canal.
        w.setnchannels(1)

        # On conserve la profondeur de bits et la fréquence d'échantillonnage.
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)

        # On encode les échantillons puis on les écrit dans le fichier.
        w.writeframes(encode(samples, params.sampwidth))


def main() -> None:
    """Gère les arguments et lance les différentes opérations."""

    # Création du gestionnaire d'arguments du programme.
    parser = argparse.ArgumentParser(description=__doc__)

    # Permet de donner le chemin du fichier WAV directement dans le terminal.
    # Si aucun chemin n'est fourni, une saisie sera demandée.
    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
        default=None,
        help="path to the .wav file (asked interactively if omitted)"
    )

    # Option permettant de demander la création d'une version mono.
    parser.add_argument(
        "--mono",
        action="store_true",
        help="also write a mono copy next to the original"
    )

    args = parser.parse_args()

    # Si aucun fichier n'a été indiqué, on demande son chemin à l'utilisateur.
    if args.file is None:
        raw = input("Chemin du fichier .wav : ").strip().strip('"').strip("'")
        args.file = Path(raw)

    # Affichage des informations du fichier original.
    print_info(get_info(args.file), args.file.name)

    if args.mono:
        # Création du nom du fichier mono à partir du nom original.
        dst = args.file.with_name(args.file.stem + "_mono.wav")

        # Conversion et affichage des informations du nouveau fichier.
        to_mono(args.file, dst)
        print_info(get_info(dst), dst.name)


if __name__ == "__main__":
    main()
