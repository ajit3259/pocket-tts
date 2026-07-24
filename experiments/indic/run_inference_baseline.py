"""Generate controlled English, Hindi, and Hinglish baseline samples."""

import argparse
import json
import time
from pathlib import Path

import scipy.io.wavfile
import torch

from pocket_tts import TTSModel

DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e2_english_model"
PROBES = {
    "english": "The weather is very pleasant today.",
    "hindi": "आज मौसम बहुत अच्छा है।",
    "hinglish": "Aaj weather bahut accha hai.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model(language="english")
    voice_state = model.get_state_for_audio_prompt(args.voice)
    results = []

    for label, text in PROBES.items():
        torch.manual_seed(args.seed)
        started = time.monotonic()
        audio = model.generate_audio(voice_state, text)
        elapsed = time.monotonic() - started
        duration = audio.numel() / model.sample_rate
        output_path = args.output_dir / f"{label}.wav"
        scipy.io.wavfile.write(output_path, model.sample_rate, audio.numpy())
        results.append(
            {
                "label": label,
                "text": text,
                "seed": args.seed,
                "voice": args.voice,
                "sample_rate": model.sample_rate,
                "duration_seconds": round(duration, 3),
                "generation_seconds": round(elapsed, 3),
                "real_time_factor": round(elapsed / duration, 3),
                "output": str(output_path),
            }
        )

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
