"""Test whether the released Mimi codec preserves human-spoken Hindi."""

import argparse
import json
import time
from pathlib import Path

import scipy.io.wavfile
import torch
from huggingface_hub import hf_hub_download

from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.modules.stateful_module import init_states

DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e3_codec_reconstruction"
DEFAULT_REPO = "dhruvkys/hi-asr-1k"
DEFAULT_AUDIO_FILE = "test/audio/cv_000803.mp3"
DEFAULT_TRANSCRIPT = "मैं मुसीबत में पड़ गया।"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--audio-file", default=DEFAULT_AUDIO_FILE)
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(
        hf_hub_download(repo_id=args.repo_id, filename=args.audio_file, repo_type="dataset")
    )

    model = TTSModel.load_model(language="english")
    source_audio, source_rate = audio_read(source_path)
    source_audio = convert_audio(
        source_audio, source_rate, model.sample_rate, model.config.mimi.channels
    )

    started = time.monotonic()
    latents = model.mimi.encode_to_latent(source_audio.unsqueeze(0))
    projected_latents = model.mimi.quantizer(latents)
    mimi_state = init_states(model.mimi, batch_size=1, sequence_length=10_000)
    reconstruction = model.mimi.decode_from_latent(projected_latents, mimi_state)
    elapsed = time.monotonic() - started

    source_output = args.output_dir / "source_24khz.wav"
    reconstruction_output = args.output_dir / "reconstructed.wav"
    scipy.io.wavfile.write(source_output, model.sample_rate, source_audio.squeeze().numpy())
    scipy.io.wavfile.write(
        reconstruction_output, model.sample_rate, reconstruction.squeeze().numpy()
    )

    source_duration = source_audio.shape[-1] / model.sample_rate
    reconstruction_duration = reconstruction.shape[-1] / model.sample_rate
    metadata = {
        "dataset": args.repo_id,
        "dataset_audio_file": args.audio_file,
        "license": "CC0-1.0",
        "transcript": args.transcript,
        "source_sample_rate": source_rate,
        "codec_sample_rate": model.sample_rate,
        "latent_dimension": latents.shape[1],
        "latent_frames": latents.shape[2],
        "source_duration_seconds": round(source_duration, 3),
        "reconstruction_duration_seconds": round(reconstruction_duration, 3),
        "codec_seconds": round(elapsed, 3),
        "source_output": str(source_output),
        "reconstruction_output": str(reconstruction_output),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
