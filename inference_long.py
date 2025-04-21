import sys
import time
import math
import torch
import torch.nn.functional as F
import torchaudio
import argparse
from tqdm import tqdm
import librosa
from pathlib import Path

# Append BigVGAN to the system path
sys.path.append('./BigVGAN')

from BigVGAN.meldataset import get_mel_spectrogram
from model import OptimizedAudioRestorationModel

# Set device and optimize CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def measure_gpu_memory(device):
    """Measure GPU memory usage in MB"""
    if device == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0

def measure_performance(func):
    """Decorator to measure time and memory usage of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        initial_memory = measure_gpu_memory(device)

        result = func(*args, **kwargs)

        end_time = time.time()
        peak_memory = measure_gpu_memory(device)

        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        print(f"Memory used: {peak_memory - initial_memory:.2f} MB")
        return result
    return wrapper

def apply_overlap_windowing_waveform(waveform, window_size_samples, overlap):
    """Extract overlapping windows from a waveform with optimized memory usage"""
    step_size = int(window_size_samples * (1 - overlap))
    total_samples = waveform.shape[-1]
    num_windows = math.ceil((total_samples - window_size_samples) / step_size) + 1

    # Pre-allocate memory for windows
    windows = torch.zeros((num_windows, waveform.shape[0], window_size_samples), dtype=waveform.dtype)

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size_samples, total_samples)

        if end_idx - start_idx < window_size_samples:
            # Handle last window with padding
            windows[i, :, :(end_idx - start_idx)] = waveform[:, start_idx:end_idx]
        else:
            windows[i] = waveform[:, start_idx:end_idx]

    return windows

def reconstruct_waveform_from_windows(windows, window_size_samples, overlap, original_length=None):
    """Reconstruct waveform from processed windows using weight‑aware overlap‑add"""
    step_size = int(window_size_samples * (1 - overlap))

    # Handle different window shapes
    shape = windows.shape
    if len(shape) == 2:
        num_windows, window_len = shape
        channels = 1
        windows = windows.unsqueeze(1)
    elif len(shape) == 3:
        num_windows, channels, window_len = shape
    else:
        raise ValueError(f"Unexpected windows.shape: {windows.shape}")

    # Create output buffer
    output_length = (num_windows - 1) * step_size + window_size_samples
    reconstructed = torch.zeros((channels, output_length), dtype=torch.float32, device=windows.device)
    window_counts = torch.zeros((channels, output_length), dtype=torch.float32, device=windows.device)

    # Cross‑fade parameters
    fade_len = min(step_size, window_size_samples // 4)
    if fade_len > 0:
        fade_in = torch.linspace(0.0, 1.0, fade_len, device=windows.device)
        fade_out = torch.linspace(1.0, 0.0, fade_len, device=windows.device)

    # Process each window
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_len

        win = windows[i]  # (C, L)
        weight = torch.ones_like(win)  # matching weights

        if fade_len > 0:
            if i > 0:
                win[:, :fade_len] *= fade_in
                weight[:, :fade_len] = fade_in
            if i < num_windows - 1:
                win[:, -fade_len:] *= fade_out
                weight[:, -fade_len:] = fade_out

        reconstructed[:, start_idx:end_idx] += win
        window_counts[:, start_idx:end_idx] += weight

    # Normalise by the exact sum of weights, avoiding division by zero
    mask = window_counts > 0
    reconstructed[mask] /= window_counts[mask]

    # Trim to original length if specified
    if original_length is not None:
        reconstructed = reconstructed[:, :original_length]

    if channels == 1:
        reconstructed = reconstructed.squeeze(0)

    return reconstructed.cpu()

def load_bigvgan_model(device):
    """Load and optimise BigVGAN model"""
    from BigVGAN import bigvgan

    cache_dir = Path("./model_cache")
    cache_dir.mkdir(exist_ok=True)

    print("Loading BigVGAN model…")
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        'nvidia/bigvgan_v2_24khz_100band_256x',
        use_cuda_kernel=False,
        force_download=False,
        cache_dir=str(cache_dir)
    )

    # Move to device *before* removing weight norm
    bigvgan_model = bigvgan_model.to(device)
    bigvgan_model.remove_weight_norm()

    return bigvgan_model.eval()

def load_model(save_path, device, decoder):
    """Load and optimise audio restoration model"""
    optimized_model = OptimizedAudioRestorationModel(device=device)

    if decoder == 'bigvgan':
        optimized_model.bigvgan_model = load_bigvgan_model(device)
    else:
        raise ValueError(f"Unsupported decoder: {decoder}")

    # Load checkpoint
    state_dict = torch.load(save_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    optimized_model.voice_restore.load_state_dict(state_dict, strict=True)

    optimized_model.voice_restore = optimized_model.voice_restore.to(device)
    return optimized_model

def process_batch(model, batch_wav_windows, steps, cfg_strength, decoder):
    """Process a batch of windows through the model"""
    batch_processed_mel = get_mel_spectrogram(
        batch_wav_windows.squeeze(1),
        model.bigvgan_model.h
    ).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == 'cuda'):
        restored_mel = model.voice_restore.sample(
            batch_processed_mel.transpose(1, 2),
            steps=steps,
            cfg_strength=cfg_strength
        ).transpose(1, 2)

        if decoder == 'bigvgan':
            restored_wav = model.bigvgan_model(restored_mel).cpu()
        else:
            raise ValueError(f"Unsupported decoder: {decoder}")

    return restored_wav

@measure_performance
def restore_audio(model, input_path, output_path, steps=16, cfg_strength=0.1,
                 window_size_sec=5.0, overlap=0.1, batch_size=16, decoder='bigvgan',
                 max_memory_mb=4000):
    """Restore audio with optimised processing and memory management"""
    print(f"Processing audio: {input_path} -> {output_path}")
    print(f"Parameters: steps={steps}, cfg_strength={cfg_strength}, window_size={window_size_sec}s, overlap={overlap}")

    # Adaptive batch size
    if device == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        available_memory = min(max_memory_mb, total_memory * 0.8)  # Use at most 80 % of VRAM
        estimated_memory_per_sample = 500  # rough estimate in MB
        adaptive_batch_size = max(1, min(batch_size, int(available_memory / estimated_memory_per_sample)))
        if adaptive_batch_size < batch_size:
            print(f"Reducing batch size to {adaptive_batch_size} due to memory constraints")
            batch_size = adaptive_batch_size

    sr = model.bigvgan_model.h.sampling_rate

    # Try librosa first (streamed)
    try:
        wav, sr_orig = librosa.load(input_path, mono=True, sr=sr)
        wav = torch.FloatTensor(wav).unsqueeze(0)
    except Exception as e:
        print(f"Librosa failed: {e}. Falling back to torchaudio…")
        wav, sr_orig = torchaudio.load(input_path)
        if sr_orig != sr:
            wav = torchaudio.functional.resample(wav, sr_orig, sr)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

    window_size_samples = int(window_size_sec * sr)

    wav_windows = apply_overlap_windowing_waveform(wav, window_size_samples, overlap)
    num_windows = wav_windows.size(0)

    print(f"Processing {num_windows} windows with batch size {batch_size}")
    restored_wav_windows = []

    with tqdm(total=num_windows) as pbar:
        for i in range(0, num_windows, batch_size):
            current_batch_size = min(batch_size, num_windows - i)
            batch_wav_windows = wav_windows[i:i + current_batch_size].to(device)

            restored_wav = process_batch(model, batch_wav_windows, steps, cfg_strength, decoder)
            restored_wav_windows.append(restored_wav)

            del batch_wav_windows
            if device == 'cuda':
                torch.cuda.empty_cache()
            pbar.update(current_batch_size)

    restored_wav_windows = torch.cat(restored_wav_windows, dim=0)

    restored_wav = reconstruct_waveform_from_windows(
        restored_wav_windows, window_size_samples, overlap, original_length=wav.shape[-1]
    )

    restored_wav = restored_wav / max(restored_wav.abs().max().item(), 1e-6)
    if restored_wav.dim() == 1:
        restored_wav = restored_wav.unsqueeze(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving restored audio to {output_path}")
    torchaudio.save(str(output_path), restored_wav, sr, encoding="PCM_F")

    del restored_wav_windows, restored_wav
    if device == 'cuda':
        torch.cuda.empty_cache()

    print("Audio restoration complete ✓")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Audio Restoration – fixed cross‑fade")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--input', type=str, required=True, help="Path to the input audio file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the restored audio file")
    parser.add_argument('--steps', type=int, default=32, help="Number of sampling steps")
    parser.add_argument('--cfg_strength', type=float, default=1.0, help="CFG strength value")
    parser.add_argument('--window_size_sec', type=float, default=5.0, help="Window size in seconds")
    parser.add_argument('--overlap', type=float, default=0.5, help="Overlap ratio for windowing")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for processing")
    parser.add_argument('--decoder', type=str, choices=['bigvgan'], default='bigvgan', help="Decoder type")
    parser.add_argument('--precision', type=str, choices=['float32', 'bfloat16', 'float16'], default='bfloat16', help="Precision for model inference")
    parser.add_argument('--max_memory_mb', type=int, default=16000, help="Maximum GPU memory to use in MB (0 for no limit)")

    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}\nDevice: {device}")
    if device == 'cuda':
        print(f"CUDA: {torch.version.cuda}\nGPU: {torch.cuda.get_device_name()}\nVRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    try:
        print(f"Loading model from {args.checkpoint}")
        model = load_model(args.checkpoint, device, args.decoder)

        # Precision settings
        if device == 'cuda':
            if args.precision == 'bfloat16' and torch.cuda.is_bf16_supported():
                model.voice_restore = model.voice_restore.bfloat16()
                model.bigvgan_model = model.bigvgan_model.bfloat16()
                print('Using bf16')
            elif args.precision == 'float16':
                model.voice_restore = model.voice_restore.half()
                model.bigvgan_model = model.bigvgan_model.half()
                print('Using fp16')
            else:
                print('Using fp32')

        model.voice_restore = model.voice_restore.eval()
        model.bigvgan_model = model.bigvgan_model.eval()

        restore_audio(
            model,
            args.input,
            args.output,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            window_size_sec=args.window_size_sec,
            overlap=args.overlap,
            batch_size=args.batch_size,
            decoder=args.decoder,
            max_memory_mb=args.max_memory_mb
        )

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
