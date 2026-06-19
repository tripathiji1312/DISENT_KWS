from __future__ import annotations
import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
import config
from enrollment.enroll import LFBETransform, enroll_user, save_enrollment, load_enrollment
from models.disent_v2  import DISENT_KWS_v2

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

_G = "\033[92m"; _Y = "\033[93m"; _C = "\033[96m"; _B = "\033[1m"; _R = "\033[0m"


def _bar(score: float, w: int = 40) -> str:
    n = max(0, min(int(score * w), w))
    col = _G if score >= 0.75 else (_Y if score >= 0.50 else _C)
    return col + "█" * n + _R + "░" * (w - n)


class RealTimeDetector:
    BLOCK = 2560  # 160 ms at 16 kHz

    def __init__(self, model, enrollment, device="cpu"):
        self.model     = model.to(device).eval()
        self.p_kw      = enrollment["p_kw"].to(device)
        self.p_spk     = enrollment["p_spk"].to(device)
        self.threshold = enrollment["threshold"]
        self.xform     = LFBETransform()
        self.device    = device
        buf_len        = int(config.SAMPLE_RATE * config.MAX_AUDIO_SEC)
        self._buf      = torch.zeros(1, buf_len)
        self._last_t   = 0.0
        self.model.scorer.reset()

    def callback(self, indata, frames, time_info, status):
        chunk = torch.tensor(indata[:, 0], dtype=torch.float32).unsqueeze(0)
        self._buf = torch.cat([self._buf[:, frames:], chunk], dim=1)
        with torch.no_grad():
            feat = self.xform(self._buf).unsqueeze(0).to(self.device)
            z_p, z_s = self.model(feat, p_spk=self.p_spk, p_kw=self.p_kw)
            smooth, hit = self.model.scorer.detect_streaming(
                z_p, z_s, self.p_kw, self.p_spk, threshold=self.threshold)
        now = time.monotonic()
        if hit and (now - self._last_t) < 1.5:
            hit = False
        if hit:
            self._last_t = now
        label = f" {_B}{_G}🎯 DETECTED!{_R}" if hit else ""
        print(f"\r{_C}Score{_R} {_bar(smooth)} {smooth:5.3f} [τ={self.threshold:.2f}]{label}   ",
              end="", flush=True)

    def run(self):
        if not _SD_OK:
            print("❌ Install sounddevice:  pip install sounddevice"); return
        print(f"\n{_B}DISENT-KWS v2 — Real-Time Demo{_R}")
        print(f"   threshold={self.threshold:.4f}  device={self.device}")
        print(f"   Press {_B}Ctrl+C{_R} to stop.\n")
        with sd.InputStream(callback=self.callback, channels=1,
                             samplerate=config.SAMPLE_RATE,
                             blocksize=self.BLOCK, dtype="float32"):
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print(f"\n\n{_B}Stopped.{_R}\n")


def _load_model(path, device):
    model = DISENT_KWS_v2()
    if path and os.path.exists(path):
        sd_ = torch.load(path, map_location=device)
        if isinstance(sd_, dict) and "model" in sd_:
            sd_ = sd_["model"]
        model.load_state_dict(sd_, strict=False)
        print(f"✅ Loaded {path}")
    else:
        print("⚠️  No weights loaded — random init (testing only)")
    return model.to(device).eval()


def _record_utterance(duration: float = 2.0, mic_device=None) -> torch.Tensor:
    """Record `duration` seconds from microphone. Returns (1, samples) tensor."""
    if not _SD_OK:
        raise RuntimeError("Install sounddevice: pip install sounddevice")
    sr = config.SAMPLE_RATE
    print(f"🎤  Recording (say your keyword now)…", end="", flush=True)
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32",
                   device=mic_device)
    sd.wait()
    print(f"\r✅  Recorded {duration:.1f}s\n")
    return torch.tensor(audio.T, dtype=torch.float32)  # (1, samples)


def _do_live_enroll(args):
    """Record N utterances, then run enrollment."""
    import tempfile
    import os

    if not _SD_OK:
        print("❌ Install sounddevice:  pip install sounddevice"); return

    if args.mic is None:
        try:
            args.mic = sd.default.device[0]
        except Exception:
            print("❌ No microphone found. Run `python -c \"import sounddevice as sd; print(sd.query_devices())\"` to list devices.")
            return

    model = _load_model(args.model, args.device)
    print(f"\n{_B}Live Enrollment{_R}")
    print(f"  Keyword: say it {args.n_record} times when prompted")
    print(f"  Mic:     {args.mic}\n")

    audio_dir = args.out_dir
    os.makedirs(audio_dir, exist_ok=True)
    paths = []

    for i in range(args.n_record):
        input(f"{_B}[{i+1}/{args.n_record}]{_R} Press Enter to record… ")
        wav = _record_utterance(duration=args.duration, mic_device=args.mic)
        p = os.path.join(audio_dir, f"kw_{i:02d}.wav")
        import torchaudio
        torchaudio.save(p, wav, config.SAMPLE_RATE)
        paths.append(p)

    enr = enroll_user(model, paths,
                      background_audio_paths=args.background,
                      target_fa_per_hr=args.target_fa,
                      n_augmented=args.n_aug, device=args.device)
    save_enrollment(enr, args.out)
    print(f"\n{_G}✅  Ready! Run: python src/demo.py detect --enrollment {args.out}{_R}")


def main():
    p = argparse.ArgumentParser(description="DISENT-KWS v2 demo")
    p.add_argument("--device", default="cpu")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("enroll")
    pe.add_argument("--recordings", nargs="+", required=True)
    pe.add_argument("--model",      default="model_final.pt")
    pe.add_argument("--background", nargs="*", default=None)
    pe.add_argument("--target-fa",  type=float, default=1.0)
    pe.add_argument("--n-aug",      type=int,   default=30)
    pe.add_argument("--out",        default="enrollment.pt")

    pr = sub.add_parser("record")
    pr.add_argument("--model",      default="model_final.pt")
    pr.add_argument("--mic",        default=None,
                    help="Microphone device name/ID (default: system default)")
    pr.add_argument("--n-record",   type=int,   default=5,
                    help="Number of utterances to record (default: 5)")
    pr.add_argument("--duration",   type=float, default=2.0,
                    help="Recording duration in seconds (default: 2.0)")
    pr.add_argument("--out",        default="enrollment.pt")
    pr.add_argument("--out-dir",    default="recordings",
                    help="Directory to save raw recordings (default: recordings/)")
    pr.add_argument("--background", nargs="*", default=None)
    pr.add_argument("--target-fa",  type=float, default=1.0)
    pr.add_argument("--n-aug",      type=int,   default=30)

    pd = sub.add_parser("detect")
    pd.add_argument("--enrollment", default="enrollment.pt")
    pd.add_argument("--model",      default="model_final.pt")
    pd.add_argument("--threshold",  type=float, default=None)

    args = p.parse_args()

    if args.cmd == "enroll":
        model = _load_model(args.model, args.device)
        enr   = enroll_user(model, args.recordings,
                             background_audio_paths=args.background,
                             target_fa_per_hr=args.target_fa,
                             n_augmented=args.n_aug, device=args.device)
        save_enrollment(enr, args.out)

    elif args.cmd == "record":
        _do_live_enroll(args)

    elif args.cmd == "detect":
        model = _load_model(args.model, args.device)
        enr   = load_enrollment(args.enrollment)
        if args.threshold is not None:
            enr["threshold"] = args.threshold
        RealTimeDetector(model, enr, device=args.device).run()


if __name__ == "__main__":
    main()
