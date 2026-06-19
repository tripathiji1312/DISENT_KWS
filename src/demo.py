from __future__ import annotations
import argparse, os, sys, time, subprocess
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import config
import math
from enrollment.enroll import LFBETransform, enroll_user, save_enrollment, load_enrollment
from models.disent_v2  import DISENT_KWS_v2

_G = "\033[92m"; _Y = "\033[93m"; _C = "\033[96m"; _B = "\033[1m"; _R = "\033[0m"

_ALSA_DMIC = "hw:2,0"
_MIC_RATE  = 48000
_MIC_FMT   = "S32_LE"
_MIC_CH    = 2
_BYTES_PER_FRAME = 4 * _MIC_CH


def _bar(score: float, w: int = 40) -> str:
    n = max(0, min(int(score * w), w))
    col = _G if score >= 0.75 else (_Y if score >= 0.50 else _C)
    return col + "█" * n + _R + "░" * (w - n)


def _read_exactly(pipe, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def _alsa_record(duration: float, alsa_device: str = _ALSA_DMIC) -> torch.Tensor:
    n_frames = int(duration * _MIC_RATE)
    cmd = ["arecord", "-D", alsa_device, "-r", str(_MIC_RATE),
           "-c", str(_MIC_CH), "-f", _MIC_FMT, "-t", "raw", "-q",
           "-d", str(int(duration))]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    expected = n_frames * _BYTES_PER_FRAME
    raw = raw[:expected]
    frames = np.frombuffer(raw, dtype=np.int32).reshape(-1, _MIC_CH)
    mono = frames.astype(np.float32).mean(axis=1) / 2147483648.0
    chunk = torch.from_numpy(mono).unsqueeze(0)
    if _MIC_RATE != config.SAMPLE_RATE:
        import torchaudio
        chunk = torchaudio.functional.resample(chunk, _MIC_RATE, config.SAMPLE_RATE)
    return chunk


def _calibrate_threshold(model, p_kw, p_spk, alsa_dev, xform, device,
                         dur=4.0) -> float:
    print(f"\n{_Y}Calibrating: recording {dur:.0f}s of background noise…{_R}")
    print(f"  (stay quiet, don't speak){_R}")
    wav = _alsa_record(dur, alsa_dev)  # (1, N) at 16 kHz
    hop = int(config.SAMPLE_RATE * 0.16)
    scores = []
    for start in range(0, wav.shape[1] - hop + 1, hop):
        chunk = wav[:, start:start + hop]
        with torch.no_grad():
            buf = torch.zeros(1, int(config.SAMPLE_RATE * config.MAX_AUDIO_SEC))
            buf[:, -chunk.shape[1]:] = chunk
            feat = xform(buf).unsqueeze(0).to(device)
            z_p, z_s = model(feat, p_spk=p_spk, p_kw=p_kw)
            raw, _, _ = model.scorer(z_p, z_s, p_kw, p_spk)
            scores.append(raw.item())
    max_s = max(scores) if scores else 0.0
    thresh = max(max_s + 0.15, 0.30)
    print(f"  Background max score={max_s:.3f}  →  threshold={thresh:.3f}{_R}")
    return thresh


class RealTimeDetector:
    BLOCK_SEC = 0.16

    def __init__(self, model, enrollment, device="cpu",
                 alsa_device=_ALSA_DMIC, vad_threshold=0.02):
        self.model      = model.to(device).eval()
        self.p_kw       = enrollment["p_kw"].to(device)
        self.p_spk      = enrollment["p_spk"].to(device)
        self.threshold  = enrollment.get("threshold", config.DEFAULT_THRESHOLD)
        self.xform      = LFBETransform()
        self.device     = device
        self.alsa_dev   = alsa_device
        self.vad_thr    = vad_threshold
        buf_len         = int(config.SAMPLE_RATE * config.MAX_AUDIO_SEC)
        self._buf       = torch.zeros(1, buf_len)
        self._last_t    = 0.0
        self._last_denied_t = 0.0
        self._silent_frames = 0
        self._was_speech = False
        self._pending_deny = False
        self.model.scorer.reset()

    def _voice_energy(self, chunk: torch.Tensor) -> float:
        chunk_centered = chunk - chunk.mean()
        return float(chunk_centered.pow(2).mean().sqrt().item())

    def _process_chunk(self, chunk: torch.Tensor):
        self._buf = torch.cat([self._buf[:, chunk.shape[1]:], chunk], dim=1)

        energy = self._voice_energy(chunk)
        is_speech = energy >= self.vad_thr

        with torch.no_grad():
            feat = self.xform(self._buf).unsqueeze(0).to(self.device)
            z_p, z_s = self.model(feat, p_spk=self.p_spk, p_kw=self.p_kw)
            raw_score, sim_kw, sim_spk = self.model.scorer(z_p, z_s, self.p_kw, self.p_spk)
            raw = raw_score.item()

        ema = self.model.scorer._ema_state.item()
        if is_speech:
            smooth = self.model.scorer.ema_alpha * raw + (1 - self.model.scorer.ema_alpha) * ema
            self._silent_frames = 0
        else:
            self._silent_frames += 1
            smooth = ema * 0.85
            if self._silent_frames >= 6:
                smooth = 0.0
                self.model.scorer.reset()

        self.model.scorer._ema_state.fill_(smooth)
        
        # 1. Check for Valid Hit
                # 1. Check for Valid Hit
        # Raised keyword gate to 0.60 to block "hello samsung" (which scores 0.47)
        # Added speaker gate to 0.50 to ensure it only triggers for you
        is_keyword_match = sim_kw.item() >= 0.60
        is_speaker_match = sim_spk.item() >= 0.50
        is_score_match = smooth >= self.threshold
        
        hit = is_score_match and is_keyword_match and is_speaker_match
        now = time.monotonic()
        
        if hit and (now - self._last_t) < 3.0:  # Increased cooldown to keep art on screen
            hit = False
        
        if hit:
            self._last_t = now
            self._pending_deny = False  # Cancel any pending denial if it's a hit

        # 2. Track Pending Deny 
        # Triggers if you are speaking, it recognizes your voice, but the keyword is wrong
        if is_speech and is_speaker_match and not is_keyword_match:
            self._pending_deny = True

        # 3. Trigger Deny ONLY when user stops speaking
        denied = False
        if not is_speech and self._was_speech and self._pending_deny:
            if (now - self._last_denied_t) >= 3.0:  # Cooldown for denied art
                denied = True
                self._last_denied_t = now
            self._pending_deny = False  # Consume the pending deny
        
        self._was_speech = is_speech

        # 4. Non-Blocking UI Rendering
        if hit:
            os.system('clear' if os.name == 'posix' else 'cls')
            art = (
                f"{_G}{_B}"
                f"======================================================\n"
                f"||                                                  ||\n"
                f"||            🎉  ACCESS GRANTED  🎉               ||\n"
                f"||                                                  ||\n"
                f"======================================================"
                f"{_R}"
            )
            print(f"\n\n{art}\n\n\n")
        elif denied:
            os.system('clear' if os.name == 'posix' else 'cls')
            art = (
                f"\033[91m{_B}"
                f"======================================================\n"
                f"||                                                  ||\n"
                f"||              🚫  ACCESS DENIED  🚫               ||\n"
                f"||                                                  ||\n"
                f"======================================================"
                f"{_R}"
            )
            print(f"\n\n{art}\n\n\n")
        else:
            vad_mark = f"{_C}🎤{_R}" if is_speech else "  "
            print(f"\r{vad_mark} {_C}Score{_R} {_bar(smooth)} {smooth:5.3f} [τ={self.threshold:.2f}] rms={energy:.4f}  kw={sim_kw.item():.2f} spk={sim_spk.item():.2f}   ",
                  end="", flush=True)
    def run(self):
        block_frames = int(_MIC_RATE * self.BLOCK_SEC)
        block_bytes  = block_frames * _BYTES_PER_FRAME

        w_kw = self.model.scorer.w_kw.item()
        w_spk = self.model.scorer.w_spk.item()
        print(f"\n{_B}DISENT-KWS — Real-Time Demo{_R}")
        print(f"   weights: kw={w_kw:.3f}  spk={w_spk:.3f}")
        print(f"   threshold={self.threshold:.4f}  VAD≥{self.vad_thr:.3f}  device={self.device}")
        print(f"   mic=ALSA {self.alsa_dev} @ {_MIC_RATE} Hz  {_C}🎤{_R}=speech  Press Ctrl+C to stop.\n")

        cmd = ["arecord", "-D", self.alsa_dev, "-r", str(_MIC_RATE),
               "-c", str(_MIC_CH), "-f", _MIC_FMT, "-t", "raw", "-q"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        try:
            while True:
                raw = _read_exactly(proc.stdout, block_bytes)
                if not raw or len(raw) < block_bytes:
                    err = proc.stderr.read()
                    if err:
                        print(f"\n❌ arecord error: {err.decode().strip()}")
                    break

                frames = np.frombuffer(raw[:block_bytes], dtype=np.int32).reshape(-1, _MIC_CH)
                mono = frames.astype(np.float32).mean(axis=1) / 2147483648.0
                chunk = torch.from_numpy(mono).unsqueeze(0)
                if _MIC_RATE != config.SAMPLE_RATE:
                    import torchaudio
                    chunk = torchaudio.functional.resample(chunk, _MIC_RATE, config.SAMPLE_RATE)

                self._process_chunk(chunk)
        except KeyboardInterrupt:
            print(f"\n\n{_B}Stopped.{_R}\n")
        finally:
            proc.terminate()
            proc.wait()


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
    
    # FORCE the calibrated weights (0.30 and 0.65) to override the bad checkpoint values
    with torch.no_grad():
        # softplus(x) = target  =>  x = log(e^target - 1)
        model.scorer._w_kw_raw.fill_(math.log(math.exp(0.30) - 1.0))
        model.scorer._w_spk_raw.fill_(math.log(math.exp(0.65) - 1.0))
        
    return model.to(device).eval()


def _record_utterance(duration: float = 2.0, alsa_device=_ALSA_DMIC) -> torch.Tensor:
    print(f"🎤  Recording (say your keyword now)…", end="", flush=True)
    wav = _alsa_record(duration, alsa_device)
    print(f"\r✅  Recorded {duration:.1f}s\n")
    return wav


def _do_live_enroll(args):
    model = _load_model(args.model, args.device)
    alsa_dev = args.alsa_dev or _ALSA_DMIC
    print(f"\n{_B}Live Enrollment{_R}")
    print(f"  Keyword: say it {args.n_record} times when prompted")
    print(f"  Mic:     ALSA {alsa_dev}\n")

    audio_dir = args.out_dir
    os.makedirs(audio_dir, exist_ok=True)
    paths = []

    for i in range(args.n_record):
        input(f"{_B}[{i+1}/{args.n_record}]{_R} Press Enter to record… ")
        wav = _record_utterance(duration=args.duration, alsa_device=alsa_dev)
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
    p = argparse.ArgumentParser(description="DISENT-KWS demo")
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
    pr.add_argument("--alsa-dev",   default=_ALSA_DMIC,
                    help=f"ALSA capture device (default: {_ALSA_DMIC})")
    pr.add_argument("--n-record",   type=int,   default=5)
    pr.add_argument("--duration",   type=float, default=2.0)
    pr.add_argument("--out",        default="enrollment.pt")
    pr.add_argument("--out-dir",    default="recordings")
    pr.add_argument("--background", nargs="*", default=None)
    pr.add_argument("--target-fa",  type=float, default=1.0)
    pr.add_argument("--n-aug",      type=int,   default=30)

    pd = sub.add_parser("detect")
    pd.add_argument("--enrollment", default="enrollment.pt")
    pd.add_argument("--model",      default="model_final.pt")
    pd.add_argument("--alsa-dev",   default=_ALSA_DMIC,
                    help=f"ALSA capture device (default: {_ALSA_DMIC})")
    pd.add_argument("--threshold",  type=float, default=None,
                    help="Detection threshold (overrides enrollment)")
    pd.add_argument("--vad-threshold", type=float, default=0.02,
                    help="Voice activity energy threshold (default: 0.02)")
    pd.add_argument("--auto-threshold", action="store_true",
                    help="Calibrate threshold from background noise before detection")

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
        if args.auto_threshold:
            xform = LFBETransform()
            enr["threshold"] = _calibrate_threshold(
                model, enr["p_kw"].to(args.device), enr["p_spk"].to(args.device),
                args.alsa_dev, xform, args.device)
        RealTimeDetector(model, enr, device=args.device,
                         alsa_device=args.alsa_dev,
                         vad_threshold=args.vad_threshold).run()


if __name__ == "__main__":
    main()
