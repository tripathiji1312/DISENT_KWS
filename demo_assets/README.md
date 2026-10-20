# Demo audio assets

These WAV files are short excerpts from **LibriSpeech dev-clean**, used only to
seed the demo registry and to exercise the fallback upload/replay path.

- Dataset: LibriSpeech ASR corpus (OpenSLR 12), <https://www.openslr.org/12>
- License: Creative Commons Attribution 4.0 (CC BY 4.0)
- Speakers: 1988 (Alice), 6345 (Bob), 3752 (unknown), 3853 (unknown sentence)
- Clips were trimmed/resampled to 16 kHz mono PCM16 for this demo.

`manifest.json` seeds Alice and Bob, both with the shared keyword `stop`.
`unknown_stop.wav` is a held-out speaker saying `stop`; use it to demonstrate
`keyword_unknown_speaker` (no action). `alice_speech.wav` is a held-out Alice
sentence for arbitrary-speech identification.

The files are committed only so the demo can run fully offline. Replace them
with your own recordings for any real deployment.
