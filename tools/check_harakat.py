"""Decode a 16 kHz mono float32 .raw file with an exported model, as the app does."""
import json, sys
import numpy as np, onnxruntime as ort

model, vocab, pcm = sys.argv[1:4]
vocab = json.load(open(vocab, encoding="utf8"))
audio = np.fromfile(pcm, dtype=np.float32)[np.newaxis, :]
s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
out = s.run(["log_probs"], {"audio_signal": audio, "length": np.array([audio.shape[1]], np.int64)})[0]
assert out.shape[0] == 1 and out.shape[2] == 1025, out.shape
prev, text = -1, ""
for t in out[0].argmax(-1):
    if t != 1024 and t != prev:
        text += vocab[str(t)]
    prev = t
print(model, "->", text.replace("▁", " ").strip())
