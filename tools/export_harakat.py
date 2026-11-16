"""Export mohammed/fastconformer-quran-ar (CTC branch) to the single-graph ONNX
contract used by Sukun's native CTC engine:
inputs audio_signal [1,T] float32 16 kHz + length [1] int64, output log_probs [1,F,1025]."""
import json, sys
import torch
from nemo.collections.asr.models import ASRModel

nemo_path, out_onnx, out_vocab = sys.argv[1:4]
model = ASRModel.restore_from(nemo_path, map_location="cpu").eval()
if hasattr(model.encoder, "set_default_att_context_size"):
    model.encoder.set_default_att_context_size([-1, -1])  # offline, full context
featurizer = model.preprocessor.featurizer
featurizer.dither = 0.0
featurizer.pad_to = 0


class Graph(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, audio_signal, length):
        features, lengths = self.m.preprocessor(input_signal=audio_signal, length=length)
        encoded, _ = self.m.encoder(audio_signal=features, length=lengths)
        return self.m.ctc_decoder(encoder_output=encoded)


vocab_size = model.tokenizer.vocab_size
assert vocab_size == 1024, f"engine expects 1024 pieces + blank, got {vocab_size}"
json.dump({str(i): model.tokenizer.ids_to_tokens([i])[0] for i in range(vocab_size)},
          open(out_vocab, "w", encoding="utf8"), ensure_ascii=False)

audio = torch.zeros(1, 16000 * 5)
with torch.no_grad():
    torch.onnx.export(Graph(model), (audio, torch.tensor([audio.shape[1]])), out_onnx,
                      input_names=["audio_signal", "length"], output_names=["log_probs"],
                      dynamic_axes={"audio_signal": {1: "T"}, "log_probs": {1: "F"}},
                      opset_version=17, dynamo=False)
print("exported", out_onnx)
