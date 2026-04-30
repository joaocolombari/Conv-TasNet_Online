import torch
import torchaudio
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading ConvTasNet...")

bundle = torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX
model = bundle.get_model()
model.eval()

# =====================================================
# EXPORT ONNX (FP32 BASE)
# =====================================================
dummy = torch.randn(1, 1, 8000)

onnx_fp32 = "convtasnet_fp32.onnx"

torch.onnx.export(
    model,
    dummy,
    onnx_fp32,
    input_names=["audio"],
    output_names=["sources"],
    opset_version=17,
    dynamic_axes={
        "audio": {2: "time"},
        "sources": {2: "time"}
    }
)

print("FP32 ONNX exported.")

# =====================================================
# SIZE FP32
# =====================================================
fp32_size = os.path.getsize(onnx_fp32) / (1024**2)
print(f"FP32 ONNX size: {fp32_size:.2f} MB")

# =====================================================
# QUANTIZATION (INT8 DYNAMIC)
# =====================================================
onnx_int8 = "convtasnet_int8.onnx"

quantize_dynamic(
    model_input=onnx_fp32,
    model_output=onnx_int8,
    weight_type=QuantType.QInt8
)

# =====================================================
# SIZE INT8
# =====================================================
int8_size = os.path.getsize(onnx_int8) / (1024**2)

# =====================================================
# RESULTADO
# =====================================================
print("\n================ RESULTS ================")
print(f"FP32 ONNX : {fp32_size:.2f} MB")
print(f"INT8 ONNX : {int8_size:.2f} MB")
print(f"Compression: {fp32_size / int8_size:.2f}x")
print("=========================================")