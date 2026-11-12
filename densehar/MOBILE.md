# Mobile Deployment — DenseHAR

## Path: PyTorch → ONNX → TFLite / ONNX Runtime Mobile
1. `python export_onnx.py` produces `densehar.onnx` (dynamic time axis).
2. ONNX → TFLite: `onnx2tf -i densehar.onnx` or ONNX Runtime Mobile directly (often simpler — no conversion, ~ same size).
3. Quantise: dynamic INT8 quantisation typically cuts a <400K-param model to <500 KB and keeps CPU latency low; validate accuracy drop on the test split before shipping.
4. Alternative: ExecuTorch / PyTorch Mobile export (`torch.export`) if TFLite ops (dilated Conv1d) block conversion — fall back to dilation=1 export variant.

## Real-time inference design
- **Sensor:** phone / watch IMU at 50 Hz, ring buffer of 512 samples (10.24 s).
- **Streaming:** slide the buffer by 25–50 samples (0.5–1 s hop), run the net, emit only the labels for the newest hop — earlier labels are already committed. This bounds latency to the hop, not the window.
- **Smoothing:** majority vote over the last 5 emitted labels for display; keep raw labels for transition timestamps (smoothing would erase the boundaries this project exists to find).
- **Budget:** <400K params, 1D convs only — a single forward pass on 512×6 is a few ms on a modern phone CPU; no GPU/NPU required.
- **On-device personalisation (future):** fine-tune only the classification head on the wearer's labelled minutes — the LOSO gap in the README is exactly what this addresses.
