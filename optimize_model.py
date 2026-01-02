import torch
import torch.nn as nn
import time
import os
from rego_fsat_classes import SocioAcousticGuardian

def optimize_agent():
    print("="*60)
    print("Socio-Acoustic Edge Guardian | Optimization Module")
    print("Target: Int8 Quantization & TorchScript Tracing")
    print("="*60)

    # 1. Load Original Model
    print("\n[1] Loading Baseline Agent...")
    model = SocioAcousticGuardian()
    model.eval()
    
    # Calculate Baseline Size
    torch.save(model.state_dict(), "temp_baseline.pt")
    baseline_size = os.path.getsize("temp_baseline.pt") / 1024 / 1024
    print(f"    - Baseline Size: {baseline_size:.2f} MB")

    # 2. Apply Dynamic Quantization
    print("\n[2] Applying Dynamic Quantization (Int8)...")
    # Quantize Linear and LSTM layers (if any) to Int8
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # Calculate Quantized Size
    torch.save(quantized_model.state_dict(), "agent_quantized.pt")
    quantized_size = os.path.getsize("agent_quantized.pt") / 1024 / 1024
    print(f"    - Quantized Size: {quantized_size:.2f} MB")
    print(f"    - Reduction: {(1 - quantized_size/baseline_size)*100:.1f}%")

    # 3. TorchScript Conversion (Tracing)
    print("\n[3] Converting to TorchScript (JIT Tracing)...")
    # Create dummy input for tracing
    example_audio = torch.randn(1, 16000)
    example_text = torch.randn(1, 100)
    
    try:
        # Tracing the quantized model
        traced_model = torch.jit.trace(quantized_model, (example_audio, example_text))
        torch.jit.save(traced_model, "agent_optimized.pt")
        print("    - Success: Model traced and saved as 'agent_optimized.pt'")
    except Exception as e:
        print(f"    - Error during tracing: {e}")
        return

    # 4. Benchmarking
    print("\n[4] Benchmarking Inference Speed (Avg of 100 runs)...")
    
    def benchmark(m, name):
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = m(example_audio, example_text)
                times.append(time.time() - start)
        avg_time = sum(times) / len(times) * 1000
        print(f"    - {name}: {avg_time:.2f} ms")
        return avg_time

    base_time = benchmark(model, "Baseline (Float32)")
    opt_time = benchmark(traced_model, "Optimized (Int8 + JIT)")
    
    print(f"\n[Result] Speedup: {base_time / opt_time:.2f}x")
    
    # Cleanup
    if os.path.exists("temp_baseline.pt"):
        os.remove("temp_baseline.pt")
    
    print("\n" + "="*60)
    print("Optimization Complete. Deploy 'agent_optimized.pt' to edge devices.")
    print("="*60)

if __name__ == "__main__":
    optimize_agent()
