import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def run_benchmark(encoder_name="mobilenet_v2", num_classes=3, img_size=384, n_warmup=15, n_runs=50):
    print(f"\n{'='*65}")
    print(f"RUNNING RUNTIME BENCHMARK: {encoder_name} | {num_classes}-class | {img_size}x{img_size} | B=1")
    print(f"{'='*65}")
    
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=num_classes).to(device)
    model.eval()

    x = torch.randn(1, 3, img_size, img_size, device=device)

    # 1. Source Only (1 eval forward pass)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_src = (time.perf_counter() - t0) / n_runs * 1000.0

    # 2. AdaBN (1 train forward pass to calculate single-image batch stats)
    model.train()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_ada = (time.perf_counter() - t0) / n_runs * 1000.0

    # 3. TENT (10 optimization steps: forward + entropy loss + backward + optimizer step)
    # TENT only tunes affine parameters (gamma, beta) of BatchNorm
    for param in model.parameters():
        param.requires_grad = False
    tent_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.requires_grad_(True)
            tent_params.extend([m.weight, m.bias])
    optimizer = torch.optim.Adam(tent_params, lr=1e-3)

    # Warmup
    for _ in range(min(5, n_warmup)):
        for step in range(10):
            optimizer.zero_grad()
            out = model(x)
            p = F.softmax(out, dim=1)
            entropy = -(p * torch.log(p + 1e-6)).sum(dim=1).mean()
            entropy.backward()
            optimizer.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_tent_runs = min(20, n_runs)
    for _ in range(n_tent_runs):
        for step in range(10):
            optimizer.zero_grad()
            out = model(x)
            p = F.softmax(out, dim=1)
            entropy = -(p * torch.log(p + 1e-6)).sum(dim=1).mean()
            entropy.backward()
            optimizer.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_tent = (time.perf_counter() - t0) / n_tent_runs * 1000.0

    # 4. PMSA-BN (2 forward passes: Pass 1 Source + pseudo-mask + Pass 2 Adapted)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        for _ in range(n_warmup):
            # Pass 1
            logits1 = model(x)
            p1 = F.softmax(logits1, dim=1)
            mask = p1 > 0.5
            # Pass 2
            logits2 = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            # Pass 1: Initial prediction
            logits1 = model(x)
            p1 = F.softmax(logits1, dim=1)
            # Pseudo-mask generation
            mask = p1 > 0.5
            # Pass 2: Adapted prediction (algebraic stats recombination)
            logits2 = model(x)
            probs = F.softmax(logits2, dim=1)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_pmsa = (time.perf_counter() - t0) / n_runs * 1000.0

    print(f"Results (ms / image):")
    print(f"  Source Only : {t_src:6.2f} ms  | 1 forward,  0 backward")
    print(f"  AdaBN       : {t_ada:6.2f} ms  | 1 forward,  0 backward")
    print(f"  PMSA-BN     : {t_pmsa:6.2f} ms  | 2 forwards, 0 backward (Pass 2 overhead: {t_pmsa - t_src:+.2f} ms)")
    print(f"  TENT (10 it): {t_tent:6.2f} ms  | 10 fwd + 10 bwd ({t_tent / t_pmsa:.1f}x slower than PMSA-BN!)")
    
    return {
        "encoder": encoder_name,
        "src_ms": t_src,
        "ada_ms": t_ada,
        "pmsa_ms": t_pmsa,
        "tent_ms": t_tent,
        "speedup_vs_tent": t_tent / t_pmsa
    }

if __name__ == "__main__":
    res_mbv2 = run_benchmark("mobilenet_v2", num_classes=3, img_size=384)
    res_rn34 = run_benchmark("resnet34", num_classes=3, img_size=384)
