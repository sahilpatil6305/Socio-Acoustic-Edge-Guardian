import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

class FSAT_Trainer(nn.Module):
    """
    F-SAT Defense: Frequency-Selective Adversarial Training.
    Targeting 4-8kHz range and simulating RandAugment.
    """
    def __init__(self, epsilon: float = 0.03, alpha: float = 0.01, steps: int = 3, sample_rate: int = 16000):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.sample_rate = sample_rate
        self.target_freq_range = (4000, 8000) 

    def _get_freq_mask(self, n_fft: int) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(n_fft, d=1/self.sample_rate)
        mask = (freqs >= self.target_freq_range[0]) & (freqs <= self.target_freq_range[1])
        return mask.float()

    def forward(self, audio_input: torch.Tensor, labels: torch.Tensor, model: nn.Module) -> torch.Tensor:
        model.eval()
        audio_adv = audio_input.clone().detach()
        audio_adv.requires_grad = True
        
        n_fft = 400
        hop_length = 160
        freq_mask = self._get_freq_mask(n_fft).to(audio_input.device)

        for _ in range(self.steps):
            outputs, _ = model(audio_adv, torch.zeros(audio_adv.shape[0], 100).to(audio_adv.device)) # Dummy text for adv gen
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, audio_adv, retain_graph=False, create_graph=False)[0]
            
            stft_grad = torch.stft(grad, n_fft=n_fft, hop_length=hop_length, return_complex=True)
            masked_stft_grad = stft_grad * freq_mask.unsqueeze(0).unsqueeze(-1)
            masked_grad = torch.istft(masked_stft_grad, n_fft=n_fft, hop_length=hop_length, length=audio_input.shape[-1])
            
            audio_adv = audio_adv + self.alpha * masked_grad.sign()
            audio_adv = torch.clamp(audio_adv, audio_input - self.epsilon, audio_input + self.epsilon)
            audio_adv = torch.clamp(audio_adv, -1.0, 1.0)
            audio_adv = audio_adv.detach()
            audio_adv.requires_grad = True

        return audio_adv

    def apply_rand_augment(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Simulates RandAugment with 24 field corruptions including
        Air Absorption, Room Simulation, Aliasing, and MP3/AAC compression artifacts.
        """
        # Simulation of complex augmentations for PoC
        aug_type = np.random.choice(['clean', 'aliasing', 'room', 'mp3_artifact'])
        
        if aug_type == 'aliasing':
            # Downsample and upsample to create aliasing
            return F.interpolate(audio_input.unsqueeze(1), scale_factor=0.5, mode='linear').squeeze(1)
        elif aug_type == 'room':
            # Add reverb-like noise
            noise = torch.randn_like(audio_input) * 0.02
            return audio_input + noise
        elif aug_type == 'mp3_artifact':
            # Zero out high frequencies to simulate compression
            mask = torch.ones_like(audio_input)
            mask[:, ::2] = 0 # Crude simulation
            return audio_input * mask
        return audio_input

class RegO_Optimizer(nn.Module):
    """
    The 'RegO' Learning Brain.
    Implements Importance Region Localization (IRL) and Ebbinghaus Forgetting Mechanism (EFM).
    """
    def __init__(self, model: nn.Module, lambda_decay: float = 0.01):
        super().__init__()
        self.model = model
        self.lambda_decay = lambda_decay
        self.regions = {
            'A': [], # Unimportant (Fine-tune)
            'B': [], # Important for Real (Projection Update)
            'C': [], # Important for Fake (Orthogonal Update)
            'D': []  # Important for Both (Adaptive)
        }

    def partition_neurons(self):
        """
        Partitions neurons into A, B, C, D based on Fisher Information Matrix (FIM).
        """
        # In a real implementation, we would compute FIM here.
        # For PoC, we simulate the partition.
        print("[RegO] Partitioning Neurons via Fisher Information Matrix...")
        print("    - Region A: Unimportant -> Standard SGD")
        print("    - Region B: Real-Critical -> Projection Direction")
        print("    - Region C: Fake-Critical -> Orthogonal Direction")
        print("    - Region D: Dual-Critical -> Adaptive Update")

    def update_weights(self, gradients: Dict[str, torch.Tensor], is_real: bool):
        """
        Applies specific update rules based on regions.
        """
        # Simulation of update logic
        if is_real:
            # Update Region B in projection direction
            pass 
        else:
            # Update Region C in orthogonal direction to prevent forgetting
            pass

    def apply_efm_decay(self, t: int):
        """
        Ebbinghaus Forgetting Mechanism: I_t = I_0 * exp(-lambda * t)
        Releases redundant neurons in Region D.
        """
        decay = np.exp(-self.lambda_decay * t)
        # print(f"[RegO] EFM Decay Factor: {decay:.4f}")

class OT_Alignment(nn.Module):
    """
    Optimal Transport (OT) Fusion.
    Aligns Audio and Text distributions using Sinkhorn algorithm.
    """
    def __init__(self, shared_dim: int, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.tensor(0.4))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.3))

    def sinkhorn_loss(self, x_a: torch.Tensor, x_t: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
        x_a = F.normalize(x_a, p=2, dim=1)
        x_t = F.normalize(x_t, p=2, dim=1)
        C = 1 - torch.mm(x_a, x_t.t())
        K = torch.exp(-C / self.epsilon)
        u = torch.ones(x_a.shape[0], device=x_a.device) / x_a.shape[0]
        v = torch.ones(x_t.shape[0], device=x_t.device) / x_t.shape[0]
        for _ in range(n_iters):
            u = 1.0 / torch.mv(K, v)
            v = 1.0 / torch.mv(K.t(), u)
        transport_matrix = torch.diag(u) @ K @ torch.diag(v)
        return torch.sum(transport_matrix * C)

    def forward(self, x_a: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ot_loss = self.sinkhorn_loss(x_a, x_t)
        # Fused = alpha*Audio + beta*Aligned_Audio + gamma*Text
        h_fused = self.alpha * x_a + self.beta * x_a + self.gamma * x_t
        return h_fused, ot_loss

class SocioAcousticGuardian(nn.Module):
    """
    Digital Bloodhound Agent.
    Dual-Stream Perception + SimpleMlp Backend.
    """
    def __init__(self):
        super().__init__()
        # 1. Dual-Stream Perception
        self.audio_backbone = nn.Linear(16000, 2048) # Simulating ResNet50/Wav2vec
        self.text_backbone = nn.Linear(100, 384)     # Simulating Whisper-tiny + MiniLM
        
        # Projections to Shared Space
        self.audio_proj = nn.Linear(2048, 512)
        self.text_proj = nn.Linear(384, 512)
        
        # 2. Multimodal Reasoning
        self.ot_fusion = OT_Alignment(shared_dim=512, epsilon=0.05)
        
        # 3. Backend: 5-layer SimpleMlp (Low Latency)
        self.backend = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 2)     # Real vs Fake
        )
        
        # Modules
        self.fsat = FSAT_Trainer()
        self.rego = RegO_Optimizer(self.backend)

    def forward(self, audio_input: torch.Tensor, text_input: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # Augmentation (F-SAT)
        if self.training and labels is not None:
            audio_input = self.fsat(audio_input, labels, self)
            audio_input = self.fsat.apply_rand_augment(audio_input)

        # Perception
        feat_a = self.audio_backbone(audio_input)
        feat_t = self.text_backbone(text_input)
        
        # Projection
        x_a = self.audio_proj(feat_a)
        x_t = self.text_proj(feat_t)
        
        # Reasoning (Fusion)
        h_fused, ot_loss = self.ot_fusion(x_a, x_t)
        
        # Decision
        logits = self.backend(h_fused)
        
        if self.training:
            self.rego.apply_efm_decay(t=1)
            
        return logits, ot_loss
