#!/usr/bin/env python3
"""Analyze why confidence weighting causes learning to fail.

Paper loss: L = c * λ3D * L3D - λconf * log(c)

The gradient with respect to confidence c:
∂L/∂c = λ3D * L3D - λconf/c

Setting to zero for optimal c:
c_optimal = λconf / (λ3D * L3D)

The gradient with respect to xyz (through L3D):
∂L/∂xyz = c * λ3D * ∂L3D/∂xyz

Problem: If the model outputs low c, the gradient for xyz becomes tiny!
"""

import numpy as np
import matplotlib.pyplot as plt

# Paper parameters
lambda_3d = 1.0
lambda_conf = 0.2

# L3D values (L1 error)
L3D_values = np.linspace(0.1, 20, 100)

# Optimal confidence for each L3D value
c_optimal = lambda_conf / (lambda_3d * L3D_values)
c_optimal = np.clip(c_optimal, 0.01, 0.99)  # Practical bounds

# The gradient multiplier for xyz learning
grad_multiplier = c_optimal * lambda_3d

print("=" * 60)
print("ANALYSIS: Why Confidence Weighting Causes Learning Failure")
print("=" * 60)

print("\nPaper loss formula:")
print("  L = c * λ3D * L3D - λconf * log(c)")
print(f"\nWith λ3D={lambda_3d}, λconf={lambda_conf}")

print("\n--- Optimal Confidence vs L3D Error ---")
for L3D in [1.0, 2.0, 5.0, 10.0, 20.0]:
    c_opt = lambda_conf / (lambda_3d * L3D)
    c_opt = np.clip(c_opt, 0.01, 0.99)
    grad_mult = c_opt * lambda_3d
    print(f"  L3D={L3D:5.1f} → c_optimal={c_opt:.4f}, xyz gradient multiplier={grad_mult:.4f}")

print("\n--- The Problem ---")
print("When L3D=5 (reasonable initial error):")
L3D = 5.0
c_opt = lambda_conf / (lambda_3d * L3D)
print(f"  c_optimal = λconf/(λ3D*L3D) = {lambda_conf}/({lambda_3d}*{L3D}) = {c_opt:.4f}")
print(f"  xyz gradient = c * λ3D * ∂L3D/∂xyz = {c_opt:.4f} * {lambda_3d} * ∂L3D/∂xyz")
print(f"  = {c_opt * lambda_3d:.4f} * ∂L3D/∂xyz")
print(f"\n  The xyz gradient is multiplied by {c_opt * lambda_3d:.4f} ≈ 0 !")
print("  The model barely updates its xyz predictions!")

print("\n--- Comparison with Simple L1 ---")
print("Simple L1: L = L1(pred, gt)")
print("  xyz gradient = ∂L1/∂xyz = sign(pred - gt)")
print("  Full gradient signal flows to xyz!")

print("\n--- Observed Results ---")
print("Simple L1 (500 steps):   pred_std/gt_std = 0.42 (model learning!)")
print("Paper Loss (50k steps):  pred_std/gt_std = 0.03 (model collapsed!)")

print("\n--- Solution Options ---")
print("1. Remove confidence weighting entirely (simplest)")
print("2. Use fixed confidence schedule (c=1 early, then learn c)")
print("3. Detach confidence from xyz loss: L = c.detach() * L3D - λconf * log(c)")
print("4. Lower λconf or increase λ3D to reduce optimal c")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Optimal c vs L3D
ax1 = axes[0]
ax1.plot(L3D_values, c_optimal, 'b-', linewidth=2)
ax1.axhline(y=0.5, color='r', linestyle='--', label='c=0.5')
ax1.set_xlabel('L3D (L1 error)', fontsize=12)
ax1.set_ylabel('Optimal confidence', fontsize=12)
ax1.set_title('Optimal c vs L3D Error', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Plot 2: XYZ gradient multiplier
ax2 = axes[1]
ax2.plot(L3D_values, grad_multiplier, 'g-', linewidth=2)
ax2.axhline(y=1.0, color='r', linestyle='--', label='Full gradient')
ax2.set_xlabel('L3D (L1 error)', fontsize=12)
ax2.set_ylabel('XYZ gradient multiplier', fontsize=12)
ax2.set_title('Gradient Flow to XYZ', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.5)

# Plot 3: Loss landscape
c_range = np.linspace(0.01, 0.99, 100)
L3D_fixed = 5.0  # Fix L3D to see loss landscape
loss = c_range * lambda_3d * L3D_fixed - lambda_conf * np.log(c_range)
ax3 = axes[2]
ax3.plot(c_range, loss, 'purple', linewidth=2)
c_opt = lambda_conf / (lambda_3d * L3D_fixed)
loss_opt = c_opt * lambda_3d * L3D_fixed - lambda_conf * np.log(c_opt)
ax3.scatter([c_opt], [loss_opt], color='red', s=100, zorder=5, label=f'Optimal c={c_opt:.2f}')
ax3.set_xlabel('Confidence c', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title(f'Loss Landscape (L3D={L3D_fixed})', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure/confidence_weighting_analysis.png', dpi=150)
print("\nSaved analysis plot to figure/confidence_weighting_analysis.png")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("The paper's confidence-weighted loss allows the model to minimize")
print("loss by outputting low confidence, rather than learning accurate 3D.")
print("This explains why AJ=0 and APD3D=0 despite low training loss.")
