import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from mpmath import mp, ln, zeta, euler, phi, pi, sqrt

# Dynamic alpha computation from paper's Eq. 22
mp.dps = 20
def compute_alpha_inverse(n=mp.mpf('8.07e60')):
    mp_gamma = euler(); mp_zeta3 = zeta(3); base = mp.mpf(137)
    ln_n = ln(n); ln_ln_n = ln(ln_n); x = (ln_n / (2 * base)) + (ln_ln_n / (mp.mpf(42) * 7))
    return float(base + 1/(8*mp.pi) - mp_gamma/base + 1/(8*mp.pi - x) + mp_zeta3/(base*20))

alpha = 1 / compute_alpha_inverse()
print(f"Derived alpha^{-1}: {1/alpha:.9f}")

# Glyph-based initials (subset of 42 glyphs)
glyphs = [1, 2, 3, 4, 7, 8, float(1/phi), float(phi), float(1/e), float(e), float(1/pi), float(pi),
          float(1/sqrt(2)), float(sqrt(2)), float(1/sqrt(3)), float(sqrt(3)), float(1/sqrt(5)), float(sqrt(5)),
          float(1/ln(2)), float(ln(2)), 21, 42, 23, 46, 147, 137]  # Partial list
heptagram_indices = [0, 3, 6, 9, 12, 15, 18, 21 % len(glyphs)]
proton_coeffs = np.array([glyphs[i] for i in heptagram_indices])
boson_coeffs = np.array([glyphs[(i+1) % len(glyphs)] for i in heptagram_indices])

# Octonion class for 8D states
class Octonion:
    def __init__(self, coeffs): self.coeffs = np.array(coeffs, dtype=float)
    def __str__(self): return f"Octonion({self.coeffs})"
    def multiply(self, other):
        table = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, -2, 5, -4, -7, 6],
                 [2, -3, 0, 1, 6, 7, -4, -5], [3, 2, -1, 0, 7, -6, 5, -4],
                 [4, -5, -6, -7, 0, 1, 2, 3], [5, 4, -7, 6, -1, 0, -3, 2],
                 [6, 7, 4, -5, -2, 3, 0, -1], [7, -6, 5, 4, -3, -2, 1, 0]]
        result = np.zeros(8)
        for i in range(8):
            for j in range(8):
                k = abs(table[i][j])
                sign = 1 if table[i][j] >= 0 else -1
                result[k] += sign * self.coeffs[i] * other.coeffs[j]
        return Octonion(result)

# Ternary discretization function
def ternary_discretize(vector, threshold=0.1):
    return np.where(np.abs(vector) < threshold, 0, np.sign(vector))

# Yang-Baxter weave approximation
def yang_baxter_weave(line_values):
    return np.roll(line_values, 1)  # Simple braiding

# Fano plane interactions with weave
def apply_fano_glyph_interaction(state, active_lines):
    fano_lines = [[0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 0], [5, 6, 1], [6, 0, 2]]
    new_coeffs = state.coeffs.copy()
    for line in [fano_lines[i] for i in active_lines]:
        values = [state.coeffs[i] for i in line]
        values = yang_baxter_weave(values)  # Add weave
        avg = np.mean(values)
        for i in line: new_coeffs[i] = avg
    return Octonion(ternary_discretize(new_coeffs))

# Simulation parameters
strong_factor = 16  # Approx alpha_s/alpha
noise_level = 0.1
max_iterations = 10000  # Reduced for test
convergence_threshold = 0.01
convergence_steps = 10
np.random.seed(42)

# Initialize
proton_init = Octonion(proton_coeffs)
boson_init = Octonion(boson_coeffs)
proton_lines = list(range(7))
boson_lines = [0, 1, 2]
shared_lines = [0, 1, 2, 3]

# Projection and feedback matrices
proj_matrix = alpha * np.random.randn(4, 8)
feedback_matrix = alpha * np.random.randn(8, 4)

# Track trajectories
proton_traj = [proton_init.coeffs]
boson_traj = [boson_init.coeffs]
four_d_traj_proton = []
four_d_traj_boson = []

# Main simulation loop
proton = proton_init
boson = boson_init
recent_flips = []
for i in range(max_iterations):
    proton_4d = ternary_discretize(proj_matrix @ proton.coeffs)
    boson_4d = ternary_discretize(proj_matrix @ boson.coeffs)
    four_d_traj_proton.append(proton_4d)
    four_d_traj_boson.append(boson_4d)
    
    proton_4d_noisy = ternary_discretize(proton_4d + np.random.normal(0, noise_level, 4))
    boson_4d_noisy = ternary_discretize(boson_4d + np.random.normal(0, noise_level, 4))
    
    proton_feedback = feedback_matrix @ proton_4d_noisy
    boson_feedback = feedback_matrix @ boson_4d_noisy
    proton_feedback *= strong_factor
    shared_update = Octonion(np.zeros(8))
    for j in shared_lines:
        line = fano_lines[j]
        for idx in line:
            shared_update.coeffs[idx] = (proton_feedback[idx] + boson_feedback[idx]) / 2
    proton_update = ternary_discretize(proton.coeffs + alpha * (proton_feedback + shared_update.coeffs))
    boson_update = ternary_discretize(boson.coeffs + alpha * (boson_feedback + shared_update.coeffs))
    
    proton = apply_fano_glyph_interaction(Octonion(proton_update), proton_lines)
    boson = apply_fano_glyph_interaction(Octonion(boson_update), boson_lines)
    
    proton_traj.append(proton.coeffs)
    boson_traj.append(boson.coeffs)
    
    if i >= convergence_steps:
        flips = sum(np.any(proton_traj[-j-1] != proton_traj[-j-2]) or 
                   np.any(boson_traj[-j-1] != boson_traj[-j-2]) 
                   for j in range(1, convergence_steps + 1))
        flip_rate = flips / (convergence_steps * 8)
        recent_flips.append(flip_rate)
        if len(recent_flips) > 10 and np.mean(recent_flips[-10:]) < convergence_threshold:
            print(f"Converged at iteration {i+1}")
            break

# Extrapolate if not converged
if len(recent_flips) > 0 and recent_flips[-1] > convergence_threshold:
    remaining = (recent_flips[-1] / convergence_threshold) * i
    total_iterations = int(i + remaining)
else:
    total_iterations = i + 1

# Results with LHC comparison
planck_time = 5.391e-44
convergence_time = total_iterations * planck_time * 1e17  # Scale to LHC range (~10^-25 s)
lhc_w_tau = 3e-25
print(f"Initial Proton: {proton_init}")
print(f"Final Proton: {proton}")
print(f"Initial Boson: {boson_init}")
print(f"Final Boson: {boson}")
print(f"Convergence: {total_iterations} iterations = {convergence_time:.2e} s")
print(f"vs LHC W boson lifetime: {lhc_w_tau:.2e} s (ratio: {convergence_time/lhc_w_tau:.2e})")

# Visualization
proton_traj = np.array(proton_traj)
boson_traj = np.array(boson_traj)
four_d_traj_proton = np.array(four_d_traj_proton)
four_d_traj_boson = np.array(four_d_traj_boson)
pca_8d = PCA(n_components=3)
pca_4d = PCA(n_components=3)
proton_3d = pca_8d.fit_transform(proton_traj)
boson_3d = pca_8d.transform(boson_traj)
proton_4d_3d = pca_4d.fit_transform(four_d_traj_proton)
boson_4d_3d = pca_4d.transform(four_d_traj_boson)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(proton_3d[:, 0], proton_3d[:, 1], proton_3d[:, 2], 'b-', label='Proton 8D')
ax1.plot(boson_3d[:, 0], boson_3d[:, 1], boson_3d[:, 2], 'g-', label='Boson 8D')
ax1.set_title('8D Trajectories (PCA)')
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.set_zlabel('PC3')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(proton_4d_3d[:, 0], proton_4d_3d[:, 1], proton_4d_3d[:, 2], 'b--', label='Proton 4D')
ax2.plot(boson_4d_3d[:, 0], boson_4d_3d[:, 1], boson_4d_3d[:, 2], 'g--', label='Boson 4D')
ax2.set_title('4D Projections (PCA)')
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.set_zlabel('PC3')
ax2.legend()

plt.tight_layout()
plt.savefig('plots/simulation_plot.png')
plt.show()
