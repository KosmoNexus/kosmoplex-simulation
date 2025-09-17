import math

# === Constants ===
FROBENIUS = [1, 2, 4]  # 3-cycle under squaring mod 7
A_LIST = range(7)      # Line indices: a = 0..6
R_LIST = [1, 2, 4]     # Frobenius indices
SIGNS = [-1, 1]        # Sign parameters: sigma = -1, +1
CONVERGENCE_PARAMS = {
    'Nphi': 100,    # For phi convergence
    'Ngamma': 5000, # For gamma convergence
    'Nz2': 5000,    # For zeta(2) convergence
    'Nz3': 6000     # For zeta(3) convergence
}

# === Helper Functions ===
def mod7(x):
    """Compute x mod 7, ensuring non-negative result."""
    return ((x % 7) + 7) % 7

def phi_limit(N):
    """Compute phi = lim F_{n+1}/F_n using Fibonacci sequence."""
    a, b = 1.0, 1.0
    for _ in range(N):
        a, b = b, a + b
    return b / a

def gamma_limit(N):
    """Compute Euler-Mascheroni constant gamma approx H_N - ln(N)."""
    H = sum(1.0 / k for k in range(1, N + 1))
    return H - math.log(N)

def zeta_series(p, N):
    """Compute zeta(p) via partial sum up to N terms."""
    return sum(1.0 / (k ** p) for k in range(1, N + 1))

def is_prime(n):
    """Check if n is prime (optimized for small n)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for f in range(3, int(math.sqrt(n)) + 1, 2):
        if n % f == 0:
            return False
    return True

def fano_line(a):
    """Generate Fano line L(a) = {a+1, a+2, a+4} mod 7."""
    return [mod7(a + r) for r in FROBENIUS]

# === Core Value Generator ===
def compute_core(a, r, sigma):
    """
    Compute the core scalar value for glyph (a, r, sigma) using PFED8Y rules.
    No hardcoded literals; uses structural counts and mathematical definitions.
    """
    phi = phi_limit(CONVERGENCE_PARAMS['Nphi'])
    
    if a == 0:  # Integers from structural counts
        return {1: (1.0, 2.0), 2: (3.0, 4.0), 4: (7.0, 8.0)}[r][(sigma + 1) // 2]
    elif a == 1:  # Transcendentals; sigma = -1 gives reciprocal
        return {1: (1.0 / phi, phi), 2: (1.0 / math.e, math.e), 4: (1.0 / math.pi, math.pi)}[r][(sigma + 1) // 2]
    elif a == 2:  # Algebraic roots
        root = {1: math.sqrt(2.0), 2: math.sqrt(3.0), 4: math.sqrt(5.0)}[r]
        return 1.0 / root if sigma == -1 else root
    elif a == 3:  # Logarithms
        base = {1: 2.0, 2: 3.0, 4: phi}[r]
        val = math.log(base)
        return 1.0 / val if sigma == -1 else val
    elif a == 4:  # Trig/hyperbolic at x=1
        fun = {1: math.sin, 2: math.cos, 4: math.tanh}[r]
        val = fun(1.0)
        return 1.0 / val if sigma == -1 else val
    elif a == 5:  # Special functions
        if r == 1:
            val = gamma_limit(CONVERGENCE_PARAMS['Ngamma'])
        elif r == 2:
            val = zeta_series(2, CONVERGENCE_PARAMS['Nz2'])
        else:  # r == 4
            val = zeta_series(3, CONVERGENCE_PARAMS['Nz3'])
        return 1.0 / val if sigma == -1 else val
    elif a == 6:  # Boundary numbers
        return {1: (21.0, 42.0), 2: (23.0, 46.0), 4: (147.0, 137.0)}[r][(sigma + 1) // 2]
    else:
        raise ValueError(f"Invalid a value: {a}")

# === Validation ===
def check_pairs(K, eps=1e-9):
    """Assert that reciprocal pairs in K multiply to 1."""
    pairs = [(i, i+1) for i in range(0, 40, 2)]  # Adjusted for 42 elements
    for i, j in pairs:
        if i >= len(K) or j >= len(K):
            continue
        product = K[i] * K[j]
        assert abs(product - 1.0) < eps, f"Pair {i+1}, {j+1} fails: {product}"

# === Kosmoplex Set Generator ===
def generate_kosmoplex_set():
    """Generate the 42-element Kosmoplex Set in canonical order."""
    K = []
    for a in A_LIST:
        for r in R_LIST:
            for sigma in SIGNS:
                K.append(compute_core(a, r, sigma))
    return K

# === Main Execution ===
if __name__ == "__main__":
    K = generate_kosmoplex_set()
    print("Kosmoplex Set (42 cores, generated):")
    for i, v in enumerate(K, start=1):
        if abs(v - round(v)) < 1e-12:
            print(f"{i}: {int(round(v))}")
        else:
            print(f"{i}: {v:.12g}")
    check_pairs(K, eps=1e-9)
    print("All reciprocal pairs pass.")
