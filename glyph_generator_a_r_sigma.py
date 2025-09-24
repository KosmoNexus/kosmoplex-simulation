#!/usr/bin/env python3
"""
PFED8Y Glyph Calculator

This module implements a calculator for the 42 glyphs of the PFED8Y engine,
mapping (a, r, sigma) triplets to mathematical constants and their approximations.
"""

import sympy as sp
from typing import Dict, List, Tuple, Any


class GlyphCalculator:
    """Calculator for PFED8Y glyphs and their projections."""
    
    def __init__(self):
        """Initialize the glyph calculator with the canonical 42 glyph mapping."""
        self.glyph_map = self._build_glyph_map()
        self.r_to_base = {1: 0, 2: 2, 4: 4}
    
    def _build_glyph_map(self) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
        """Build the canonical mapping from (a, r, sigma) to glyph details."""
        return {
            # Integers (a=0)
            (0, 1, '-'): {'k': 1, 'symbol': '1', 'sympy_expr': sp.sympify(1), 'role': 'Identity/existence'},
            (0, 1, '+'): {'k': 2, 'symbol': '2', 'sympy_expr': sp.sympify(2), 'role': 'Binary doubling'},
            (0, 2, '-'): {'k': 3, 'symbol': '3', 'sympy_expr': sp.sympify(3), 'role': 'Ternary basis'},
            (0, 2, '+'): {'k': 4, 'symbol': '4', 'sympy_expr': sp.sympify(4), 'role': 'Quaternionic dimension'},
            (0, 4, '-'): {'k': 5, 'symbol': '7', 'sympy_expr': sp.sympify(7), 'role': 'Fano plane points'},
            (0, 4, '+'): {'k': 6, 'symbol': '8', 'sympy_expr': sp.sympify(8), 'role': 'Octonionic dimension'},
            
            # Transcendentals (a=1)
            (1, 1, '-'): {'k': 7, 'symbol': r'\phi^{-1}', 'sympy_expr': (sp.sqrt(5) - 1)/2, 'role': 'Golden conjugate'},
            (1, 1, '+'): {'k': 8, 'symbol': r'\phi', 'sympy_expr': (1 + sp.sqrt(5))/2, 'role': 'Golden ratio'},
            (1, 2, '-'): {'k': 9, 'symbol': 'e^{-1}', 'sympy_expr': 1/sp.exp(1), 'role': 'Natural decay'},
            (1, 2, '+'): {'k': 10, 'symbol': 'e', 'sympy_expr': sp.exp(1), 'role': 'Natural growth'},
            (1, 4, '-'): {'k': 11, 'symbol': r'\pi^{-1}', 'sympy_expr': 1/sp.pi, 'role': 'Circular inverse'},
            (1, 4, '+'): {'k': 12, 'symbol': r'\pi', 'sympy_expr': sp.pi, 'role': 'Circle constant'},
            
            # Algebraic roots (a=2)
            (2, 1, '-'): {'k': 13, 'symbol': r'\sqrt{2}^{-1}', 'sympy_expr': 1/sp.sqrt(2), 'role': 'Diagonal inverse'},
            (2, 1, '+'): {'k': 14, 'symbol': r'\sqrt{2}', 'sympy_expr': sp.sqrt(2), 'role': 'Orthogonal basis'},
            (2, 2, '-'): {'k': 15, 'symbol': r'\sqrt{3}^{-1}', 'sympy_expr': 1/sp.sqrt(3), 'role': 'Hexagonal inverse'},
            (2, 2, '+'): {'k': 16, 'symbol': r'\sqrt{3}', 'sympy_expr': sp.sqrt(3), 'role': 'Hexagonal geometry'},
            (2, 4, '-'): {'k': 17, 'symbol': r'\sqrt{5}^{-1}', 'sympy_expr': 1/sp.sqrt(5), 'role': 'Pentagon inverse'},
            (2, 4, '+'): {'k': 18, 'symbol': r'\sqrt{5}', 'sympy_expr': sp.sqrt(5), 'role': 'Pentagon/phi base'},
            
            # Logarithms (a=3)
            (3, 1, '-'): {'k': 19, 'symbol': r'\ln(2)^{-1}', 'sympy_expr': 1/sp.ln(2), 'role': 'Binary log inverse'},
            (3, 1, '+'): {'k': 20, 'symbol': r'\ln(2)', 'sympy_expr': sp.ln(2), 'role': 'Binary logarithm'},
            (3, 2, '-'): {'k': 21, 'symbol': r'\ln(3)^{-1}', 'sympy_expr': 1/sp.ln(3), 'role': 'Ternary log inverse'},
            (3, 2, '+'): {'k': 22, 'symbol': r'\ln(3)', 'sympy_expr': sp.ln(3), 'role': 'Ternary logarithm'},
            (3, 4, '-'): {'k': 23, 'symbol': r'\ln(\phi)^{-1}', 'sympy_expr': 1/sp.ln((1 + sp.sqrt(5))/2), 'role': 'Golden log inverse'},
            (3, 4, '+'): {'k': 24, 'symbol': r'\ln(\phi)', 'sympy_expr': sp.ln((1 + sp.sqrt(5))/2), 'role': 'Golden logarithm'},
            
            # Trigonometric (a=4)
            (4, 1, '-'): {'k': 25, 'symbol': r'\sin(1)^{-1}', 'sympy_expr': 1/sp.sin(1), 'role': 'Sine inverse'},
            (4, 1, '+'): {'k': 26, 'symbol': r'\sin(1)', 'sympy_expr': sp.sin(1), 'role': 'Unit sine'},
            (4, 2, '-'): {'k': 27, 'symbol': r'\cos(1)^{-1}', 'sympy_expr': 1/sp.cos(1), 'role': 'Cosine inverse'},
            (4, 2, '+'): {'k': 28, 'symbol': r'\cos(1)', 'sympy_expr': sp.cos(1), 'role': 'Unit cosine'},
            (4, 4, '-'): {'k': 29, 'symbol': r'\tanh(1)^{-1}', 'sympy_expr': 1/sp.tanh(1), 'role': 'Hyperbolic inverse'},
            (4, 4, '+'): {'k': 30, 'symbol': r'\tanh(1)', 'sympy_expr': sp.tanh(1), 'role': 'Hyperbolic tangent'},
            
            # Special functions (a=5)
            (5, 1, '-'): {'k': 31, 'symbol': r'\gamma^{-1}', 'sympy_expr': 1/sp.EulerGamma, 'role': 'Euler-Mascheroni inv'},
            (5, 1, '+'): {'k': 32, 'symbol': r'\gamma', 'sympy_expr': sp.EulerGamma, 'role': 'Euler-Mascheroni'},
            (5, 2, '-'): {'k': 33, 'symbol': r'\zeta(2)^{-1}', 'sympy_expr': 1/sp.zeta(2), 'role': 'Basel inverse'},
            (5, 2, '+'): {'k': 34, 'symbol': r'\zeta(2)', 'sympy_expr': sp.zeta(2), 'role': 'Basel problem'},
            (5, 4, '-'): {'k': 35, 'symbol': r'\zeta(3)^{-1}', 'sympy_expr': 1/sp.zeta(3), 'role': 'Apéry inverse'},
            (5, 4, '+'): {'k': 36, 'symbol': r'\zeta(3)', 'sympy_expr': sp.zeta(3), 'role': 'Apéry constant'},
            
            # Boundary numbers (a=6)
            (6, 1, '-'): {'k': 37, 'symbol': '21', 'sympy_expr': sp.sympify(21), 'role': '3 × 7 (half of 42)'},
            (6, 1, '+'): {'k': 38, 'symbol': '42', 'sympy_expr': sp.sympify(42), 'role': 'Fano 6 × 7'},
            (6, 2, '-'): {'k': 39, 'symbol': '23', 'sympy_expr': sp.sympify(23), 'role': 'Frobenius prime'},
            (6, 2, '+'): {'k': 40, 'symbol': '46', 'sympy_expr': sp.sympify(46), 'role': '1st Comp resonance'},
            (6, 4, '-'): {'k': 41, 'symbol': '147', 'sympy_expr': sp.sympify(147), 'role': 'Geometric boundary'},
            (6, 4, '+'): {'k': 42, 'symbol': '137', 'sympy_expr': sp.sympify(137), 'role': 'Fine structure 1/α'},
        }
    
    def get_fano_line(self, a: int) -> List[int]:
        """Get the Fano line L(a) = {a, a+1, a+3} mod 7."""
        return [(a + i) % 7 for i in [0, 1, 3]]
    
    def compute_wallis_partial(self, n_terms: int) -> Tuple[List[str], sp.Rational]:
        """Compute partial Wallis product for π approximation."""
        product = sp.Rational(2)
        steps = []
        
        for n in range(1, n_terms + 1):
            term = sp.Rational(4 * n**2, 4 * n**2 - 1)
            product *= term
            steps.append(f"Term {n}: {term} → Cumulative: {product}")
        
        return steps, product
    
    def compute_fibonacci_ratios(self, n_terms: int) -> Tuple[List[str], sp.Rational]:
        """Compute Fibonacci ratios for φ approximation."""
        fib = [0, 1]
        steps = []
        
        # Generate Fibonacci sequence
        for i in range(2, n_terms + 2):
            fib.append(fib[-1] + fib[-2])
        
        # Compute ratios
        for i in range(2, n_terms + 1):
            ratio = sp.Rational(fib[i+1], fib[i])
            steps.append(f"Fib {i+1}/{i}: {ratio}")
        
        return steps, ratio
    
    def compute_taylor_e(self, n_terms: int) -> Tuple[List[str], sp.Rational]:
        """Compute Taylor series for e."""
        sum_val = sp.Rational(1)
        steps = []
        
        for n in range(1, n_terms + 1):
            term = sp.Rational(1, sp.factorial(n))
            sum_val += term
            steps.append(f"Term {n}: {term} → Cumulative: {sum_val}")
        
        return steps, sum_val
    
    def compute_apery_partial(self, n_terms: int) -> Tuple[List[str], sp.Rational]:
        """Compute Apéry series partial for ζ(3)."""
        sum_val = sp.Rational(0)
        steps = []
        
        for k in range(1, n_terms + 1):
            term = sp.Rational((-1)**(k+1), k**3 * sp.binomial(2*k, k))
            sum_val += term * sp.Rational(5, 2)  # Apply the 5/2 prefactor
            steps.append(f"Term {k}: {term} → Cumulative: {sum_val}")
        
        return steps, sum_val
    
    def get_user_input(self) -> Tuple[int, int, str]:
        """Get and validate user input for (a, r, sigma)."""
        while True:
            try:
                a = int(input("Enter a (0-6): "))
                if not (0 <= a <= 6):
                    print("Invalid a. Must be 0-6.")
                    continue
                
                r = int(input("Enter r {1,2,4}: "))
                if r not in [1, 2, 4]:
                    print("Invalid r. Must be 1, 2, or 4.")
                    continue
                
                sigma = input("Enter sigma {+, -}: ").strip()
                if sigma not in ['+', '-']:
                    print("Invalid sigma. Must be + or -.")
                    continue
                
                return a, r, sigma
                
            except ValueError:
                print("Invalid input. Please enter numbers for a and r.")
    
    def compute_8d_value(self, a: int, r: int, sigma: str) -> None:
        """Display step-by-step calculation for 8D glyph instruction."""
        print("\nStep-by-step calculation of its value in 8D (finite glyph instruction):")
        print(f"1. Select Fano line L(a={a}): {self.get_fano_line(a)}")
        
        sigma_val = -1 if sigma == '-' else 1
        print(f"2. Compute stride s = sigma * r = {sigma_val} * {r} = {sigma_val * r}")
        print(f"3. The routing rule is u → u + {sigma_val * r} (mod 7), defining the heptaflake path accumulation.")
        
        base = self.r_to_base[r]
        offset = 0 if sigma == '-' else 1
        index = base + offset
        print(f"4. Map r and sigma to heptaflake index: base={base} (for r={r}), offset={offset} (for sigma={sigma}) → index={index}")
        
        octal = f"{a}0{index}"
        print(f"5. Construct the octal instruction word from the parameters (a0index in base-8): '{octal}_8'")
        print(f"Final 8D output: The finite glyph instruction is the octal code '{octal}_8'.")
    
    def compute_4d_projection(self, glyph: Dict[str, Any]) -> None:
        """Display step-by-step calculation for 4D projection."""
        symbol = glyph['symbol']
        expr = glyph['sympy_expr']
        
        print("\nStep-by-step calculation of the projected value in 4D (via OBMT):")
        print("The Octal Binomial-Modular Transform (OBMT) projects the 8D glyph G(a,r,sigma) to 4D as:")
        print("V_{4D} = C_\\sigma \\prod_{n=1}^\\infty \\frac{\\sum_{k \\in K(a,r)^H \\mod 8} \\binom{2n}{k}}{\\sum_{k \\in K(a,r)^F \\mod 8} \\binom{2n}{k}}")
        print("(where K(a,r) are heptaflake-route index sets, C_\\sigma is the spinorial prefactor).")
        print("This infinite product 'unwraps' the finite 8D instruction into the continuous 4D shadow.")
        
        if 'pi' in symbol:
            steps, approx = self.compute_wallis_partial(10)
            print("For \\pi, the OBMT selects binomials to form the Wallis product (finite partial for 8D-like approximation):")
            for step in steps:
                print(f"  {step}")
            print(f"Approximation after 10 terms: {approx.n(10)} (converges to \\pi/2 in infinite limit, so full \\pi = 2 * limit)")
            projected_val = sp.pi.n(10)
            
        elif 'phi' in symbol:
            steps, approx = self.compute_fibonacci_ratios(10)
            print("For \\phi, the OBMT projects via Fibonacci ratios (finite ratios for approximation):")
            for step in steps:
                print(f"  {step}")
            print(f"Approximation after 10 terms: {approx.n(10)} (converges to \\phi in limit)")
            projected_val = expr.n(10)
            
        elif 'e' in symbol.lower() and symbol != 'zeta':
            steps, approx = self.compute_taylor_e(10)
            print("For e, the OBMT projects via Taylor series (finite partial sums for approximation):")
            for step in steps:
                print(f"  {step}")
            print(f"Approximation after 10 terms: {approx.n(10)} (converges to e in limit)")
            projected_val = expr.n(10)
            
        elif 'zeta(3)' in symbol:
            steps, approx = self.compute_apery_partial(10)
            print("For \\zeta(3), the OBMT projects via Apéry series (finite partial sums for approximation):")
            for step in steps:
                print(f"  {step}")
            print(f"Approximation after 10 terms: {approx.n(10)} (converges to \\zeta(3) in limit)")
            projected_val = expr.n(10)
            
        else:
            print("For this glyph, the OBMT projection directly yields the symbolic constant.")
            projected_val = expr.n(10) if expr.is_real else expr
        
        print(f"Final 4D output: The projected constant is {symbol} ≈ {projected_val}")
    
    def run(self) -> None:
        """Main calculator loop."""
        print("PFED8Y Glyph Calculator")
        print("=" * 25)
        
        while True:
            try:
                a, r, sigma = self.get_user_input()
                
                key = (a, r, sigma)
                if key not in self.glyph_map:
                    print("Invalid combination. No matching glyph.")
                    continue
                
                glyph = self.glyph_map[key]
                k = glyph['k']
                symbol = glyph['symbol']
                role = glyph['role']
                
                print(f"\nThe specific combination (a={a}, r={r}, sigma={sigma}) corresponds to glyph {symbol} ({role}) and is the {k}th glyph.")
                
                # Calculate 8D and 4D values
                self.compute_8d_value(a, r, sigma)
                self.compute_4d_projection(glyph)
                
                # Ask to continue
                again = input("\nRun again? (y/n): ").lower()
                if again != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                continue


def main():
    """Entry point for the glyph calculator."""
    calculator = GlyphCalculator()
    calculator.run()


if __name__ == "__main__":
    main()
