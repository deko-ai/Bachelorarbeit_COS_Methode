import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si

# --- 1. Der "wahre" Preis zum Vergleich (Black-Scholes Formel) ---
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price

# --- 2. Die Charakteristische Funktion ---
def cf_black_scholes(u, T, r, sigma, S0):
    x0 = np.log(S0)
    mu = r - 0.5 * sigma**2
    phi = np.exp(1j * u * x0 + 1j * u * mu * T - 0.5 * sigma**2 * u**2 * T)
    return phi

# --- 3. Die COS-Methode (Korrigierte Version) ---
def cos_method_call(S0, K, T, r, sigma, N, L):
    # a) Integrationsgrenzen bestimmen
    x0 = np.log(S0)
    c1 = x0 + (r - 0.5 * sigma**2) * T
    c2 = sigma**2 * T
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)
    
    k = np.arange(0, N)
    u_k = k * np.pi / (b - a)
    
    # b) Fourier-Koeffizienten Fk
    phi_k = cf_black_scholes(u_k, T, r, sigma, S0) * np.exp(-1j * u_k * a)
    F_k = np.real(phi_k)
    F_k[0] = 0.5 * F_k[0]

    # c) Payoff-Koeffizienten Vk (Analytische Lösung)
    # x_star ist der Log-Strike, ab dem der Call "im Geld" ist
    x_star = np.log(K)
    c = x_star
    d = b

    # Stabile Chi-Funktion
    def chi_func(k, a, b, c, d):
        p = k * np.pi / (b - a)
        # Nenner ist immer >= 1, daher keine Division durch 0
        res = (np.cos(p * (d - a)) * np.exp(d) - np.cos(p * (c - a)) * np.exp(c) + 
               p * np.sin(p * (d - a)) * np.exp(d) - p * np.sin(p * (c - a)) * np.exp(c)) / (1 + p**2)
        return res

    # Stabile Psi-Funktion (behandelt k=0 separat ohne if-Abfrage im Array)
    def psi_func(k, a, b, c, d):
        p = k * np.pi / (b - a)
        res = np.zeros_like(p)
        # Fall k = 0
        res[0] = d - c
        # Fall k > 0 (Vermeidung Division durch Null)
        res[1:] = (np.sin(p[1:] * (d - a)) - np.sin(p[1:] * (c - a))) / p[1:]
        return res

    V_k = 2 / (b - a) * K * (chi_func(k, a, b, c, d) - psi_func(k, a, b, c, d))

    # d) Finale Summe
    option_price = np.exp(-r * T) * np.sum(F_k * V_k)
    return option_price

# --- Ausführung ---
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
N, L = 128, 10

bs_p = black_scholes_call(S0, K, T, r, sigma)
cos_p = cos_method_call(S0, K, T, r, sigma, N, L)

print(f"Ergebnis: {cos_p:.6f} EUR (Fehler: {abs(bs_p - cos_p):.2e})")

# Plot erstellen
u_plot = np.linspace(-50, 50, 500)
cf_vals = [cf_black_scholes(u, T, r, sigma, S0).real for u in u_plot]
plt.figure(figsize=(8, 4))
plt.plot(u_plot, cf_vals, color='red', label='Re[phi(u)]')
plt.grid(True, linestyle='--')
plt.savefig('figures/plot_cf.pdf')
print("Grafik erstellt! ✅")