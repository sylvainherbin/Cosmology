# Script Python pour l'Analyse des Données SNIa 
# Inclus le chargement de la matrice de covariance complète et le calcul des incertitudes sur les paramètres

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import io 
import requests 
import numdifftools as nd # Pour le calcul numérique de la Hessienne

# --- 1. Téléchargement et chargement des données Pantheon+ ---

# Liens RAW corrects vers les fichiers sur GitHub
# Pantheon+SH0ES.dat est à la racine du dossier Pantheon+_Data.
PANTHEON_DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/Pantheon%2BSH0ES.dat"
# UTILISATION DE LA MATRICE DE COVARIANCE COMPLÈTE (STATISTIQUE + SYSTÉMATIQUE)
PANTHEON_FULL_COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"


print("Téléchargement et chargement des données Pantheon+ (avec matrice de covariance complète)...")

try:
    # --- Téléchargement et traitement de Pantheon+SH0ES.dat ---
    response_data = requests.get(PANTHEON_DATA_URL)
    response_data.raise_for_status() 
    data_content = response_data.text.splitlines()

    header_line = None
    data_lines = []
    for line in data_content:
        if line.startswith('#name'):
            header_line = line[1:].strip() 
        elif not line.startswith('#') and line.strip(): 
            data_lines.append(line.strip())

    if header_line is None:
        raise ValueError("Ligne d'en-tête '#name' non trouvée dans Pantheon+SH0ES.dat.")

    file_content_for_pandas = io.StringIO(header_line + '\n' + '\n'.join(data_lines))
    df_sn = pd.read_csv(file_content_for_pandas, sep='\s+')
    
    z_cmb = df_sn['zCMB'].values
    mb_obs = df_sn['m_b'].values 
    
    N_sn = len(mb_obs) 
    print(f"Nombre de supernovae détectées (Pantheon+SH0ES.dat) : {N_sn}")

    # --- Téléchargement et traitement de Pantheon+SH0ES_STAT+SYS.cov ---
    print(f"Tentative de téléchargement et chargement de la matrice de covariance complète ({PANTHEON_FULL_COV_URL})...")
    
    response_cov_full = requests.get(PANTHEON_FULL_COV_URL)
    response_cov_full.raise_for_status() 
    
    cov_full_content_lines = response_cov_full.text.splitlines()
    
    num_sn_in_cov = int(cov_full_content_lines[0].strip())
    
    if num_sn_in_cov != N_sn:
        raise ValueError(f"Incohérence du nombre de SN entre .dat ({N_sn}) et .cov ({num_sn_in_cov}).")

    C_full = np.zeros((N_sn, N_sn))
    
    for line in cov_full_content_lines[1:]: 
        if line.strip() and not line.startswith('#'):
            try:
                parts = line.split()
                row_idx = int(parts[0]) - 1 
                col_idx = int(parts[1]) - 1
                value = float(parts[2])
                
                C_full[row_idx, col_idx] = value
                if row_idx != col_idx:
                    C_full[col_idx, row_idx] = value
            except ValueError:
                print(f"Attention: Ligne non parsable dans le fichier de covariance complète: {line.strip()}")
                continue 

    C_inv = np.linalg.inv(C_full)
    
    print(f"Chargement réussi de la matrice de covariance complète (STAT + SYS).")
    print(f"Taille de la matrice de covariance: {C_full.shape}")

except requests.exceptions.RequestException as e:
    print(f"Erreur de réseau lors du téléchargement des fichiers : {e}")
    print("Vérifiez votre connexion internet ou les liens des fichiers. Les URLs utilisées sont:")
    print(f"  DATA: {PANTHEON_DATA_URL}")
    print(f"  COV: {PANTHEON_FULL_COV_URL}")
    print("Utilisation de données synthétiques pour continuer l'exemple.")
    np.random.seed(42)
    z_cmb = np.linspace(0.01, 1.0, 50)
    mb_obs = 5 * np.log10(z_cmb * 300000 / 70) + 25 + 5 * np.log10(299792.458) + np.random.normal(0, 0.1, z_cmb.shape)
    C_full = np.diag(0.1 * np.ones_like(z_cmb)**2) 
    C_inv = np.linalg.inv(C_full)
    print(f"Utilisation de {len(z_cmb)} supernovae synthétiques.")
except Exception as e:
    print(f"Une erreur inattendue est survenue lors du chargement ou de la manipulation des données : {e}")
    print("Utilisation de données synthétiques pour continuer l'exemple.")
    np.random.seed(42)
    z_cmb = np.linspace(0.01, 1.0, 50)
    mb_obs = 5 * np.log10(z_cmb * 300000 / 70) + 25 + 5 * np.log10(299792.458) + np.random.normal(0, 0.1, z_cmb.shape)
    C_full = np.diag(0.1 * np.ones_like(z_cmb)**2) 
    C_inv = np.linalg.inv(C_full)
    print(f"Utilisation de {len(z_cmb)} supernovae synthétiques.")

# --- 2. Définition du modèle cosmologique (inchangé) ---

c_kms = 299792.458
phi0 = 1.5
phi_inf = 1.618
Gamma = 0.23

def phi_z(z, phi_inf=phi_inf, phi0=phi0, Gamma=Gamma):
    return phi_inf - (phi_inf - phi0) * np.exp(-Gamma * z)

def H_z_model(z, H0, Om_m, phi_func):
    Omega_L = 1.0 - Om_m
    current_phi = phi_func(z)
    return H0 * np.sqrt(Om_m * (1 + z)**(3 * current_phi) + Omega_L * (1 + z)**(3 * (2 - current_phi)))

def luminosity_distance(z_array, H0, Om_m, phi_func):
    integral_values = np.zeros_like(z_array, dtype=float)
    z_integration_grid = np.linspace(0.0, z_array.max() * 1.05, 500) 
    Hz_integrated = H_z_model(z_integration_grid, H0, Om_m, phi_func)
    inv_Hz_interp = interp1d(z_integration_grid, 1.0 / Hz_integrated, kind='cubic', fill_value="extrapolate")
    for i, z_val in enumerate(z_array):
        if z_val == 0: integral_values[i] = 0.0
        else:
            z_to_integrate = z_integration_grid[z_integration_grid <= z_val]
            if len(z_to_integrate) < 2: integral_values[i] = z_val / H0
            else: integral_values[i] = np.trapezoid(inv_Hz_interp(z_to_integrate), z_to_integrate)
    dL_Mpc = (1 + z_array) * (c_kms / H0) * integral_values
    return dL_Mpc

def distance_modulus(dL_Mpc, M):
    dL_Mpc[dL_Mpc <= 0] = 1e-10 
    return 5 * np.log10(dL_Mpc) + 25 + M

# --- 3. Fonction de vraisemblance (chi-carré) ---

def chi2(params, z_data, mb_obs, C_inv_data, phi_func_fixed):
    H0, Om_m, M = params
    # Ajout d'une petite pénalité si les contraintes physiques ne sont pas respectées
    if H0 <= 0 or not (0 < Om_m < 1): return np.inf
    dL_model = luminosity_distance(z_data, H0, Om_m, phi_func_fixed)
    mu_model = distance_modulus(dL_model, M)
    delta_mu = mb_obs - mu_model
    chi2_val = delta_mu.T @ C_inv_data @ delta_mu
    return chi2_val

# --- 4. Exécution de l'ajustement ---

print("\nDébut de l'ajustement du modèle aux données SNIa...")

initial_params = [70.0, 0.3, -19.0] 
bounds = [(60.0, 80.0), (0.1, 0.5), (-20.0, -18.0)]

# Nous allons utiliser une fonction wrapper pour chi2 afin de la passer à Hessian
# car Hessian n'accepte pas directement les arguments supplémentaires de chi2
def chi2_wrapper(params):
    return chi2(params, z_cmb, mb_obs, C_inv, phi_z)

result = minimize(chi2_wrapper, initial_params, bounds=bounds, method='L-BFGS-B')

# Affichage des résultats
if result.success:
    best_H0, best_Om_m, best_M = result.x
    min_chi2 = result.fun
    dof = len(z_cmb) - len(initial_params) 
    chi2_per_dof = min_chi2 / dof

    print("\n--- Résultats de l'Ajustement ---")
    print(f"Convergence réussie: {result.success}")
    print(f"Meilleurs paramètres: ")
    print(f"  H0 = {best_H0:.2f} km/s/Mpc")
    print(f"  Omega_m = {best_Om_m:.3f}")
    print(f"  M (Magnitude Absolue SNIa) = {best_M:.2f} mag")
    print(f"Chi-carré minimum = {min_chi2:.2f}")
    print(f"Degrés de liberté = {dof}")
    print(f"Chi-carré par degré de liberté (chi2/dof) = {chi2_per_dof:.2f}")

    # --- Calcul des incertitudes sur les paramètres ---
    print("\nCalcul des incertitudes sur les paramètres...")
    try:
        # Calcul numérique de la Hessienne au point optimal
        # nd.Hessian(func)(x) retourne la Hessienne de func évaluée en x
        hessian_at_min = nd.Hessian(chi2_wrapper)(result.x)
        
        # Vérifier si la Hessienne est bien définie (non singulière)
        if np.linalg.det(hessian_at_min) == 0:
            print("Attention: La Hessienne est singulière, impossible de calculer l'inverse pour la matrice de covariance des paramètres.")
            cov_params = np.full((len(initial_params), len(initial_params)), np.nan)
        else:
            # L'inverse de la Hessienne est la matrice de covariance des paramètres
            cov_params = np.linalg.inv(hessian_at_min)
        
        # Les incertitudes sont les racines carrées des éléments diagonaux
        errors = np.sqrt(np.diag(cov_params))

        err_H0, err_Om_m, err_M = errors

        print(f"Incertitudes (1-sigma):")
        print(f"  σ(H0) = {err_H0:.2f} km/s/Mpc")
        print(f"  σ(Omega_m) = {err_Om_m:.3f}")
        print(f"  σ(M) = {err_M:.2f} mag")
        print("\nMatrice de covariance des paramètres:")
        print(cov_params)
        print("\nMatrice de corrélation des paramètres:")
        # Calcul de la matrice de corrélation pour voir les dépendances
        std_devs = np.sqrt(np.diag(cov_params))
        correlation_matrix = cov_params / np.outer(std_devs, std_devs)
        print(correlation_matrix)

    except Exception as e:
        print(f"Erreur lors du calcul des incertitudes : {e}")
        print("Assurez-vous que la bibliothèque 'numdifftools' est installée (pip install numdifftools).")


    # --- 5. Visualisation des résultats ---
    print("\nVisualisation du Diagramme de Hubble...")

    z_plot = np.linspace(0.01, z_cmb.max() * 1.1, 100)
    dL_model_plot = luminosity_distance(z_plot, best_H0, best_Om_m, phi_z)
    mu_model_plot = distance_modulus(dL_model_plot, best_M)

    plt.figure(figsize=(10, 6))
    diag_errors = np.sqrt(np.diag(C_full)) 
    plt.errorbar(z_cmb, mb_obs, yerr=diag_errors, fmt='o', markersize=3, capsize=2, label='Données SNIa Pantheon+ (incertitudes diagonales)', alpha=0.7)
    plt.plot(z_plot, mu_model_plot, color='red', linestyle='-', linewidth=2, label='Modèle Fractal Dynamique Ajusté')
    plt.xscale('log')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Module de distance (m - M)')
    plt.title('Diagramme de Hubble - Modèle Fractal Dynamique vs Données SNIa')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()

else:
    print("\n--- L'ajustement n'a pas convergé ---")
    print(result)

print("\nAnalyse terminée.")
