Import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from google.colab import drive # Pour le montage de Google Drive

# --- 1. Téléchargement des données Pantheon+ directement dans Colab ---
print("--- Téléchargement des données Pantheon+ ---")

# Créer les répertoires nécessaires
!mkdir -p Pantheon_Data/4_DISTANCES_AND_COVAR

# Liens directs vers les fichiers bruts sur GitHub
PANTHEON_DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2BSH0ES_DATA/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
PANTHEON_COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2BSH0ES_DATA/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"

# Chemins locaux où les fichiers seront sauvegardés dans Colab
PANTHEON_DATA_FILE = "Pantheon_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
PANTHEON_COV_FILE = "Pantheon_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"

# Télécharger les fichiers
print(f"Téléchargement de {PANTHEON_DATA_URL} vers {PANTHEON_DATA_FILE}...")
!wget -q -O {PANTHEON_DATA_FILE} {PANTHEON_DATA_URL}
print(f"Téléchargement de {PANTHEON_COV_URL} vers {PANTHEON_COV_FILE}...")
!wget -q -O {PANTHEON_COV_FILE} {PANTHEON_COV_URL}

print("--- Vérification des fichiers téléchargés ---")
if os.path.getsize(PANTHEON_DATA_FILE) == 0:
    print(f"ERREUR CRITIQUE: Le fichier {PANTHEON_DATA_FILE} est vide après le téléchargement. L'URL est peut-être incorrecte ou le téléchargement a échoué.")
    exit(1)
else:
    print(f"Le fichier {PANTHEON_DATA_FILE} a une taille de {os.path.getsize(PANTHEON_DATA_FILE)} octets (OK si > 0).")

if os.path.getsize(PANTHEON_COV_FILE) == 0:
    print(f"ERREUR CRITIQUE: Le fichier {PANTHEON_COV_FILE} est vide après le téléchargement. L'URL est peut-être incorrecte ou le téléchargement a échoué.")
    exit(1)
else:
    print(f"Le fichier {PANTHEON_COV_FILE} a une taille de {os.path.getsize(PANTHEON_COV_FILE)} octets (OK si > 0).")
print("--- Téléchargement et vérification terminés ! ---")

# --- Fonctions de votre modèle cosmologique (inchangées) ---

def phi(z, phi0, phi_inf, Gamma):
    return phi_inf - (phi_inf - phi0) * np.exp(-Gamma * z)

def E(z, Omegam, phi0, phi_inf, Gamma):
    Omegalambda = 1.0 - Omegam
    current_phi = phi(z, phi0, phi_inf, Gamma)
    term_m = Omegam * (1 + z)**(3 * current_phi)
    term_l = Omegalambda * (1 + z)**(3 * (2 - current_phi))
    if term_m + term_l <= 0: return np.inf 
    return np.sqrt(term_m + term_l)

def integrand(z_prime, Omegam, phi0, phi_inf, Gamma):
    val_E = E(z_prime, Omegam, phi0, phi_inf, Gamma)
    if val_E == 0 or np.isinf(val_E): return np.inf
    return 1.0 / val_E

def lum_distance(z, H0, Omegam, phi0, phi_inf, Gamma):
    c_light = 299792.458 # Vitesse de la lumière en km/s
    try:
        integral, abserr = quad(integrand, 0, z, args=(Omegam, phi0, phi_inf, Gamma), limit=100)
    except Exception as e: return np.inf 
    if integral <= 0 or np.isinf(integral) or np.isnan(integral): return np.inf
    dL = (1 + z) * (c_light / H0) * integral
    return dL

def mu_theory(z, H0, Omegam, phi0, phi_inf, Gamma):
    dL = lum_distance(z, H0, Omegam, phi0, phi_inf, Gamma)
    if dL <= 0 or np.isinf(dL) or np.isnan(dL): return np.inf
    return 5 * np.log10(dL) + 25

def chi2_snia_fixed_phi(params_to_optimize, z_obs, mu_obs, cov_matrix_full, phi0_fixed, phi_inf_fixed):
    H0, Omegam, Gamma = params_to_optimize
    if not (H0 > 0 and Gamma >= 0): return np.inf 
    mu_th_values = np.array([mu_theory(z, H0, Omegam, phi0_fixed, phi_inf_fixed, Gamma) for z in z_obs])
    if np.any(np.isinf(mu_th_values)) or np.any(np.isnan(mu_th_values)): return np.inf
    delta_mu = mu_obs - mu_th_values
    try:
        chi2 = np.dot(delta_mu, np.dot(np.linalg.inv(cov_matrix_full), delta_mu))
    except np.linalg.LinAlgError: return np.inf
    return chi2

# --- Fonctions pour le modèle Lambda-CDM (inchangées) ---

def E_LCDM(z, Omegam_LCDM):
    Omegalambda_LCDM = 1.0 - Omegam_LCDM
    return np.sqrt(Omegam_LCDM * (1 + z)**3 + Omegalambda_LCDM)

def integrand_LCDM(z_prime, Omegam_LCDM):
    val_E = E_LCDM(z_prime, Omegam_LCDM)
    if val_E == 0 or np.isinf(val_E): return np.inf
    return 1.0 / val_E

def lum_distance_LCDM(z, H0_LCDM, Omegam_LCDM):
    c_light = 299792.458 # Vitesse de la lumière en km/s
    try:
        integral, abserr = quad(integrand_LCDM, 0, z, args=(Omegam_LCDM,), limit=100)
    except Exception as e: return np.inf 
    if integral <= 0 or np.isinf(integral) or np.isnan(integral): return np.inf
    dL = (1 + z) * (c_light / H0_LCDM) * integral
    return dL

def mu_theory_LCDM(z, H0_LCDM, Omegam_LCDM):
    dL = lum_distance_LCDM(z, H0_LCDM, Omegam_LCDM)
    if dL <= 0 or np.isinf(dL) or np.isnan(dL): return np.inf
    return 5 * np.log10(dL) + 25

# --- Fonctions de chargement des fichiers (modifiée pour load_data_file) ---
def load_data_file(filepath):
    if not os.path.exists(filepath):
        print(f"Erreur: Le fichier de données '{filepath}' est introuvable. Vérifiez le téléchargement.")
        exit(1)
    
    # MODIFICATION CLÉ ICI : Spécifier delimiter=' ' pour gérer les espaces comme séparateurs
    # et skip_header=1 pour sauter la ligne d'en-tête
    # usecols=(5, 7) pour ne lire que les colonnes zHD (index 5) et m_b_corr (index 7)
    data = np.genfromtxt(filepath, skip_header=1, usecols=(5, 7), encoding=None, delimiter=' ')
    
    # On assigne des noms explicites aux colonnes que nous avons lues
    zHD_data = data[:, 0]
    mb_corr_data = data[:, 1]

    # Pour que le reste du code fonctionne avec des noms, nous pouvons créer un "objet" avec ces attributs
    class DataContainer:
        def __init__(self, zHD, m_b_corr):
            self.zHD = zHD
            self.m_b_corr = m_b_corr

    loaded_data = DataContainer(zHD_data, mb_corr_data)

    print(f"\n--- Diagnostic du fichier de données '{filepath}' ---")
    print(f"Nombre de lignes de données lues : {len(loaded_data.zHD)}")
    
    print(f"Premiers 5 redshifts (zHD) : {loaded_data.zHD[:5]}")
    print(f"Premiers 5 modules de distance (m_b_corr) : {loaded_data.m_b_corr[:5]}")
    print(f"Min/Max zHD : {np.min(loaded_data.zHD):.4f} / {np.max(loaded_data.zHD):.4f}")
    print(f"Min/Max m_b_corr : {np.min(loaded_data.m_b_corr):.4f} / {np.max(loaded_data.m_b_corr):.4f}")
    print("-------------------------------------------------------------------")
    return loaded_data

def load_cov_file(filepath, expected_dim):
    if not os.path.exists(filepath):
        print(f"Erreur: Le fichier de covariance '{filepath}' est introuvable. Vérifiez le téléchargement.")
        exit(1)
    
    flat_cov = np.loadtxt(filepath, dtype='float')
    
    expected_num_elements = expected_dim * expected_dim
    
    if flat_cov.size != expected_num_elements:
        print(f"Erreur grave: Le nombre d'éléments dans le fichier de covariance ({flat_cov.size})")
        print(f"ne correspond pas à la dimension carrée attendue ({expected_num_elements}). Le fichier est probablement corrompu ou incomplet.")
        exit(1)
        
    cov_matrix = flat_cov.reshape((expected_dim, expected_dim))

    print(f"\n--- Diagnostic du fichier de covariance '{filepath}' ---")
    print(f"Dimension de la matrice de covariance : {cov_matrix.shape}")
    print(f"Premiers 5 éléments de la diagonale (variances) : {np.diag(cov_matrix)[:5]}")
    print(f"Min/Max de la diagonale : {np.min(np.diag(cov_matrix)):.4e} / {np.max(np.diag(cov_matrix)):.4e}")
    
    if np.any(np.diag(cov_matrix) < 0):
        print("!!!! ALERTE ROUGE : Des valeurs négatives trouvées sur la diagonale de la matrice de covariance. !!!!")
        print("!!!! Ceci indique un problème majeur avec le fichier ou son chargement. Le Chi^2 sera incorrect. !!!!")
    
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        print("Matrice de covariance inversible avec succès.")
        print(f"Premiers 5 éléments de la diagonale de l'inverse : {np.diag(inv_cov_matrix)[:5]}")
        print(f"Min/Max de la diagonale de l'inverse : {np.min(np.diag(inv_cov_matrix)):.4e} / {np.max(np.diag(inv_cov_matrix)):.4e}")
    except np.linalg.LinAlgError:
        print("ATTENTION : La matrice de covariance est singulière ou mal conditionnée, son inversion a échoué.")
    print("-------------------------------------------------------------------")
    return cov_matrix


# --- Point d'entrée principal du script ---
if __name__ == "__main__":
    # --- Optionnel : Monter Google Drive pour sauvegarder les résultats ---
    # Pour activer, décommenter la ligne ci-dessous et suivre les instructions d'authentification
    # drive.mount('/content/drive')
    # Définissez un chemin de sauvegarde si vous montez Drive
    # SAVE_PATH = '/content/drive/MyDrive/CosmoResults/'
    # os.makedirs(SAVE_PATH, exist_ok=True)
    # Sinon, les fichiers seront sauvegardés dans l'environnement Colab temporaire
    SAVE_PATH = './'


    print("Début de la préparation des données Pantheon+...")

    # 1. Chargement des données
    print(f"Chargement des données depuis {PANTHEON_DATA_FILE}...")
    data_container = load_data_file(PANTHEON_DATA_FILE) # ATTENTION : Renommé en data_container

    z_obs = data_container.zHD
    mu_obs = data_container.m_b_corr 
    
    num_sn = len(z_obs)

    # 2. Chargement de la matrice de covariance totale (et redimensionnement)
    print(f"Chargement et redimensionnement de la matrice de covariance totale depuis {PANTHEON_COV_FILE}...")
    cov_matrix_full = load_cov_file(PANTHEON_COV_FILE, num_sn) 
    
    print(f"Données chargées : {num_sn} points de Supernovae Ia.")
    print(f"Matrice de covariance totale de dimension : {cov_matrix_full.shape}")
    
    print("\nDonnées Pantheon+ prêtes pour l'analyse !\n")

    # --- Paramètres Phi fixes basés sur votre théorie ---
    PHI0_THEORETICAL = 1.5
    PHI_INF_THEORETICAL = 1.618

    print(f"--- Début de la minimisation Chi^2 avec phi0={PHI0_THEORETICAL} et phi_inf={PHI_INF_THEORETICAL} fixés ---")

    # Paramètres initiaux pour l'optimisation : (H0, Omegam, Gamma)
    initial_params_opt = [70.0, 0.3, 0.2] 
    print(f"Paramètres initiaux pour H0, Omegam, Gamma : {initial_params_opt}")

    # Bornes pour les paramètres H0, Omegam, Gamma
    bounds_opt = [(50.0, 100.0),    
                  (0.001, 1.0),     
                  (0.001, 2.0)]     

    print("Lancement de la minimisation... (cela peut prendre quelques instants)")
    result = minimize(chi2_snia_fixed_phi, initial_params_opt, 
                      args=(z_obs, mu_obs, cov_matrix_full, PHI0_THEORETICAL, PHI_INF_THEORETICAL), 
                      bounds=bounds_opt, 
                      method='L-BFGS-B')

    optimal_H0, optimal_Omegam, optimal_Gamma = result.x

    print("\n--- Résultats de la minimisation de Votre Modèle ---")
    print(f"Statut de la minimisation : {result.message}")
    print(f"Paramètres optimaux : H0={optimal_H0:.2f}, Omegam={optimal_Omegam:.3f}, phi0={PHI0_THEORETICAL:.1f} (fixé), phi_inf={PHI_INF_THEORETICAL:.3f} (fixé), Gamma={optimal_Gamma:.3f}")
    
    dof = len(z_obs) - len(initial_params_opt) 
    print(f"Valeur de Chi^2 minimale : {result.fun:.2f}")
    print(f"Degrés de liberté : {dof}")
    print(f"Chi^2 / dof : {result.fun / dof:.2f}")

    print("\n--- Fin de la minimisation de Votre Modèle ---")

    # --- Section de vérification du Chi^2 LCDM standard (Planck 2018) ---
    print("\n--- Calcul du Chi^2 pour le modèle Lambda-CDM standard (Planck 2018) ---")
    H0_LCDM_Planck = 67.4 
    Omegam_LCDM_Planck = 0.315

    mu_th_LCDM_values_check = np.array([mu_theory_LCDM(z, H0_LCDM_Planck, Omegam_LCDM_Planck) for z in z_obs])

    if np.any(np.isinf(mu_th_LCDM_values_check)) or np.any(np.isnan(mu_th_LCDM_values_check)):
        print("ATTENTION: Des valeurs inf ou NaN trouvées dans les modules de distance LCDM. Cela peut indiquer un problème avec les fonctions LCDM.")
        
    delta_mu_LCDM_check = mu_obs - mu_th_LCDM_values_check
    
    try: 
        chi2_LCDM_check = np.dot(delta_mu_LCDM_check, np.dot(np.linalg.inv(cov_matrix_full), delta_mu_LCDM_check))
        dof_LCDM_check = len(z_obs) - 2 
        print(f"Chi^2 Lambda-CDM (Planck) : {chi2_LCDM_check:.2f}")
        print(f"Degrés de liberté Lambda-CDM : {dof_LCDM_check}")
        print(f"Chi^2 / dof Lambda-CDM : {chi2_LCDM_check / dof_LCDM_check:.2f}")
        print("Ce Chi^2 / dof devrait être très proche de 1 pour un bon ajustement du modèle standard.")
    except np.linalg.LinAlgError:
        print("ÉCHEC DU CALCUL DU CHI^2 LCDM : La matrice de covariance est singulière ou mal conditionnée.")

    print("-------------------------------------------------------------------")


    # --- Section de visualisation des résultats ---
    print("\n--- Début de la visualisation des résultats ---")

    z_theory = np.linspace(np.min(z_obs), np.max(z_obs), 500)
    
    mu_theory_values = []
    for zval in z_theory:
        mu_val = mu_theory(zval, optimal_H0, optimal_Omegam, PHI0_THEORETICAL, PHI_INF_THEORETICAL, optimal_Gamma)
        if np.isinf(mu_val) or np.isnan(mu_val):
            mu_theory_values.append(np.nan)
        else:
            mu_theory_values.append(mu_val)
    mu_theory_values = np.array(mu_theory_values)

    H0_LCDM = 67.4 
    Omegam_LCDM = 0.315
    
    mu_theory_LCDM_values = []
    for zval in z_theory:
        mu_val_LCDM = mu_theory_LCDM(zval, H0_LCDM, Omegam_LCDM)
        if np.isinf(mu_val_LCDM) or np.isnan(mu_val_LCDM):
            mu_theory_LCDM_values.append(np.nan)
        else:
            mu_theory_LCDM_values.append(mu_val_LCDM)
    mu_theory_LCDM_values = np.array(mu_theory_LCDM_values)


    plt.figure(figsize=(12, 8))
    
    stat_err = np.sqrt(np.diag(cov_matrix_full).clip(min=0)) 
    plt.errorbar(z_obs, mu_obs, yerr=stat_err, fmt='.', color='blue', alpha=0.5, label='Données Pantheon+ Observées (avec erreurs stat.)')
    
    plt.plot(z_theory, mu_theory_values, color='red', linestyle='-', linewidth=2, label=f'Votre Modèle ($\\phi_0$={PHI0_THEORETICAL}, $\\phi_\\infty$={PHI_INF_THEORETICAL})')

    plt.plot(z_theory, mu_theory_LCDM_values, color='green', linestyle='--', linewidth=2, label=f'Modèle $\\Lambda$CDM ($H_0$={H0_LCDM}, $\\Omega_m$={Omegam_LCDM})')


    plt.title('Module de Distance vs Redshift : Votre Modèle vs $\\Lambda$CDM et Données SNIa', fontsize=16)
    plt.xlabel('Redshift (z)', fontsize=14)
    plt.ylabel('Module de Distance ($\mu$)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = "snia_model_comparison_plot_colab.png"
    plt.savefig(os.path.join(SAVE_PATH, plot_filename)) 
    print(f"Graphique sauvegardé sous : {os.path.join(SAVE_PATH, plot_filename)}")
    
    plt.show()

    print("\n--- Fin de la visualisation des résultats ---")
    print("\nSi le problème de matrice de covariance est résolu, la courbe Lambda-CDM devrait maintenant bien s'ajuster aux données.")
    print("La prochaine étape sera d'explorer comment tester votre modèle sur l'univers primordial.")

