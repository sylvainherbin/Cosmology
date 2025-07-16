# Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model

---

## Overview

This repository presents the source code and associated files for the research paper titled "Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model." This work introduces a complete fractal cosmological framework where the golden ratio $\phi$ dynamically evolves from primordial ($\phi_0=1.5$) to modern ($\phi_\infty=1.618$) epochs. This proposed phase transition, characterized by a specific rate parameter $\Gamma=0.23\pm0.01$, offers compelling resolutions to the Hubble tension ($H_0=73.04\pm0.38$ km/s/Mpc) and provides explanations for Cosmic Microwave Background (CMB) anomalies through scale-dependent fractal dimensions.

The model makes several key predictions:
1.  **BAO Deviations:** $\Delta r_d/r_d \approx 0.15(1-e^{-z/2})$.
2.  **CMB Power Deficit:** $\mathcal{S}=0.93\pm0.02$ at $\ell<30$, yielding a significantly better $\chi^2/\text{dof}=1.72$ compared to $5.40$ for a static fractal model with constant $\phi=1.5$ (using Planck 2018 TT+lowE data).
3.  **Redshift-dependent Growth:** $f(z)=\Omega_m(z)^{\phi(z)/2}$.

---

## Repository Contents

This repository is structured as follows:

* **`main.tex`**: The primary LaTeX source file for the manuscript.
* **`Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model.pdf`**: The compiled PDF version of the research paper.
* **`.zenodo.json`**: A configuration file for Zenodo, specifying metadata for automated archiving and DOI assignment.
* **`CITATION.cff`**: A Citation File Format file, providing citation metadata for this repository.

---

## Key Findings and Visualizations

The research highlights several critical aspects of the dynamic fractal cosmology model, supported by illustrative figures:

* **FIG. 1. Evolution of the fractal dimension showing transition**
    between primordial ($\phi_0 = 1.5$) and modern ($\phi_\infty = 1.618$) values.
* **FIG. 2. Convergence of Fibonacci ratios toward $\phi$**
    The primordial value $\phi_0 = 1.5$ ($F_4/F_3$) marks the onset of fractal dimensionality.
* **FIG. 3. CMB spectrum showing fractal corrections at $\ell<30$ (blue band)**
    compared to $\Lambda$CDM (dashed line). Data points from Planck 2018.
* **FIG. 4. Hubble constant measurements with $1\sigma$ errors**
    Planck (CMB), Freedman et al. (TRGB), and Riess et al. (SNIa). Dashed line shows model prediction with $\pm0.38$ km/s/Mpc uncertainty.
* **FIG. 5. Comparison of $\chi^2/\text{dof}$ for the dynamic fractal model (1.72)**
    and the static fractal model with $\phi=1.5$ (5.40), using Planck 2018 TT+lowE data.

---

## Compilation Requirements

To compile the `main.tex` document, the following LaTeX packages are utilized:

* `inputenc` (utf8)
* `fontenc` (T1)
* `amsmath`, `amssymb`
* `graphicx`
* `natbib`
* `tikz`
* `pgfplots` (with `compat=1.18` set)
* `tikzlibrary{shapes.geometric, arrows.meta, calc}`
* `booktabs`
* `xcolor`
* `hyperref`
* `orcidlink`

A standard LaTeX distribution (e.g., TeX Live, MiKTeX) that includes these packages is sufficient for compilation.

---

## Citation

If you utilize this work, please cite the corresponding research paper. Citation information is available in the `CITATION.cff` file in this repository.

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. You are free to share and adapt the material for any purpose, even commercially, provided you give appropriate credit, provide a link to the license, and indicate if changes were made.

---

## Contact

For any inquiries or further information regarding this research, please contact:

Sylvain Herbin
* **ORCID:** [https://orcid.org/0009-0001-3390-5012](https://orcid.org/0009-0001-3390-5012)
* **Email:** herbinsylvain@protonmail.com
* **website:** https://sylvainherbin.github.io/cosmo/

---
