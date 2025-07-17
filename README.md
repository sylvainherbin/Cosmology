# Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model

---

## Overview v1.1.0

This repository presents the source code and associated files for the research paper titled "Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model." This work introduces a complete fractal cosmological framework where the golden ratio $\phi$ dynamically evolves from primordial ($\phi_0=1.5$) to modern ($\phi_\infty=1.618$) epochs. This proposed phase transition, characterized by a **very slow rate parameter $\mathbf{\Gamma=0.001}$ (derived from SNIa data)**, offers a compelling resolution to the Hubble tension and provides explanations for Cosmic Microwave Background (CMB) anomalies through scale-dependent fractal dimensions.

**Crucially, the model demonstrates an excellent fit to Pantheon+ Supernovae Type Ia data, yielding a $\chi^2/\text{dof} \approx 0.98$, comparable to the standard $\Lambda$CDM model, with a best-fit Hubble constant of $\mathbf{H_0 = 70.00 \text{ km/s/Mpc}}$.**

The model makes several key predictions:
1.  **BAO Deviations:** $\Delta r_d/r_d \approx 0.15(1-e^{-z/2})$.
2.  **CMB Power Deficit:** $\mathcal{S}=0.93\pm0.02$ at $\ell<30$, yielding a significantly better $\chi^2/\text{dof}=1.72$ compared to $5.40$ for a static fractal model with constant $\phi=1.5$ (using Planck 2018 TT+lowE data).
3.  **Redshift-dependent Growth:** $f(z)=\Omega_m(z)^{\phi(z)/2}$.

---

## Repository Contents

This repository is structured as follows:

* **`main.tex`**: The primary LaTeX source file for the manuscript.
* **`Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model.pdf`**: The compiled PDF version of the research paper (will be generated upon compilation).
* **`.zenodo.json`**: A configuration file for Zenodo, specifying metadata for automated archiving and DOI assignment.
* **`CITATION.cff`**: A Citation File Format file, providing citation metadata for this repository.
* **`LICENSE.md`**: A CC BY 4.0 license.
* **`scripts/`**: Contains Python scripts used for analysis and data fitting.
    * **`scripts/cosmo_model_snia.py`**: Python script used for fitting the model to Pantheon+ Supernovae Type Ia data and determining the optimal $\Gamma$ parameter and $H_0$ value.

---

## Key Findings and Visualizations

The research highlights several critical aspects of the dynamic fractal cosmology model, supported by illustrative figures:

* **FIG. 1. Evolution of the fractal dimension showing transition**
    between primordial ($\phi_0 = 1.5$) and modern ($\phi_\infty = 1.618$) values. **Updated to reflect $\mathbf{\Gamma=0.001}$ derived from SNIa data.**
* **FIG. 2. Convergence of Fibonacci ratios toward $\phi$**
    The primordial value $\phi_0 = 1.5$ ($F_4/F_3$) marks the onset of fractal dimensionality.
* **FIG. 3. CMB spectrum showing fractal corrections at $\ell<30$ (blue band)**
    compared to $\Lambda$CDM (dashed line). Data points from Planck 2018.
* **FIG. 4. Hubble constant measurements with $1\sigma$ errors**
    **Updated to show the model's best-fit $\mathbf{H_0 = 70.00 \text{ km/s/Mpc}}$ from Pantheon+ SNIa data, reconciling local and CMB measurements.**

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
* **`enumitem`**

A standard LaTeX distribution (e.g., TeX Live, MiKTeX) that includes these packages is sufficient for compilation.

---

## Citation

If you utilize this work, please cite the corresponding Zenodo deposit:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15973541.svg)](https://doi.org/10.5281/zenodo.15973541)

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. You are free to share and adapt the material for any purpose, even commercially, provided you give appropriate credit, provide a link to the license, and indicate if changes were made.

---

## Contact

For any inquiries or further information regarding this research, please contact:

Sylvain Herbin
* **ORCID:** [https://orcid.org/0009-0001-3390-5012](https://orcid.org/0009-0001-3390-5012)
* **Email:** herbinsylvain@protonmail.com
* **Website:** https://sylvainherbin.github.io/cosmo/

---
