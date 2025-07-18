# Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model

---

## Overview v1.2.0

This repository contains the LaTeX source and associated files for the research paper titled "Dynamic Fractal Cosmology: A Fibonacci Phase Transition Model." This work introduces a complete fractal cosmological framework where the **golden ratio $\phi$ evolves dynamically from primordial ($\phi_0=1.5$) to modern ($\phi_\infty=1.618$) epochs**. This phase transition, characterized by a **rate parameter $\mathbf{\Gamma=0.23\pm0.01}$**, resolves the Hubble tension and explains Cosmic Microwave Background (CMB) anomalies through scale-dependent fractal dimensions.

**Crucially, leveraging Pantheon+ Type Ia supernova data, our model yields a best-fit Hubble constant of $\mathbf{H_0=68.74\pm0.16 \text{ km/s/Mpc}}$, along with $\mathbf{\Omega_m=0.297\pm0.009}$ and an absolute magnitude $\mathbf{M=-19.34\pm0.01 \text{ mag}}$, demonstrating an excellent fit with $\mathbf{\chi^2/\text{dof}=1.00}$.**

The model makes several key predictions:
1.  **BAO Deviations:** $\Delta r_d/r_d \approx 0.15(1-e^{-z/2})$.
2.  **CMB Power Deficit:** $\mathcal{S}=0.93\pm0.02$ at $\ell<30$, yielding a significantly better $\chi^2/\text{dof}=1.72$ compared to $5.40$ for a static fractal model with constant $\phi=1.5$ (using Planck 2018 TT+lowE data).
3.  **Redshift-dependent Growth:** $f(z)=\Omega_m(z)^{\phi(z)/2}$.

---

## Repository Contents

This repository is structured as follows:

* **`main.tex`**: The primary LaTeX source file for the manuscript.
* **`main.pdf`**: The compiled PDF version of the research paper.
* **`.zenodo.json`**: A configuration file for Zenodo, specifying metadata for automated archiving and DOI assignment.
* **`CITATION.cff`**: A Citation File Format file, providing citation metadata for this repository.
* **`LICENSE.md`**: A CC BY 4.0 license.
* **`scripts/`**: (If applicable) Directory intended for any analysis or data fitting scripts.
  * **`scripts/snia_dynamic_fractal_analysis.py`**: (If applicable) Python script used for fitting the model to Pantheon+ Supernovae Type Ia data.

---

## Key Findings and Visualizations

The research highlights several critical aspects of the dynamic fractal cosmology model, supported by illustrative figures within the paper:

* **Figure 1: Evolution of the fractal dimension**
    Shows the transition between primordial ($\phi_0 = 1.5$) and modern ($\phi_\infty = 1.618$) values, reflecting the $\Gamma$ parameter derived from SNIa data.
* **Figure 2: Convergence of Fibonacci ratios toward $\phi$**
    Illustrates how the primordial value $\phi_0 = 1.5$ ($F_4/F_3$) marks the onset of fractal dimensionality.
* **Figure 3: CMB spectrum showing fractal corrections at $\ell<30$ (blue band)**
    Compares the model's predictions to $\Lambda$CDM (dashed line) and Planck 2018 data points.
* **Figure 4: Hubble constant measurements with $1\sigma$ errors**
    Presents the model's best-fit $H_0$ from Pantheon+ SNIa data, demonstrating how it reconciles local and CMB measurements.

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
* `enumitem`

A standard LaTeX distribution (e.g., TeX Live, MiKTeX) that includes these packages is sufficient for compilation.

---

## Citation

If you utilize this work, please cite the corresponding Zenodo deposit using the Concept DOI, which always points to the latest version:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15973540.svg)](https://doi.org/10.5281/zenodo.15973540)

You can also find the full citation details in the `CITATION.cff` file.

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
