# Overview
This package contains an algorithm and pipeline for multiplicity of infection (MOI) and haplotype detection of P. falciporum malaria from 24-SNP barcode-based multiplex-PCR NGS assays. The algorithm is based on the StrainPycon (https://www.ymsir.com/strainpycon/) package for malarial strain disambiguation.

# License
The software is released under the accompanying MIT license.

# Citation
If you use STIM, please cite the following paper: 

* Rebecca M. Mitchell, Zhiyong Zhou, Mili Sheth, Sheila Sergent, Mike Frace, Vishal Nayak, Bin Hu, John Giming, Feiko ter Kuile, Kim Lindblade, Laurence Slutsker, Hamel J. Mary, Meghna Desai, Kephas Otieno, Simon Kariuki, Ymir Vigfusson and Ya Ping Shi. 2020. Development of a new barcode-based, multiplex-PCR, next-generation-sequencing assay and data processing and analytical pipeline for multiplicity of infection detection of P. falciparum.

# Usage
The first half of the pipeline, `screening.Rmd` is R pseudocode for the bioinformatics tools that are run on the raw 16S sequences. It generates the 24-SNP that are included in the `data/` folder.

The second half of the pipeline is a series of self-contained Python scripts in the directory.

* To run, first create a virtual Python3 environment (optional). 
* Then run: `pip install -r requirements.txt`
* Next, each file is run by itself. 
* For example, `python3 stim-longitudinal.py` will generate a box-plot for the MOI evolution in western Kenya in the `figures/` folder.

# Contact
Please contact corresponding author Ymir Vigfusson (ymir.vigfusson@emory.edu) with any questions about the software.





