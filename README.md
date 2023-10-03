# **TcgaTargetGtex**



> This file is directly inherited from: `(2022_06_24)_UTIL_TcgaTargetGtex_CGP_venn_to_connection_mask.ipynb`






---
The purpose of this notebook is to demonstrate how to construct the connectivity mask which map 18k features input of TcgaTargetGtex to CGP of first hidden layer of AE.

---
## Input
This snippet required two things

1.   GMT file from msigdb. This file could be either `c2.cgp.v7.5.1.symbols.gmt` (genes will annotated in string alphabet) or `c2.cgp.v7.5.1.entrez.gmt` (genes will be annotated as alphanumerical)

2.   Gene list according to your input data. In this example, we will use a gene list from the TcgaTargetGtex dataset. This dataset contains 60k features from RNA-sequencing data. `TcgaTargetGtex_RSEM_Hugo_norm_count_gene_symbol.csv`







---
## Output
The final output of this snippet will generate an adjacency matrix containing membership information of 18303 genes in 2659 genesets.

output:\
`connectivity_matrix_TcgaTargetGtex_filterd_2022_06_24.pickle`


