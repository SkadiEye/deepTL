Deep Treatment Learning (deepTL) [![DOI](https://zenodo.org/badge/156120192.svg)](https://zenodo.org/badge/latestdoi/156120192)
================

Deep Treatment Learning (deepTL) is an R packages written in S4 class,
designed for,

  - ***PermFIT***: Permutation-based Feature Importance Test, a
    permutation-based feature importance test scheme for black-box
    models (DNNs, support vector machines, random forests, etc)
    \[Manuscript submitted\]
    \[[example](https://github.com/SkadiEye/deepTL/blob/master/permfit/permfit.md)\]
  - ***deepTL***: Deep Treatment Learning, an efficient semiparametric
    framework coupled with ensemble DNNs for adjusting complex
    confounding \[Manuscript submitted\]
    \[[example](https://github.com/SkadiEye/deepTL/blob/master/deeptl.md)\]
  - ***EndLot***: ENsemble Decision Learning Optimal Treatment, a
    DNN-based method for optimal individualized treatment learning
    (Paper: Mi et al. (2019))
    \[[example](https://github.com/SkadiEye/deepTL/blob/master/endlot.md)\]

You may also use it for,

  - ***DNN***: Easy implementation for feed-forward fully-connected deep
    neural networks
  - ***Bagging***: Bootstrap aggregating for DNN models, with an
    automatic scheme to select the optimal subset of DNNs (details in
    paper: Mi et al. (2019))
  - \[[example](https://github.com/SkadiEye/deepTL/blob/master/dnnet.md)\]

# Installation

  - System requirement: Rtools (Windows); None (MacOS/Linux)

  - In R:

<!-- end list -->

``` r
devtools::install_github("SkadiEye/deepTL")
```

# References

<div id="refs" class="references hanging-indent">

<div id="ref-mi2019bagging">

Mi, X., Zou, F., and Zhu, R. (2019), “Bagging and deep learning in
optimal individualized treatment rules,” *Biometrics*, Wiley Online
Library, 75, 674–684.

</div>

</div>
