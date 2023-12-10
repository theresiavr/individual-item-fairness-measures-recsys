# Individual Item Fairness Measures in Recommender Systems (RecSys) ⚖

This repository contains the code used for the experiments and analysis in "Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study" by Theresia Veronika Rampisela, Maria Maistro, Tuukka Ruotsalo, and Christina Lioma. This work has been accepted to ACM Transaction on Recommender Systems (ACM TORS). 

[[ACM]](https://doi.org/10.1145/3631943) [[arXiv]](https://arxiv.org/abs/2311.01013)

# Abstract
Fairness is an emerging and challenging topic in recommender systems. In recent years, various ways of evaluating and therefore improving fairness have emerged. In this study, we examine existing evaluation measures of fairness in recommender systems. Specifically, we focus solely on exposure-based fairness measures of individual items that aim to quantify the disparity in how individual items are recommended to users, separate from item relevance to users. We gather all such measures and we critically analyse their theoretical properties. We identify a series of limitations in each of them, which collectively may render the affected measures hard or impossible to interpret, to compute, or to use for comparing recommendations. We resolve these limitations by redefining or correcting the affected measures, or we argue why certain limitations cannot be resolved. We further perform a comprehensive empirical analysis of both the original and our corrected versions of these fairness measures, using real-world and synthetic datasets. Our analysis provides novel insights into the relationship between measures based on different fairness concepts, and different levels of measure sensitivity and strictness. We conclude with practical suggestions of which fairness measures should be used and when. Our code is publicly available. To our knowledge, this is the first critical comparison of individual item fairness measures in recommender systems.

# License and Terms of Usage
The code is usable under the MIT License. Please note that RecBole may have different terms of usage (see [their page](https://github.com/RUCAIBox/RecBole) for updated information). 

# Citation
If you use the code for the fairness measures in metrics.py, please cite our paper and the original papers proposing the measures.
```BibTeX
@article{10.1145/3631943,
author = {Rampisela, Theresia Veronika and Maistro, Maria and Ruotsalo, Tuukka and Lioma, Christina},
title = {Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3631943},
doi = {10.1145/3631943},
note = {Just Accepted},
journal = {ACM Trans. Recomm. Syst.},
month = {nov},
}
```

# Datasets
We use the following datasets, that can be downloaded from the Google Drive folder provided by [RecBole](https://recbole.io/dataset_list.html), under ProcessedDatasets:
- Amazon-lb, Amazon-dm, Amazon-is: these datasets are all in the Amazon2018 folder. The names of the folder/datasets are: Amazon_Luxury_Beauty, Amazon_Digital_music, and Amazon_Industrial_and_Scientific respectively
- Book-x is under the Book-Crossing folder
- Lastfm is under the LastFM folder
- Ml-1m is under the MovieLens folder; the file is ml-1m.zip

1. Download the zip files corresponding to the full datasets (not the examples), place them inside the empty dataset folder in the main folder.
2. Unzip the files.
3. Ensure that the name of the folder and the .inter files are the same as in the [dataset properties](https://github.com/theresiavr/individual-item-fairness-measures-recsys/tree/main/RecBole/recbole/properties/dataset).

