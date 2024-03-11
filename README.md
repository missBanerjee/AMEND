# AMEND: Adaptive Margin and Expanded Neighborhood for Efficient Generalized Category Discovery

This is the official Implementation of the paper https://openaccess.thecvf.com/content/WACV2024/papers/Banerjee_AMEND_Adaptive_Margin_and_Expanded_Neighborhood_for_Efficient_Generalized_Category_WACV_2024_paper.pdf

Generalized Category Discovery aims to discover and cluster images from previously unseen classes, in addition to classifying images from seen classes correctly. In this work, we propose a simple, yet effective framework for this task, which not only performs on-par or better with the current approaches but is also significantly more efficient in terms of computational requirements. Our first contribution is to use expanded neighborhood information in contrastive learning to generate robust and generalizable features. To generate more discriminative feature representations, especially for fine-grained datasets and confusing classes, we propose a class-wise adaptive margin regularizer that aims at increasing the angular separation among the prototypes of all classes. Extensive experiments on three generic as well as four fine-grained benchmark datasets show the usefulness of the proposed Adaptive Margin and Expanded Neighborhood (AMEND) framework.
## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet](https://image-net.org/download.php)


### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

Then check the results starting with `Metrics with best model on test set:` in the logs.
This means the model is picked according to its performance on the test set, and then evaluated on the unlabelled instances of the train set.



## Acknowledgements

The codebase is largely built on this repo: https://github.com/CVMI-Lab/SimGCD


