# Personalized Product Search on JDsearch

*BT4222 Machine Learning for Business Applications · Group 16 · AY 2026/27, Term 1*

If two shoppers type the same query, can their past activity help us show each of them a better ranking? We use JDsearch to test this by re-ranking the products already in each query’s candidate list. We compare two LightGBM LambdaRank models on the same candidates, labels, and user splits.

| Model | What it uses |
| --- | --- |
| **A · Query and popularity** | Seven features covering query–product match, product popularity, and metadata availability. |
| **B · Personalized** | The same seven features plus 23 history features, including prior interactions with the candidate, category and brand preferences, past-query similarity, and collaborative filtering. |

We keep clicks, cart additions, and purchases separate in the history features because the dataset records them over different lookback periods. We shuffle candidate lists and break score ties with a fixed seed because relevant items tend to appear near the top of the original lists.

## Get the data

The [shared `Data/` folder](https://drive.google.com/drive/folders/1TvsNb6MmUC30buB5bnybRowW8mONZhzJ) contains the two raw JDsearch files, the processed Parquet tables, the features, and the trained models. The [official JDsearch repository](https://github.com/rucliujn/JDsearch#obtain-the-data) also explains how to get the full archive from the dataset authors.

For a full run from raw data, copy the two text files into your own Google Drive using this layout.

```text
MyDrive/
└── bt4222_project/
    └── Data/
        ├── user_behavior_data.txt   # queries, candidates, labels, and shopper histories
        └── product_meta_data.txt    # product names, brands, categories, and shops
```

The two files take up about 5.8 GB together. The notebooks write their outputs under the same `Data/` directory. For a quicker check of the reported results, use the saved test files and models listed below.

## Run the notebooks in Colab

Open the notebooks in [Google Colab](https://colab.research.google.com/) from GitHub or the [shared notebook folder](https://drive.google.com/drive/folders/1_cM_hTzbBBVQELrixHRXXfSx0iphRb4Y). Run each notebook from top to bottom. The setup cells mount Google Drive at `/content/drive`, so the project files should appear under `/content/drive/MyDrive/bt4222_project/Data/`.

Install the following packages in your Colab runtime before running the notebooks. Repeat the setup if you start a fresh runtime. `statsmodels` is used for the regression in the evaluation notebook.

```python
%pip install -q numpy pandas pyarrow scipy scikit-learn matplotlib seaborn lightgbm statsmodels
```

### Full run from raw data

The rankers use CPU training. For a full run, use a runtime with enough memory and local disk for the raw files and intermediate tables. Feature generation runs in chunks, but training loads roughly 10.75 million candidate rows together.

| Step | Notebook | What it does and where it writes |
| --- | --- | --- |
| 1 | [EDA and data splitting](Notebooks/EDA_DataSplitting.ipynb) | Reads the raw text files, checks the interaction history, and creates a 70/15/15 split by shopper with seed 42. Writes Parquet tables to `Data/Processed/` and exploratory plots to `Data/Figures/`. |
| 2 | [Feature engineering](Notebooks/FeatureEngineering.ipynb) | Reads the processed tables and builds aligned candidate, label, and feature tables in `Data/Features/`. Run through Section 4 to generate all four training chunks and the test features. |
| 3 | [Baseline model](Notebooks/Baseline_Model.ipynb) | Trains Model A on the feature tables and selects the number of trees on validation. Writes the model, run configuration, validation scores, and shared evaluator to `Data/Models/Baseline/`. |
| 4 | [Personalized model](Notebooks/Advanced_Model.ipynb) | Uses the baseline run configuration, verifies the shared inputs, and trains Model B with the additional history features. Writes its model and run files to `Data/Models/Advanced/`. |
| 5 | [Model comparison](Notebooks/ModelComparison.ipynb) | Loads both saved models and the test tables. Compares ranking metrics, estimates paired confidence intervals, and checks where personalization helps. Writes tables and plots to `Data/Results/ModelComparison/test/`. |

The preprocessing and feature notebooks can reuse files already on Drive. To rebuild from scratch, start with the two raw files in a clean project folder at the path above and use a fresh runtime.

### Test evaluation with saved models

To run just the final comparison, copy these files into your own `MyDrive/bt4222_project/Data/` directory, keeping the subfolders shown here. Then run [Model comparison](Notebooks/ModelComparison.ipynb) with the default `SPLIT = 'test'` and `RUN_ABLATIONS = False`.

| Subfolder in `Data/` | Files to copy |
| --- | --- |
| [`Processed/`](https://drive.google.com/drive/folders/1FdUUKZCwH6jVSfpQSk9hANc0U7hdScvK) | `tests_test.parquet` |
| [`Features/`](https://drive.google.com/drive/folders/1GSsFGlmNPXlINn1WNHa1THdMo0sN7vGV) | `base_test.parquet`, `featA_test.parquet`, `featB_hist_test.parquet` |
| [`Models/Baseline/`](https://drive.google.com/drive/folders/1LIxDzHNPMOycGM_UDAZNvCNmdDoE9BgP) | `model_A.txt`, `run_config.json`, `ranking_evaluation.py` |
| [`Models/Advanced/`](https://drive.google.com/drive/folders/1KRKdEfyFkLveCe8Y8u8UxFnSl_Gghc-b) | `model_B.txt`, `run_config.json`, `ranking_evaluation.py` |

## Test results

We evaluate **25,767 held-out queries** with at least one positive candidate. NDCG@10 gives higher weight to purchases than to carts or clicks. MRR and Hit@10 treat any of these interactions as relevant.

| Ranking | NDCG@10 | MRR | Hit@10 |
| --- | ---: | ---: | ---: |
| Random | 0.0852 | 0.1022 | 0.2299 |
| Popularity only | 0.3400 | 0.3367 | 0.6394 |
| Model A | 0.3526 | 0.3503 | 0.6541 |
| **Model B** | **0.4060** | **0.4112** | **0.7046** |

Adding shopper history increased NDCG@10 by **0.0533 (+15.1% relative)**, with a 95% paired bootstrap interval of **[+0.0505, +0.0561]**. Most of the gain came from searches where a relevant product was already in the shopper’s history. For searches where every relevant product was new to that shopper, the average change was **−0.0034**. The saved outputs show the full breakdown. [Model comparison](Notebooks/ModelComparison.ipynb)

## Dataset citation

Jiongnan Liu, Zhicheng Dou, Guoyu Tang, and Sulong Xu. 2023. *JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions*. Proceedings of SIGIR 2023. [Paper](https://doi.org/10.1145/3539618.3591900)
