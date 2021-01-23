# **Natixis Data Challenge - Entreprise Risk Management**

![Sign Image](/img/natixis.jpg)

## **Group 4**

Charles Cazals: charles.cazals@hec.edu <br>
Jean Chillet: jean.chillet@hec.edu <br>
Antoine Demeire: antoine.demeire@hec.edu <br>
Katrin Dimitrova: katrin.dimitrova@hec.edu <br>
Alexdandre Leboucher: alexandre.leboucher@hec.edu <br>

## **Project Overview**

### Project Scope

### Resources

### Our Approach: using correlations and observed growth rates

Suppose we are missing value at timestamp **t** for time serie **i**:

- We look at the **growth rate** between time **t-1** and **t** for all available time series.
- To **weight** the actual relevance of the obtained growth rate for each time series, we use the overall correlation with the original time series **i**.

- We then infer the growth rate of series **i** at time **t**: <br> <br>
  <img src="https://render.githubusercontent.com/render/math?math=ImputedGrowthRate_i (t) = \frac{\sum_{j \neq i} GrowthRate_j (t)  *  Corr(i,j)} {\sum_{j \neq i} Corr(i,j)}">
  <br><br> where <img src="https://render.githubusercontent.com/render/math?math=Corr(i,j)"> is the correlation of **returns** (not absolute values) of series i and j across all period.

- And thus the value of series **i** at time **t**: <br> <br> <img src="https://render.githubusercontent.com/render/math?math=TimeSerie_i(t) = ImputedGrowthRate_i (t) * TimeSerie_i(t-1)">

Finally, instead of using all correlations raw, we can pre-process them before using them as weights. Here are a few examples:

![Sign Image](/img/activations.jpg)

## **Repository Presentation**

To understand how to use this repo, we advise to look at our [presentation](https://docs.google.com/presentation/d/1oCFw7ImvZkCLmg2PxTQyyKNNJZkJuBnCzzEaFrgWmoM/edit?usp=sharing) before hand.

The repository is divided into 5 main folders:

- **`./utils/`** :\
   This folder contains all utilities required to use this project: - The script **`setup.py`** loads the initial data set and creates a symbolic link to the credentials needed to access blob storage.<br>
  To run this script outside of **`main.py`** , type in the command line :
  `bash python3 setup.py ` - The script **`preprocessing.py`** loads the initial data set and saves two preprocessed datasets to `/data/preprocessing`.<br>
  `df_full`: data frame imputed using the baseline (linear interpolation) <br>
  `df_miss`: data frame with values missing at random <br>
  To run this script, only type in the command line :
  `bash python filtering.py <DATA_IN> <ACTION> `
  where : \
   `<DATA_IN>` is the initial data set to be processed.
  <br>
  <br>

- **`./notebooks/`** :\
   This folder contains examples on how to apply the different functions in a python notebook environment. - **`correlations.ipynb`** shows how to impute the dataset using the correlations-based model - **`evaluation.ipynb`** shows how to impute the dataset using one of the two methods and how to evaluate this method relative to the baseline.
  <br>
  <br>

- **`./data/`** :\
   This folder isn't pushed in this repo as it is to heavy. It contains sub-folders with the initial and processed data set.
  <br>
  <br>

- **`./results/`** :\
   Running the algorithm will write a CSV with imputed missing values inside this folder.
  <br>
  <br>

- **`./img/`** :\
   This folder contains image resources.
  <br>
  <br>

## **Run**

Please run in the command line:

```bash
python3 main.py <ACTION> <MODEL>
```

where : \
**`<ACTION>`** denotes the task you want to perform

- `impute` will use the selected model to impute the initial dataset
- `evaluate` will evaluate the imputation model against a reference dataset

**`<MODEL>`** denotes the model you want to use **(Facultative)**

- `xgboost` will use a regression-based model to impute series individually (based on relevant data)
- `correlations` will use a correlation-based model to impute series based on correlated assets

This will load the data (assuming you have the required credentials locally) process it and, depending on the ACTION, output an imputed dataset or evaluation results (pickled results dictionary and rmse boxplots) to `./results/`.
