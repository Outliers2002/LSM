# Load the necessary python libraries




```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

# Import datasets


```python
os.chdir("D:/SEM VI PROJECT")
```


```python
Landslide = pd.read_excel('Landslide_Coordinates.xlsx')
Landslide.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sr. No.</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Lithology</th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>Land Cover</th>
      <th>Watershed</th>
      <th>VARI</th>
      <th>Landslide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A1</td>
      <td>25.711032</td>
      <td>91.891803</td>
      <td>Gneissic Complex</td>
      <td>14.652761</td>
      <td>5.904561</td>
      <td>4.736543</td>
      <td>834</td>
      <td>69.08145</td>
      <td>0.463648</td>
      <td>11</td>
      <td>6</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2</td>
      <td>25.708426</td>
      <td>91.896406</td>
      <td>Gneissic Complex</td>
      <td>14.366824</td>
      <td>5.562040</td>
      <td>4.199919</td>
      <td>891</td>
      <td>75.79417</td>
      <td>0.833748</td>
      <td>11</td>
      <td>5</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A3</td>
      <td>25.695552</td>
      <td>91.903777</td>
      <td>Gneissic Complex</td>
      <td>12.980202</td>
      <td>4.125399</td>
      <td>2.789971</td>
      <td>879</td>
      <td>80.68076</td>
      <td>0.911049</td>
      <td>12</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A4</td>
      <td>25.663444</td>
      <td>91.915944</td>
      <td>Shillong Group</td>
      <td>9.652961</td>
      <td>1.596947</td>
      <td>0.556937</td>
      <td>948</td>
      <td>73.48592</td>
      <td>-0.228799</td>
      <td>12</td>
      <td>7</td>
      <td>0.004184</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A5</td>
      <td>25.668469</td>
      <td>91.975083</td>
      <td>Shillong Group</td>
      <td>12.895089</td>
      <td>4.706195</td>
      <td>4.716720</td>
      <td>1020</td>
      <td>80.62026</td>
      <td>-0.573571</td>
      <td>12</td>
      <td>3</td>
      <td>0.030568</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
Non_landslide = pd.read_excel('Non-Landslide_Coordinates.xlsx')
Non_landslide.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sr. No.</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Lithology</th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>Land Cover</th>
      <th>Watershed</th>
      <th>VARI</th>
      <th>Landslide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N1</td>
      <td>25.553235</td>
      <td>91.805072</td>
      <td>Shillong Group</td>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>9</td>
      <td>7</td>
      <td>-0.011111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N2</td>
      <td>25.553235</td>
      <td>91.805072</td>
      <td>Shillong Group</td>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>9</td>
      <td>7</td>
      <td>-0.011111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N3</td>
      <td>25.555388</td>
      <td>91.787007</td>
      <td>Shillong Group</td>
      <td>10.985616</td>
      <td>8.484740</td>
      <td>1.837939</td>
      <td>1680</td>
      <td>43.66924</td>
      <td>1.428899</td>
      <td>12</td>
      <td>1</td>
      <td>0.008368</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N4</td>
      <td>25.550497</td>
      <td>91.775404</td>
      <td>Shillong Group</td>
      <td>12.249265</td>
      <td>8.360397</td>
      <td>2.370490</td>
      <td>1684</td>
      <td>48.92371</td>
      <td>0.197396</td>
      <td>12</td>
      <td>6</td>
      <td>0.032389</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N5</td>
      <td>25.539885</td>
      <td>91.792327</td>
      <td>Shillong Group</td>
      <td>11.018937</td>
      <td>6.687753</td>
      <td>0.501227</td>
      <td>1243</td>
      <td>81.62968</td>
      <td>-0.907430</td>
      <td>9</td>
      <td>3</td>
      <td>-0.004484</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
ls = Landslide.drop(["Latitude","Longitude","Sr. No."],axis=1)
ls.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lithology</th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>Land Cover</th>
      <th>Watershed</th>
      <th>VARI</th>
      <th>Landslide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gneissic Complex</td>
      <td>14.652761</td>
      <td>5.904561</td>
      <td>4.736543</td>
      <td>834</td>
      <td>69.08145</td>
      <td>0.463648</td>
      <td>11</td>
      <td>6</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gneissic Complex</td>
      <td>14.366824</td>
      <td>5.562040</td>
      <td>4.199919</td>
      <td>891</td>
      <td>75.79417</td>
      <td>0.833748</td>
      <td>11</td>
      <td>5</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gneissic Complex</td>
      <td>12.980202</td>
      <td>4.125399</td>
      <td>2.789971</td>
      <td>879</td>
      <td>80.68076</td>
      <td>0.911049</td>
      <td>12</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shillong Group</td>
      <td>9.652961</td>
      <td>1.596947</td>
      <td>0.556937</td>
      <td>948</td>
      <td>73.48592</td>
      <td>-0.228799</td>
      <td>12</td>
      <td>7</td>
      <td>0.004184</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shillong Group</td>
      <td>12.895089</td>
      <td>4.706195</td>
      <td>4.716720</td>
      <td>1020</td>
      <td>80.62026</td>
      <td>-0.573571</td>
      <td>12</td>
      <td>3</td>
      <td>0.030568</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
nls = Non_landslide.drop(["Latitude","Longitude","Sr. No."],axis=1)
nls.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lithology</th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>Land Cover</th>
      <th>Watershed</th>
      <th>VARI</th>
      <th>Landslide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shillong Group</td>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>9</td>
      <td>7</td>
      <td>-0.011111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shillong Group</td>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>9</td>
      <td>7</td>
      <td>-0.011111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shillong Group</td>
      <td>10.985616</td>
      <td>8.484740</td>
      <td>1.837939</td>
      <td>1680</td>
      <td>43.66924</td>
      <td>1.428899</td>
      <td>12</td>
      <td>1</td>
      <td>0.008368</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shillong Group</td>
      <td>12.249265</td>
      <td>8.360397</td>
      <td>2.370490</td>
      <td>1684</td>
      <td>48.92371</td>
      <td>0.197396</td>
      <td>12</td>
      <td>6</td>
      <td>0.032389</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shillong Group</td>
      <td>11.018937</td>
      <td>6.687753</td>
      <td>0.501227</td>
      <td>1243</td>
      <td>81.62968</td>
      <td>-0.907430</td>
      <td>9</td>
      <td>3</td>
      <td>-0.004484</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.concat([ls,nls],axis=0)
sns.pairplot(data,hue = "Landslide", palette = "Set1")
```




    <seaborn.axisgrid.PairGrid at 0x24b7c79d1c0>




![png](output_8_1.png)



```python
plt.figure(figsize=(20,10)) 
sns.heatmap(data.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24b047b9640>




![png](output_9_1.png)


# Encoding



```python
ls.Lithology = pd.Categorical(ls.Lithology)
ls.rename(columns = {'Land Cover':'Land_Cover'}, inplace = True)
ls.Land_Cover = pd.Categorical(ls.Land_Cover)
ls.Watershed = pd.Categorical(ls.Watershed)
ls.dtypes
```




    Lithology                      category
    Distance from Shillong          float64
    Distance from Nearest River     float64
    Distance from Nearest Fault     float64
    Elevation                         int64
    Slope                           float64
    Aspect                          float64
    Land_Cover                     category
    Watershed                      category
    VARI                            float64
    Landslide                         int64
    dtype: object




```python
l1 = pd.get_dummies(ls, columns = ["Lithology"])
l2 = pd.get_dummies(l1, columns = ["Land_Cover"])
l = pd.get_dummies(l2, columns = ["Watershed"])
l
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.652761</td>
      <td>5.904561</td>
      <td>4.736543</td>
      <td>834</td>
      <td>69.08145</td>
      <td>0.463648</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.366824</td>
      <td>5.562040</td>
      <td>4.199919</td>
      <td>891</td>
      <td>75.79417</td>
      <td>0.833748</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.980202</td>
      <td>4.125399</td>
      <td>2.789971</td>
      <td>879</td>
      <td>80.68076</td>
      <td>0.911049</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.652961</td>
      <td>1.596947</td>
      <td>0.556937</td>
      <td>948</td>
      <td>73.48592</td>
      <td>-0.228799</td>
      <td>0.004184</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.895089</td>
      <td>4.706195</td>
      <td>4.716720</td>
      <td>1020</td>
      <td>80.62026</td>
      <td>-0.573571</td>
      <td>0.030568</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>94</th>
      <td>6.509233</td>
      <td>10.292946</td>
      <td>0.299994</td>
      <td>1326</td>
      <td>84.01508</td>
      <td>-0.291909</td>
      <td>0.027650</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>6.278230</td>
      <td>10.457434</td>
      <td>0.541542</td>
      <td>1357</td>
      <td>82.12014</td>
      <td>-0.688358</td>
      <td>0.036199</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>6.189393</td>
      <td>10.543218</td>
      <td>0.593796</td>
      <td>1362</td>
      <td>83.36880</td>
      <td>-0.508115</td>
      <td>0.031674</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>6.092116</td>
      <td>10.252313</td>
      <td>0.741958</td>
      <td>1442</td>
      <td>73.37644</td>
      <td>1.463107</td>
      <td>0.032558</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>98</th>
      <td>6.125016</td>
      <td>9.990172</td>
      <td>0.834280</td>
      <td>1469</td>
      <td>75.91467</td>
      <td>1.457645</td>
      <td>0.026667</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>99 rows × 26 columns</p>
</div>




```python
nls.Lithology = pd.Categorical(nls.Lithology)
nls.rename(columns = {'Land Cover':'Land_Cover'}, inplace = True)
nls.Land_Cover = pd.Categorical(nls.Land_Cover)
nls.Watershed = pd.Categorical(nls.Watershed)
nls.dtypes
```




    Lithology                      category
    Distance from Shillong          float64
    Distance from Nearest River     float64
    Distance from Nearest Fault     float64
    Elevation                         int64
    Slope                           float64
    Aspect                          float64
    Land_Cover                     category
    Watershed                      category
    VARI                            float64
    Landslide                         int64
    dtype: object




```python
nl1 = pd.get_dummies(nls,columns = ["Lithology"])
nl2 = pd.get_dummies(nl1, columns = ["Land_Cover"])
nl = pd.get_dummies(nl2, columns = ["Watershed"])
nl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_20</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>-0.011111</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>-0.011111</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.985616</td>
      <td>8.484740</td>
      <td>1.837939</td>
      <td>1680</td>
      <td>43.66924</td>
      <td>1.428899</td>
      <td>0.008368</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.249265</td>
      <td>8.360397</td>
      <td>2.370490</td>
      <td>1684</td>
      <td>48.92371</td>
      <td>0.197396</td>
      <td>0.032389</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.018937</td>
      <td>6.687753</td>
      <td>0.501227</td>
      <td>1243</td>
      <td>81.62968</td>
      <td>-0.907430</td>
      <td>-0.004484</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>114</th>
      <td>11.065703</td>
      <td>3.601773</td>
      <td>3.882881</td>
      <td>1088</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>0.004032</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115</th>
      <td>7.612916</td>
      <td>6.904581</td>
      <td>6.626066</td>
      <td>1336</td>
      <td>54.37941</td>
      <td>1.538549</td>
      <td>0.004132</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>116</th>
      <td>4.754206</td>
      <td>6.010839</td>
      <td>5.244779</td>
      <td>1579</td>
      <td>69.76953</td>
      <td>0.099669</td>
      <td>0.025974</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>11.311273</td>
      <td>2.915229</td>
      <td>0.265350</td>
      <td>1028</td>
      <td>70.36582</td>
      <td>-0.739975</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>7.918045</td>
      <td>1.647807</td>
      <td>0.754604</td>
      <td>978</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.269939</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>119 rows × 27 columns</p>
</div>



# Train (Landslide)


```python
train_l = l.sample(frac=0.8, random_state=10, replace=False, axis=0)
train_l.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>7.184743</td>
      <td>1.932943</td>
      <td>0.367224</td>
      <td>1173</td>
      <td>64.02213</td>
      <td>-0.909753</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7.933285</td>
      <td>4.999458</td>
      <td>2.672883</td>
      <td>1154</td>
      <td>81.57119</td>
      <td>-0.113603</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>12.670256</td>
      <td>17.109470</td>
      <td>10.027591</td>
      <td>1418</td>
      <td>75.21786</td>
      <td>-1.499489</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2.794152</td>
      <td>9.574601</td>
      <td>2.636359</td>
      <td>1459</td>
      <td>55.65931</td>
      <td>-0.785398</td>
      <td>0.003968</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>28.365434</td>
      <td>4.338422</td>
      <td>0.923927</td>
      <td>1254</td>
      <td>82.40997</td>
      <td>0.393869</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



# Test (Landslide) 


```python
subset_l = train_l.index
test_l = l.drop(subset_l)
test_l
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.652761</td>
      <td>5.904561</td>
      <td>4.736543</td>
      <td>834</td>
      <td>69.08145</td>
      <td>0.463648</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.308174</td>
      <td>7.971830</td>
      <td>3.625169</td>
      <td>1443</td>
      <td>76.01025</td>
      <td>-0.832981</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.168745</td>
      <td>6.273966</td>
      <td>3.249392</td>
      <td>1256</td>
      <td>77.61547</td>
      <td>-0.209023</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8.103366</td>
      <td>6.155130</td>
      <td>3.171120</td>
      <td>1273</td>
      <td>79.18964</td>
      <td>-0.332620</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.631597</td>
      <td>0.226270</td>
      <td>0.345184</td>
      <td>947</td>
      <td>70.00069</td>
      <td>-1.019141</td>
      <td>0.004000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8.419446</td>
      <td>0.503595</td>
      <td>0.247687</td>
      <td>994</td>
      <td>79.59646</td>
      <td>1.311791</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4.577974</td>
      <td>5.832644</td>
      <td>0.414137</td>
      <td>1291</td>
      <td>77.50984</td>
      <td>-0.410127</td>
      <td>0.028689</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4.548161</td>
      <td>5.598426</td>
      <td>0.602212</td>
      <td>1292</td>
      <td>81.59656</td>
      <td>-0.624886</td>
      <td>0.042194</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>4.596865</td>
      <td>5.770344</td>
      <td>0.422179</td>
      <td>1285</td>
      <td>78.06338</td>
      <td>-0.595918</td>
      <td>0.034632</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>19.349534</td>
      <td>10.858246</td>
      <td>4.059115</td>
      <td>859</td>
      <td>70.88644</td>
      <td>1.508377</td>
      <td>0.036530</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>6.301538</td>
      <td>12.693654</td>
      <td>10.633566</td>
      <td>1748</td>
      <td>67.84977</td>
      <td>0.147078</td>
      <td>-0.003968</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>20.855226</td>
      <td>21.397037</td>
      <td>1.543966</td>
      <td>1308</td>
      <td>78.49519</td>
      <td>-0.496423</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>21.421815</td>
      <td>21.506747</td>
      <td>0.912177</td>
      <td>1283</td>
      <td>73.79918</td>
      <td>0.446106</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>28.160580</td>
      <td>8.782651</td>
      <td>7.618463</td>
      <td>1506</td>
      <td>70.98907</td>
      <td>1.446441</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>27.459247</td>
      <td>8.044097</td>
      <td>3.164722</td>
      <td>1396</td>
      <td>63.57703</td>
      <td>0.179854</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>27.713246</td>
      <td>4.517201</td>
      <td>0.495904</td>
      <td>1196</td>
      <td>80.29961</td>
      <td>1.570796</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>29.578899</td>
      <td>12.640740</td>
      <td>15.195301</td>
      <td>1626</td>
      <td>77.32301</td>
      <td>-0.554307</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>21.998082</td>
      <td>8.207623</td>
      <td>8.716246</td>
      <td>1756</td>
      <td>57.42777</td>
      <td>-0.321751</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>12.823990</td>
      <td>6.885029</td>
      <td>3.689914</td>
      <td>1694</td>
      <td>76.16148</td>
      <td>-0.942815</td>
      <td>-0.013333</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>6.566114</td>
      <td>10.256608</td>
      <td>0.243635</td>
      <td>1326</td>
      <td>83.69102</td>
      <td>-0.743169</td>
      <td>0.027523</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 26 columns</p>
</div>



# Train (Non-Landslide)


```python
train_nl = nl.sample(frac=0.8, random_state=10, replace=False, axis=0)
train_nl.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_20</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>19.809354</td>
      <td>0.788160</td>
      <td>3.453853</td>
      <td>1570</td>
      <td>74.40382</td>
      <td>-1.000214</td>
      <td>0.031532</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>14.460544</td>
      <td>5.683955</td>
      <td>2.499275</td>
      <td>1261</td>
      <td>73.73318</td>
      <td>0.522404</td>
      <td>0.037500</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>8.127957</td>
      <td>11.917057</td>
      <td>11.927797</td>
      <td>1748</td>
      <td>60.83839</td>
      <td>-0.892134</td>
      <td>0.007937</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>20.015767</td>
      <td>8.520611</td>
      <td>2.072126</td>
      <td>1791</td>
      <td>73.60687</td>
      <td>0.119429</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>17.117923</td>
      <td>2.760979</td>
      <td>0.264057</td>
      <td>969</td>
      <td>77.69440</td>
      <td>-0.547153</td>
      <td>0.018779</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



# Test (Non-Landslide)


```python
subset_nl = train_nl.index
test_nl = nl.drop(subset_nl)
test_nl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_20</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>-0.011111</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.162213</td>
      <td>3.186076</td>
      <td>2.120412</td>
      <td>1734</td>
      <td>63.22770</td>
      <td>-0.881872</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.920759</td>
      <td>4.669743</td>
      <td>1.759617</td>
      <td>1738</td>
      <td>38.56893</td>
      <td>-1.284745</td>
      <td>0.004000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9.547487</td>
      <td>4.360323</td>
      <td>3.965550</td>
      <td>1787</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>0.029289</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6.460409</td>
      <td>11.481350</td>
      <td>10.413542</td>
      <td>1367</td>
      <td>46.01237</td>
      <td>-0.043451</td>
      <td>0.032922</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>7.915311</td>
      <td>10.445611</td>
      <td>10.096634</td>
      <td>1240</td>
      <td>72.30592</td>
      <td>0.683709</td>
      <td>0.032110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18.297602</td>
      <td>1.635622</td>
      <td>4.131495</td>
      <td>916</td>
      <td>47.39954</td>
      <td>-1.144169</td>
      <td>0.008065</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19.064082</td>
      <td>4.466721</td>
      <td>7.037536</td>
      <td>946</td>
      <td>75.24964</td>
      <td>-1.331565</td>
      <td>0.024390</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>25.332548</td>
      <td>11.767915</td>
      <td>0.609946</td>
      <td>938</td>
      <td>78.77841</td>
      <td>-1.215161</td>
      <td>0.033654</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>24.464295</td>
      <td>2.882315</td>
      <td>3.018219</td>
      <td>819</td>
      <td>62.53658</td>
      <td>0.188222</td>
      <td>0.029536</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>16.863144</td>
      <td>8.189220</td>
      <td>5.527071</td>
      <td>916</td>
      <td>31.12099</td>
      <td>-0.463647</td>
      <td>0.003968</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>18.907520</td>
      <td>3.642450</td>
      <td>0.740662</td>
      <td>1248</td>
      <td>70.29062</td>
      <td>-1.538549</td>
      <td>0.039130</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>10.283317</td>
      <td>15.805252</td>
      <td>15.041164</td>
      <td>1601</td>
      <td>68.93289</td>
      <td>0.982794</td>
      <td>-0.004630</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>8.681915</td>
      <td>16.010503</td>
      <td>13.440538</td>
      <td>1467</td>
      <td>84.59875</td>
      <td>-0.737243</td>
      <td>0.031915</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>15.840679</td>
      <td>20.045869</td>
      <td>10.502494</td>
      <td>1573</td>
      <td>50.09277</td>
      <td>1.225241</td>
      <td>0.029046</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>29.176947</td>
      <td>14.061931</td>
      <td>13.516024</td>
      <td>1684</td>
      <td>72.55692</td>
      <td>0.765401</td>
      <td>0.032258</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>28.417934</td>
      <td>13.751188</td>
      <td>12.605344</td>
      <td>1677</td>
      <td>56.53246</td>
      <td>0.530216</td>
      <td>0.008130</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>29.076250</td>
      <td>9.741970</td>
      <td>7.225154</td>
      <td>1470</td>
      <td>71.66394</td>
      <td>-1.267911</td>
      <td>0.026316</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>25.457765</td>
      <td>6.788406</td>
      <td>8.834618</td>
      <td>1644</td>
      <td>76.26496</td>
      <td>1.053014</td>
      <td>0.039823</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5.714698</td>
      <td>9.438736</td>
      <td>0.713949</td>
      <td>1241</td>
      <td>82.78623</td>
      <td>-0.737049</td>
      <td>0.026042</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>16.317976</td>
      <td>10.434450</td>
      <td>6.661862</td>
      <td>1011</td>
      <td>62.36830</td>
      <td>-0.752077</td>
      <td>0.028226</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>19.046556</td>
      <td>18.864426</td>
      <td>3.062125</td>
      <td>1174</td>
      <td>71.70188</td>
      <td>-0.933247</td>
      <td>0.028000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>11.162467</td>
      <td>4.850485</td>
      <td>2.827480</td>
      <td>957</td>
      <td>63.54272</td>
      <td>-0.594214</td>
      <td>0.037190</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>11.311273</td>
      <td>2.915229</td>
      <td>0.265350</td>
      <td>1028</td>
      <td>70.36582</td>
      <td>-0.739975</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 27 columns</p>
</div>



# Final Train 


```python
train_all = pd.concat([train_l, train_nl], axis = 0)
train_all["Land_Cover_20"] = train_all["Land_Cover_20"].fillna(0)
train_all.Land_Cover_20 = train_all.Land_Cover_20.astype(np.uint8)
train_all
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
      <th>Land_Cover_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>7.184743</td>
      <td>1.932943</td>
      <td>0.367224</td>
      <td>1173</td>
      <td>64.02213</td>
      <td>-0.909753</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7.933285</td>
      <td>4.999458</td>
      <td>2.672883</td>
      <td>1154</td>
      <td>81.57119</td>
      <td>-0.113603</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>12.670256</td>
      <td>17.109470</td>
      <td>10.027591</td>
      <td>1418</td>
      <td>75.21786</td>
      <td>-1.499489</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2.794152</td>
      <td>9.574601</td>
      <td>2.636359</td>
      <td>1459</td>
      <td>55.65931</td>
      <td>-0.785398</td>
      <td>0.003968</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>28.365434</td>
      <td>4.338422</td>
      <td>0.923927</td>
      <td>1254</td>
      <td>82.40997</td>
      <td>0.393869</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>13.727300</td>
      <td>19.767380</td>
      <td>10.895532</td>
      <td>1591</td>
      <td>69.49297</td>
      <td>-0.118092</td>
      <td>0.008065</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>27.393312</td>
      <td>14.466491</td>
      <td>10.649707</td>
      <td>1735</td>
      <td>46.49030</td>
      <td>0.348771</td>
      <td>0.040486</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102</th>
      <td>21.359525</td>
      <td>21.927141</td>
      <td>1.214724</td>
      <td>1273</td>
      <td>77.85667</td>
      <td>-0.812788</td>
      <td>0.027778</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>21.768656</td>
      <td>3.664025</td>
      <td>1.476358</td>
      <td>1471</td>
      <td>61.71889</td>
      <td>-1.449444</td>
      <td>0.026316</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>17.055532</td>
      <td>2.449394</td>
      <td>2.007448</td>
      <td>1601</td>
      <td>68.79291</td>
      <td>-1.138388</td>
      <td>0.008000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>174 rows × 27 columns</p>
</div>



# Final Test


```python
test_all = pd.concat([test_l, test_nl], axis = 0)
test_all["Land_Cover_20"] = test_all["Land_Cover_20"].fillna(0)
test_all.Land_Cover_20 = test_all.Land_Cover_20.astype(np.uint8)
test_all
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_22</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
      <th>Land_Cover_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.652761</td>
      <td>5.904561</td>
      <td>4.736543</td>
      <td>834</td>
      <td>69.08145</td>
      <td>0.463648</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.308174</td>
      <td>7.971830</td>
      <td>3.625169</td>
      <td>1443</td>
      <td>76.01025</td>
      <td>-0.832981</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.168745</td>
      <td>6.273966</td>
      <td>3.249392</td>
      <td>1256</td>
      <td>77.61547</td>
      <td>-0.209023</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8.103366</td>
      <td>6.155130</td>
      <td>3.171120</td>
      <td>1273</td>
      <td>79.18964</td>
      <td>-0.332620</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.631597</td>
      <td>0.226270</td>
      <td>0.345184</td>
      <td>947</td>
      <td>70.00069</td>
      <td>-1.019141</td>
      <td>0.004000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8.419446</td>
      <td>0.503595</td>
      <td>0.247687</td>
      <td>994</td>
      <td>79.59646</td>
      <td>1.311791</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4.577974</td>
      <td>5.832644</td>
      <td>0.414137</td>
      <td>1291</td>
      <td>77.50984</td>
      <td>-0.410127</td>
      <td>0.028689</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4.548161</td>
      <td>5.598426</td>
      <td>0.602212</td>
      <td>1292</td>
      <td>81.59656</td>
      <td>-0.624886</td>
      <td>0.042194</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>4.596865</td>
      <td>5.770344</td>
      <td>0.422179</td>
      <td>1285</td>
      <td>78.06338</td>
      <td>-0.595918</td>
      <td>0.034632</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>19.349534</td>
      <td>10.858246</td>
      <td>4.059115</td>
      <td>859</td>
      <td>70.88644</td>
      <td>1.508377</td>
      <td>0.036530</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>6.301538</td>
      <td>12.693654</td>
      <td>10.633566</td>
      <td>1748</td>
      <td>67.84977</td>
      <td>0.147078</td>
      <td>-0.003968</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>20.855226</td>
      <td>21.397037</td>
      <td>1.543966</td>
      <td>1308</td>
      <td>78.49519</td>
      <td>-0.496423</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>21.421815</td>
      <td>21.506747</td>
      <td>0.912177</td>
      <td>1283</td>
      <td>73.79918</td>
      <td>0.446106</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>28.160580</td>
      <td>8.782651</td>
      <td>7.618463</td>
      <td>1506</td>
      <td>70.98907</td>
      <td>1.446441</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>27.459247</td>
      <td>8.044097</td>
      <td>3.164722</td>
      <td>1396</td>
      <td>63.57703</td>
      <td>0.179854</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>27.713246</td>
      <td>4.517201</td>
      <td>0.495904</td>
      <td>1196</td>
      <td>80.29961</td>
      <td>1.570796</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>29.578899</td>
      <td>12.640740</td>
      <td>15.195301</td>
      <td>1626</td>
      <td>77.32301</td>
      <td>-0.554307</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>21.998082</td>
      <td>8.207623</td>
      <td>8.716246</td>
      <td>1756</td>
      <td>57.42777</td>
      <td>-0.321751</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>12.823990</td>
      <td>6.885029</td>
      <td>3.689914</td>
      <td>1694</td>
      <td>76.16148</td>
      <td>-0.942815</td>
      <td>-0.013333</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>6.566114</td>
      <td>10.256608</td>
      <td>0.243635</td>
      <td>1326</td>
      <td>83.69102</td>
      <td>-0.743169</td>
      <td>0.027523</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>9.301411</td>
      <td>7.965314</td>
      <td>0.350290</td>
      <td>1242</td>
      <td>83.46574</td>
      <td>-0.653810</td>
      <td>-0.011111</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.162213</td>
      <td>3.186076</td>
      <td>2.120412</td>
      <td>1734</td>
      <td>63.22770</td>
      <td>-0.881872</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.920759</td>
      <td>4.669743</td>
      <td>1.759617</td>
      <td>1738</td>
      <td>38.56893</td>
      <td>-1.284745</td>
      <td>0.004000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9.547487</td>
      <td>4.360323</td>
      <td>3.965550</td>
      <td>1787</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>0.029289</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6.460409</td>
      <td>11.481350</td>
      <td>10.413542</td>
      <td>1367</td>
      <td>46.01237</td>
      <td>-0.043451</td>
      <td>0.032922</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>7.915311</td>
      <td>10.445611</td>
      <td>10.096634</td>
      <td>1240</td>
      <td>72.30592</td>
      <td>0.683709</td>
      <td>0.032110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18.297602</td>
      <td>1.635622</td>
      <td>4.131495</td>
      <td>916</td>
      <td>47.39954</td>
      <td>-1.144169</td>
      <td>0.008065</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19.064082</td>
      <td>4.466721</td>
      <td>7.037536</td>
      <td>946</td>
      <td>75.24964</td>
      <td>-1.331565</td>
      <td>0.024390</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>25.332548</td>
      <td>11.767915</td>
      <td>0.609946</td>
      <td>938</td>
      <td>78.77841</td>
      <td>-1.215161</td>
      <td>0.033654</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>24.464295</td>
      <td>2.882315</td>
      <td>3.018219</td>
      <td>819</td>
      <td>62.53658</td>
      <td>0.188222</td>
      <td>0.029536</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>16.863144</td>
      <td>8.189220</td>
      <td>5.527071</td>
      <td>916</td>
      <td>31.12099</td>
      <td>-0.463647</td>
      <td>0.003968</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>18.907520</td>
      <td>3.642450</td>
      <td>0.740662</td>
      <td>1248</td>
      <td>70.29062</td>
      <td>-1.538549</td>
      <td>0.039130</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>10.283317</td>
      <td>15.805252</td>
      <td>15.041164</td>
      <td>1601</td>
      <td>68.93289</td>
      <td>0.982794</td>
      <td>-0.004630</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>8.681915</td>
      <td>16.010503</td>
      <td>13.440538</td>
      <td>1467</td>
      <td>84.59875</td>
      <td>-0.737243</td>
      <td>0.031915</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>15.840679</td>
      <td>20.045869</td>
      <td>10.502494</td>
      <td>1573</td>
      <td>50.09277</td>
      <td>1.225241</td>
      <td>0.029046</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>29.176947</td>
      <td>14.061931</td>
      <td>13.516024</td>
      <td>1684</td>
      <td>72.55692</td>
      <td>0.765401</td>
      <td>0.032258</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>28.417934</td>
      <td>13.751188</td>
      <td>12.605344</td>
      <td>1677</td>
      <td>56.53246</td>
      <td>0.530216</td>
      <td>0.008130</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>29.076250</td>
      <td>9.741970</td>
      <td>7.225154</td>
      <td>1470</td>
      <td>71.66394</td>
      <td>-1.267911</td>
      <td>0.026316</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>25.457765</td>
      <td>6.788406</td>
      <td>8.834618</td>
      <td>1644</td>
      <td>76.26496</td>
      <td>1.053014</td>
      <td>0.039823</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5.714698</td>
      <td>9.438736</td>
      <td>0.713949</td>
      <td>1241</td>
      <td>82.78623</td>
      <td>-0.737049</td>
      <td>0.026042</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>16.317976</td>
      <td>10.434450</td>
      <td>6.661862</td>
      <td>1011</td>
      <td>62.36830</td>
      <td>-0.752077</td>
      <td>0.028226</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>19.046556</td>
      <td>18.864426</td>
      <td>3.062125</td>
      <td>1174</td>
      <td>71.70188</td>
      <td>-0.933247</td>
      <td>0.028000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>11.162467</td>
      <td>4.850485</td>
      <td>2.827480</td>
      <td>957</td>
      <td>63.54272</td>
      <td>-0.594214</td>
      <td>0.037190</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>11.311273</td>
      <td>2.915229</td>
      <td>0.265350</td>
      <td>1028</td>
      <td>70.36582</td>
      <td>-0.739975</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>44 rows × 27 columns</p>
</div>



# Scaling (Standardisation)


```python
from sklearn.preprocessing import StandardScaler
```


```python
train_drop = train_all.drop(["Landslide","Lithology_Gneissic Complex","Lithology_Granite Pluton","Lithology_Shillong Group", "Land_Cover_1", "Land_Cover_2", "Land_Cover_9", "Land_Cover_11", "Land_Cover_12", "Land_Cover_13", "Land_Cover_22", "Watershed_1", "Watershed_2", "Watershed_3", "Watershed_4", "Watershed_5", "Watershed_6", "Watershed_7", "Watershed_8", "Land_Cover_20"],axis=1)
train_drop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>7.184743</td>
      <td>1.932943</td>
      <td>0.367224</td>
      <td>1173</td>
      <td>64.02213</td>
      <td>-0.909753</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7.933285</td>
      <td>4.999458</td>
      <td>2.672883</td>
      <td>1154</td>
      <td>81.57119</td>
      <td>-0.113603</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>12.670256</td>
      <td>17.109470</td>
      <td>10.027591</td>
      <td>1418</td>
      <td>75.21786</td>
      <td>-1.499489</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2.794152</td>
      <td>9.574601</td>
      <td>2.636359</td>
      <td>1459</td>
      <td>55.65931</td>
      <td>-0.785398</td>
      <td>0.003968</td>
    </tr>
    <tr>
      <th>66</th>
      <td>28.365434</td>
      <td>4.338422</td>
      <td>0.923927</td>
      <td>1254</td>
      <td>82.40997</td>
      <td>0.393869</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_drop = test_all.drop(["Landslide","Lithology_Gneissic Complex","Lithology_Granite Pluton","Lithology_Shillong Group", "Land_Cover_1", "Land_Cover_2", "Land_Cover_9", "Land_Cover_11", "Land_Cover_12", "Land_Cover_13", "Land_Cover_22", "Watershed_1", "Watershed_2", "Watershed_3", "Watershed_4", "Watershed_5", "Watershed_6", "Watershed_7", "Watershed_8", "Land_Cover_20"],axis=1)
test_drop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.652761</td>
      <td>5.904561</td>
      <td>4.736543</td>
      <td>834</td>
      <td>69.08145</td>
      <td>0.463648</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.308174</td>
      <td>7.971830</td>
      <td>3.625169</td>
      <td>1443</td>
      <td>76.01025</td>
      <td>-0.832981</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.168745</td>
      <td>6.273966</td>
      <td>3.249392</td>
      <td>1256</td>
      <td>77.61547</td>
      <td>-0.209023</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8.103366</td>
      <td>6.155130</td>
      <td>3.171120</td>
      <td>1273</td>
      <td>79.18964</td>
      <td>-0.332620</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.631597</td>
      <td>0.226270</td>
      <td>0.345184</td>
      <td>947</td>
      <td>70.00069</td>
      <td>-1.019141</td>
      <td>0.004</td>
    </tr>
  </tbody>
</table>
</div>




```python
scaler = StandardScaler()
train_drop_scaled_array = scaler.fit_transform(train_drop)
train_drop_scaled_array
```




    array([[-1.02404378, -1.324651  , -1.07009416, ..., -0.03417538,
            -0.97224869, -0.54163549],
           [-0.9259156 , -0.81595211, -0.51990677, ...,  0.95972036,
            -0.05926334, -0.54163549],
           [-0.30493435,  1.19295742,  1.23510901, ...,  0.59989781,
            -1.648528  , -0.54163549],
           ...,
           [ 0.83416353,  1.9921528 , -0.86785961, ...,  0.74934752,
            -0.86105438,  0.55788886],
           [ 0.88779756, -1.0374847 , -0.80542736, ..., -0.16462   ,
            -1.59113944,  0.50001916],
           [ 0.26994236, -1.23897787, -0.67869608, ...,  0.23601903,
            -1.23443607, -0.22497248]])




```python
scaler = StandardScaler()
test_drop_scaled_array = scaler.fit_transform(test_drop)
test_drop_scaled_array
```




    array([[-0.10767045, -0.54829169, -0.04607812, -1.6443576 ,  0.01697336,
             0.79987564, -0.90322445],
           [-0.89372507, -0.15752337, -0.29244617,  0.45035152,  0.56242709,
            -0.68089672, -0.90322445],
           [-0.91099945, -0.47846437, -0.3757482 , -0.19285145,  0.68879432,
             0.03167432, -0.90322445],
           [-0.91909951, -0.50092753, -0.39309941, -0.13437845,  0.81271721,
            -0.10947523, -0.90322445],
           [-0.85365491, -1.62163821, -1.01954981, -1.25568415,  0.0893384 ,
            -0.89349435, -0.66400785],
           [-0.87993918, -1.56921659, -1.04116291, -1.09402351,  0.84474317,
             1.76846927, -0.90322445],
           [-1.35587342, -0.56188587, -1.00426439, -0.07246587,  0.68047884,
            -0.19799026,  0.81246839],
           [-1.35956708, -0.60615923, -0.9625722 , -0.06902628,  1.00219641,
            -0.4432484 ,  1.62015743],
           [-1.35353288, -0.57366224, -1.00248169, -0.09310339,  0.724055  ,
            -0.41016633,  1.16791496],
           [ 0.47423024,  0.38808538, -0.19624967, -1.5583679 ,  0.15906702,
             1.99297485,  1.28140205],
           [-1.14233459,  0.73502579,  1.26116765,  1.49942587, -0.07998779,
             0.43834822, -1.14054251],
           [ 0.66077599,  2.3801946 , -0.75380509, -0.01399287,  0.75804824,
            -0.29654103, -0.90322445],
           [ 0.73097285,  2.40093268, -0.89385948, -0.09998257,  0.38836573,
             0.77984243, -0.90322445],
           [ 1.56586358, -0.00425681,  0.59278283,  0.66704556,  0.16714633,
             1.92224252, -0.90322445],
           [ 1.47897274, -0.14386298, -0.39451767,  0.28869088, -0.41634935,
             0.47577804, -0.90322445],
           [ 1.5104416 , -0.81053921, -0.98613837, -0.39922673,  0.90009702,
             2.06425831, -0.90322445],
           [ 1.74158438,  0.72502372,  2.27240834,  1.07979613,  0.66577108,
            -0.36264636, -0.90322445],
           [ 0.80236875, -0.11295219,  0.83613824,  1.52694257, -0.9004356 ,
            -0.0970624 , -0.90322445],
           [-0.33424379, -0.3629574 , -0.27809358,  1.31368811,  0.57433232,
            -0.80632872, -1.70061313],
           [-1.10955529,  0.27435988, -1.04206103,  0.04791972,  1.16707792,
            -0.57832986,  0.74276134],
           [-0.77066925, -0.15875509, -1.01841786, -0.24100568,  1.14934327,
            -0.47628004, -1.56771502],
           [-0.29234003, -1.06215701, -0.62601906,  1.45127163, -0.44384955,
            -0.73673051, -0.90322445],
           [-0.57004214, -0.78170469, -0.70599974,  1.46502999, -2.3850541 ,
            -1.19681847, -0.66400785],
           [-0.74018193, -0.84019323, -0.21699099,  1.6335698 , -2.22755678,
            -1.46344307,  0.84836155],
           [-1.12265157,  0.50586851,  1.21239304,  0.18894282, -1.79908653,
             0.22076062,  1.06563647],
           [-0.94239828,  0.31008644,  1.14214109, -0.24788486,  0.27081229,
             1.05118991,  1.01709231],
           [ 0.34390249, -1.35523367, -0.18020455, -1.36231138, -1.68988478,
            -1.03627832, -0.42093292],
           [ 0.43886447, -0.8200813 ,  0.46400354, -1.25912374,  0.50254983,
            -1.25028714,  0.55541336],
           [ 1.21548805,  0.56003672, -0.96085773, -1.28664044,  0.78034408,
            -1.11735175,  1.10941523],
           [ 1.10791706, -1.11957572, -0.4269942 , -1.69595142, -0.49825637,
             0.48533442,  0.86314286],
           [ 0.16618205, -0.11643082,  0.12916527, -1.36231138, -2.97137593,
            -0.2591108 , -0.6659064 ],
           [ 0.41946748, -0.9758902 , -0.93188058, -0.22036815,  0.11216247,
            -1.48666739,  1.43693796],
           [-0.64901724,  1.32319995,  2.23823943,  0.99380643,  0.00527832,
             1.39274929, -1.18009552],
           [-0.84742081,  1.36199769,  1.88341428,  0.53290163,  1.23853686,
            -0.57156136,  1.00541865],
           [ 0.0395051 ,  2.12478831,  1.23211184,  0.89749796, -1.47786649,
             1.66962809,  0.83382556],
           [ 1.69178504,  0.99366623,  1.90014781,  1.27929223,  0.29057169,
             1.14448282,  1.02594169],
           [ 1.59774806,  0.93492759,  1.69826935,  1.25521512, -0.97091681,
             0.87589792, -0.41701185],
           [ 1.67930933,  0.17707985,  0.50559463,  0.54322039,  0.22027391,
            -1.1775941 ,  0.67056898],
           [ 1.23100167, -0.38122161,  0.86237878,  1.14170871,  0.58247855,
             1.47294253,  1.47835676],
           [-1.2150404 ,  0.11976053, -0.93780233, -0.24444527,  1.09585042,
            -0.57133992,  0.6541753 ],
           [ 0.09863915,  0.30797672,  0.38072456, -1.03555052, -0.51150382,
            -0.58850273,  0.78479593],
           [ 0.43669314,  1.90146434, -0.41726135, -0.47489767,  0.22326064,
            -0.79540202,  0.77129176],
           [-0.54009602, -0.74753975, -0.46927713, -1.22128827, -0.41905033,
            -0.40821998,  1.32089685],
           [-0.52165988, -1.11335414, -1.03724737, -0.97707752,  0.11808242,
            -0.57468191, -0.90322445]])




```python
train_drop_scaled_df = pd.DataFrame(train_drop_scaled_array, columns = ["Distance from Shillong","Distance from Nearest River","Distance from Nearest Fault","Elevation","Slope","Aspect", "VARI"])
type(train_drop_scaled_df)
```




    pandas.core.frame.DataFrame




```python
test_drop_scaled_df = pd.DataFrame(test_drop_scaled_array, columns = ["Distance from Shillong","Distance from Nearest River","Distance from Nearest Fault","Elevation","Slope","Aspect", "VARI"])
type(test_drop_scaled_df)
```




    pandas.core.frame.DataFrame




```python
train_add = pd.DataFrame(train_all[['Landslide','Lithology_Gneissic Complex','Lithology_Granite Pluton','Lithology_Shillong Group','Land_Cover_1', 'Land_Cover_2', 'Land_Cover_9', 'Land_Cover_11','Land_Cover_12', 'Land_Cover_13','Land_Cover_22', 'Land_Cover_20', 'Watershed_1', 'Watershed_2', 'Watershed_3', 'Watershed_4', 'Watershed_5', 'Watershed_6', 'Watershed_7', 'Watershed_8']].copy())
train_add.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>Lithology_Shillong Group</th>
      <th>Land_Cover_1</th>
      <th>Land_Cover_2</th>
      <th>Land_Cover_9</th>
      <th>Land_Cover_11</th>
      <th>Land_Cover_12</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_add.to_csv('train_add_for_standardise.csv')
train_drop_scaled_df.to_csv('train_drop_scaled_df.csv')
```


```python
train_scaled = pd.read_csv('train_full_to_use_for_standardise.csv')
train_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.024044</td>
      <td>-1.324651</td>
      <td>-1.070094</td>
      <td>-0.670076</td>
      <td>-0.034175</td>
      <td>-0.972249</td>
      <td>-0.541635</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.925916</td>
      <td>-0.815952</td>
      <td>-0.519907</td>
      <td>-0.738715</td>
      <td>0.959720</td>
      <td>-0.059263</td>
      <td>-0.541635</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.304934</td>
      <td>1.192957</td>
      <td>1.235109</td>
      <td>0.215012</td>
      <td>0.599898</td>
      <td>-1.648528</td>
      <td>-0.541635</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.599617</td>
      <td>-0.056989</td>
      <td>-0.528622</td>
      <td>0.363129</td>
      <td>-0.507806</td>
      <td>-0.829645</td>
      <td>-0.384561</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.752585</td>
      <td>-0.925610</td>
      <td>-0.937251</td>
      <td>-0.377455</td>
      <td>1.007225</td>
      <td>0.522680</td>
      <td>-0.541635</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
test_add = pd.DataFrame(test_all[['Landslide','Lithology_Gneissic Complex','Lithology_Granite Pluton','Lithology_Shillong Group','Land_Cover_1', 'Land_Cover_2', 'Land_Cover_9', 'Land_Cover_11','Land_Cover_12', 'Land_Cover_13','Land_Cover_22', 'Land_Cover_20', 'Watershed_1', 'Watershed_2', 'Watershed_3', 'Watershed_4', 'Watershed_5', 'Watershed_6', 'Watershed_7', 'Watershed_8']].copy())
test_add.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>Lithology_Shillong Group</th>
      <th>Land_Cover_1</th>
      <th>Land_Cover_2</th>
      <th>Land_Cover_9</th>
      <th>Land_Cover_11</th>
      <th>Land_Cover_12</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_add.to_csv('test_add_for_standardise.csv')
test_drop_scaled_df.to_csv('test_drop_scaled_df.csv')
```


```python
test_scaled = pd.read_csv('test_full_to_use_for_standardise.csv')
test_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Landslide</th>
      <th>Lithology_Gneissic Complex</th>
      <th>Lithology_Granite Pluton</th>
      <th>...</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.107670</td>
      <td>-0.548292</td>
      <td>-0.046078</td>
      <td>-1.644358</td>
      <td>0.016973</td>
      <td>0.799876</td>
      <td>-0.903224</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.893725</td>
      <td>-0.157523</td>
      <td>-0.292446</td>
      <td>0.450352</td>
      <td>0.562427</td>
      <td>-0.680897</td>
      <td>-0.903224</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.910999</td>
      <td>-0.478464</td>
      <td>-0.375748</td>
      <td>-0.192851</td>
      <td>0.688794</td>
      <td>0.031674</td>
      <td>-0.903224</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.919100</td>
      <td>-0.500928</td>
      <td>-0.393099</td>
      <td>-0.134378</td>
      <td>0.812717</td>
      <td>-0.109475</td>
      <td>-0.903224</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.853655</td>
      <td>-1.621638</td>
      <td>-1.019550</td>
      <td>-1.255684</td>
      <td>0.089338</td>
      <td>-0.893494</td>
      <td>-0.664008</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
train_scaled.rename(columns = {'Lithology_Gneissic Complex':'Lithology_Gneissic_Complex', 'Lithology_Granite Pluton':'Lithology_Granite_Pluton', 'Lithology_Shillong Group':'Lithology_Shillong_Group'}, inplace = True)
print(train_scaled.columns)
```

    Index(['Distance from Shillong', 'Distance from Nearest River',
           'Distance from Nearest Fault', 'Elevation', 'Slope', 'Aspect', 'VARI',
           'Landslide', 'Lithology_Gneissic_Complex', 'Lithology_Granite_Pluton',
           'Lithology_Shillong_Group', 'Land_Cover_1', 'Land_Cover_2',
           'Land_Cover_9', 'Land_Cover_11', 'Land_Cover_12', 'Land_Cover_13',
           'Land_Cover_22', 'Land_Cover_20', 'Watershed_1', 'Watershed_2',
           'Watershed_3', 'Watershed_4', 'Watershed_5', 'Watershed_6',
           'Watershed_7', 'Watershed_8'],
          dtype='object')
    


```python
test_scaled.rename(columns = {'Lithology_Gneissic Complex':'Lithology_Gneissic_Complex', 'Lithology_Granite Pluton':'Lithology_Granite_Pluton', 'Lithology_Shillong Group':'Lithology_Shillong_Group'}, inplace = True)
print(test_scaled.columns)
```

    Index(['Distance from Shillong', 'Distance from Nearest River',
           'Distance from Nearest Fault', 'Elevation', 'Slope', 'Aspect', 'VARI',
           'Landslide', 'Lithology_Gneissic_Complex', 'Lithology_Granite_Pluton',
           'Lithology_Shillong_Group', 'Land_Cover_1', 'Land_Cover_2',
           'Land_Cover_9', 'Land_Cover_11', 'Land_Cover_12', 'Land_Cover_13',
           'Land_Cover_22', 'Land_Cover_20', 'Watershed_1', 'Watershed_2',
           'Watershed_3', 'Watershed_4', 'Watershed_5', 'Watershed_6',
           'Watershed_7', 'Watershed_8'],
          dtype='object')
    

# Training and Evaluating Different Models


```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report,confusion_matrix
```


```python
x_train = train_scaled.drop("Landslide", axis=1)
y_train = train_scaled["Landslide"]
```


```python
x_test = test_scaled.drop("Landslide", axis=1)
y_test = test_scaled["Landslide"]
```

# SVM without Kernel Trick


```python
model_og = SVC()
model_og.fit(x_train, y_train)
```




    SVC()




```python
predictions_og = model_og.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_og))
```

    [[20  4]
     [ 6 14]]
    


```python
print(classification_report(y_test, predictions_og))
#Precision - 78% of the predicted landslides were actual landslides
#Recall - 70% of the actual landslides were predicted correctly
#F1 score is 0.74 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
#Test Accuracy - 77%
```

                  precision    recall  f1-score   support
    
               0       0.77      0.83      0.80        24
               1       0.78      0.70      0.74        20
    
        accuracy                           0.77        44
       macro avg       0.77      0.77      0.77        44
    weighted avg       0.77      0.77      0.77        44
    
    

# Linear Kernel


```python
param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
svc_l = SVC(kernel='linear')
grid_search = GridSearchCV(svc_l, param_grid, cv=10)
grid_search.fit(x_train, y_train)
```




    GridSearchCV(cv=10, estimator=SVC(kernel='linear'),
                 param_grid={'C': [0.1, 0.5, 1, 5, 10, 50, 100]})




```python
print(f"Best C parameter value: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

    Best C parameter value: {'C': 0.1}
    Best score: 0.7699346405228757
    


```python
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)
print('Test score:', test_score)
```

    Test score: 0.7272727272727273
    


```python
model_l = SVC(C = 0.1, kernel='linear')
model_l.fit(x_train, y_train)
```




    SVC(C=0.1, kernel='linear')




```python
predictions_l = model_l.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_l))
```

    [[19  5]
     [ 7 13]]
    


```python
print(classification_report(y_test, predictions_l))
#Precision - 72% of the predicted landslides were actual landslides
#Recall - 65% of the actual landslides were predicted correctly
#F1 score is 0.68 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
#Test Accuracy - 73%
```

                  precision    recall  f1-score   support
    
               0       0.73      0.79      0.76        24
               1       0.72      0.65      0.68        20
    
        accuracy                           0.73        44
       macro avg       0.73      0.72      0.72        44
    weighted avg       0.73      0.73      0.73        44
    
    

# Polynomial (Degree 2) Kernel


```python
param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
svc_p2 = SVC(kernel='poly', degree = 2)
grid_search = GridSearchCV(svc_p2, param_grid, cv=10)
grid_search.fit(x_train, y_train)
```




    GridSearchCV(cv=10, estimator=SVC(degree=2, kernel='poly'),
                 param_grid={'C': [0.1, 0.5, 1, 5, 10, 50, 100]})




```python
print(f"Best C parameter value: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

    Best C parameter value: {'C': 1}
    Best score: 0.7545751633986928
    


```python
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)
print('Test score:', test_score)
```

    Test score: 0.7954545454545454
    


```python
model_p2 = SVC(C = 1, kernel='poly', degree = 2)
model_p2.fit(x_train, y_train)
```




    SVC(C=1, degree=2, kernel='poly')




```python
predictions_p2 = model_p2.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_p2))
```

    [[20  4]
     [ 5 15]]
    


```python
print(classification_report(y_test, predictions_p2))
#Precision - 79% of the predicted landslides were actual landslides
#Recall - 75% of the actual landslides were predicted correctly
#F1 score is 0.77 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
#Test Accuracy - 80%
```

                  precision    recall  f1-score   support
    
               0       0.80      0.83      0.82        24
               1       0.79      0.75      0.77        20
    
        accuracy                           0.80        44
       macro avg       0.79      0.79      0.79        44
    weighted avg       0.80      0.80      0.79        44
    
    

# Polynomial (Degree 3) Kernel


```python
param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
svc_p3 = SVC(kernel='poly', degree = 3)
grid_search = GridSearchCV(svc_p2, param_grid, cv=10)
grid_search.fit(x_train, y_train)
```




    GridSearchCV(cv=10, estimator=SVC(degree=2, kernel='poly'),
                 param_grid={'C': [0.1, 0.5, 1, 5, 10, 50, 100]})




```python
print(f"Best C parameter value: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

    Best C parameter value: {'C': 1}
    Best score: 0.7545751633986928
    


```python
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)
print('Test score:', test_score)
```

    Test score: 0.7954545454545454
    


```python
model_p3 = SVC(C = 1, kernel='poly', degree = 3)
model_p3.fit(x_train, y_train)
```




    SVC(C=1, kernel='poly')




```python
predictions_p3 = model_p3.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_p3))
```

    [[21  3]
     [ 6 14]]
    


```python
print(classification_report(y_test, predictions_p3))
#Precision - 82% of the predicted landslides were actual landslides
#Recall - 70% of the actual landslides were predicted correctly
#F1 score is 0.76 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
#Test Accuracy - 80%
```

                  precision    recall  f1-score   support
    
               0       0.78      0.88      0.82        24
               1       0.82      0.70      0.76        20
    
        accuracy                           0.80        44
       macro avg       0.80      0.79      0.79        44
    weighted avg       0.80      0.80      0.79        44
    
    

# Polynomial (Degree 4) Kernel


```python
param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
svc_p4 = SVC(kernel='poly', degree = 4)
grid_search = GridSearchCV(svc_p4, param_grid, cv=10)
grid_search.fit(x_train, y_train)
```




    GridSearchCV(cv=10, estimator=SVC(degree=4, kernel='poly'),
                 param_grid={'C': [0.1, 0.5, 1, 5, 10, 50, 100]})




```python
print(f"Best C parameter value: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

    Best C parameter value: {'C': 5}
    Best score: 0.7421568627450981
    


```python
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)
print('Test score:', test_score)
```

    Test score: 0.8181818181818182
    


```python
model_p4 = SVC(C = 5, kernel='poly', degree = 4)
model_p4.fit(x_train, y_train)
```




    SVC(C=5, degree=4, kernel='poly')




```python
predictions_p4 = model_p4.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_p4))
```

    [[22  2]
     [ 6 14]]
    


```python
print(classification_report(y_test, predictions_p4))
#Precision - 88% of the predicted landslides were actual landslides
#Recall - 70% of the actual landslides were predicted correctly
#F1 score is 0.78 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
#Test Accuracy - 82%
```

                  precision    recall  f1-score   support
    
               0       0.79      0.92      0.85        24
               1       0.88      0.70      0.78        20
    
        accuracy                           0.82        44
       macro avg       0.83      0.81      0.81        44
    weighted avg       0.83      0.82      0.82        44
    
    

# RBF Kernel


```python
param_grid = {'C': [0.1, 1, 5, 10, 50, 100, 500, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']} 
svc_r = SVC(kernel='rbf')
grid_search = GridSearchCV(svc_r, param_grid, cv=10)
grid_search.fit(x_train, y_train)
```




    GridSearchCV(cv=10, estimator=SVC(),
                 param_grid={'C': [0.1, 1, 5, 10, 50, 100, 500, 1000],
                             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                             'kernel': ['rbf']})




```python
print(f"Best C parameter value: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

    Best C parameter value: {'C': 5, 'gamma': 0.1, 'kernel': 'rbf'}
    Best score: 0.7872549019607843
    


```python
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)
print('Test score:', test_score)
```

    Test score: 0.8409090909090909
    


```python
model_r = SVC(C = 5, gamma = 0.1, kernel='rbf')
model_r.fit(x_train, y_train)
```




    SVC(C=5, gamma=0.1)




```python
predictions_r = model_r.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_r))
```

    [[21  3]
     [ 4 16]]
    


```python
print(classification_report(y_test, predictions_r))
#Precision - 84% of the predicted landslides were actual landslides
#Recall - 80% of the actual landslides were predicted correctly
#F1 score is 0.82 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
#Test Accuracy - 84%
```

                  precision    recall  f1-score   support
    
               0       0.84      0.88      0.86        24
               1       0.84      0.80      0.82        20
    
        accuracy                           0.84        44
       macro avg       0.84      0.84      0.84        44
    weighted avg       0.84      0.84      0.84        44
    
    

# RBF BEST MODEL USING GRID AND RANDOMIZED SEARCH


```python
# Grid Search for tuning parameters
from sklearn.model_selection import GridSearchCV
# RandomizedSearch for tuning (possibly faster than GridSearch)
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
```


```python
# CHERCHEZ FOR PARAMETERS
def cherchez(estimator, param_grid, search):
    try:
        if search == "grid":
                clf = GridSearchCV(
                estimator=estimator, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=10, 
                verbose=0,
                return_train_score=True
            )
        elif search == "random":           
                clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=100,
                n_jobs=-1,
                cv=10,
                verbose=0,
                random_state=10,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')
        sys.exit(0)
        # Fit the model
    clf.fit(X = x_train, y = y_train)
    
    return clf   
```


```python
# SVM
svm_param = {
    "C": [.001, .01, .1, 1, 5, 10, 50, 100, 1000],
    "gamma": [0, .0001, .001, .01, .1, 1, 5, 10, 100],
    "kernel": ["rbf"],
    "random_state": [10]
}

svm_dist = {
    "C": expon(scale=.01),
    "gamma": expon(scale=.01),
    "kernel": ["rbf"],
    "random_state": [10]
}

svm_grid = cherchez(SVC(), svm_param, "grid")
acc_g = accuracy_score(y_true=y_test, y_pred=svm_grid.predict(x_test))
cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=svm_grid.predict(x_test))
print("**Grid search results**")
print("Best training accuracy:\t", svm_grid.best_score_)
print("Test accuracy:\t", acc_g)

svm_random = cherchez(SVC(), svm_dist, "random")
acc_r = accuracy_score(y_true=y_test, y_pred=svm_random.predict(x_test))
cfmatrix_rand = confusion_matrix(y_true=y_test, y_pred=svm_random.predict(x_test))
print("**Random search results**")
print("Best training accuracy:\t", svm_random.best_score_)
print("Test accuracy:\t", acc_r)
```

    **Grid search results**
    Best training accuracy:	 0.7872549019607843
    Test accuracy:	 0.8409090909090909
    **Random search results**
    Best training accuracy:	 0.5457516339869282
    Test accuracy:	 0.5454545454545454
    


```python
best_model = svm_grid.best_estimator_
test_score = svm_grid.score(x_test, y_test)
print('Test accuracy:', acc_g)
```

    Test accuracy: 0.8409090909090909
    


```python
best_model
```




    SVC(C=5, gamma=0.1, random_state=10)




```python
model_r_imp = SVC(C = 5, gamma = 0.1, kernel='rbf', random_state = 10)
model_r_imp.fit(x_train, y_train)
```




    SVC(C=5, gamma=0.1, random_state=10)




```python
predictions_r_imp = model_r_imp.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_r_imp))
```

    [[21  3]
     [ 4 16]]
    


```python
print(classification_report(y_test, predictions_r_imp))
#Precision - 84% of the predicted landslides were actual landslides
#Recall - 80% of the actual landslides were predicted correctly
#F1 score is 0.82 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
#Test Accuracy - 84%
```

                  precision    recall  f1-score   support
    
               0       0.84      0.88      0.86        24
               1       0.84      0.80      0.82        20
    
        accuracy                           0.84        44
       macro avg       0.84      0.84      0.84        44
    weighted avg       0.84      0.84      0.84        44
    
    

# Prediction on Final Set


```python
mt = pd.read_excel('Prediction_Set.xlsx')
mt_1 = mt.drop(['Latitude','Longitude'], axis=1)
mt_2 = mt_1.set_index('Sr. No.')
mt_2.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lithology</th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>Land Cover</th>
      <th>Watershed</th>
      <th>VARI</th>
    </tr>
    <tr>
      <th>Sr. No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>Shillong Group</td>
      <td>9.281485</td>
      <td>0.828798</td>
      <td>0.183555</td>
      <td>952</td>
      <td>61.41991</td>
      <td>-0.197396</td>
      <td>12</td>
      <td>6</td>
      <td>0.004255</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>Shillong Group</td>
      <td>8.899308</td>
      <td>0.203190</td>
      <td>0.453964</td>
      <td>929</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>12</td>
      <td>8</td>
      <td>-0.008197</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>Shillong Group</td>
      <td>8.306454</td>
      <td>0.595944</td>
      <td>0.190703</td>
      <td>1020</td>
      <td>79.37641</td>
      <td>1.190290</td>
      <td>12</td>
      <td>8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>Shillong Group</td>
      <td>8.019259</td>
      <td>0.838066</td>
      <td>0.281174</td>
      <td>1050</td>
      <td>79.59367</td>
      <td>1.562532</td>
      <td>11</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>Shillong Group</td>
      <td>7.786250</td>
      <td>1.081203</td>
      <td>0.514782</td>
      <td>1082</td>
      <td>80.53983</td>
      <td>-0.188572</td>
      <td>9</td>
      <td>6</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>Shillong Group</td>
      <td>5.801278</td>
      <td>3.175267</td>
      <td>0.949701</td>
      <td>1234</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>12</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>Shillong Group</td>
      <td>5.465077</td>
      <td>3.514135</td>
      <td>1.236325</td>
      <td>1240</td>
      <td>76.33635</td>
      <td>0.175940</td>
      <td>12</td>
      <td>6</td>
      <td>0.033473</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>Shillong Group</td>
      <td>5.379050</td>
      <td>3.635879</td>
      <td>1.077322</td>
      <td>1229</td>
      <td>59.50669</td>
      <td>-0.558599</td>
      <td>12</td>
      <td>7</td>
      <td>0.025532</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>Shillong Group</td>
      <td>5.278350</td>
      <td>3.676023</td>
      <td>1.267916</td>
      <td>1239</td>
      <td>76.98981</td>
      <td>-0.440396</td>
      <td>12</td>
      <td>3</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>Shillong Group</td>
      <td>5.194040</td>
      <td>3.664602</td>
      <td>1.755296</td>
      <td>1302</td>
      <td>77.35306</td>
      <td>-0.813962</td>
      <td>12</td>
      <td>3</td>
      <td>0.033473</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>Shillong Group</td>
      <td>5.381736</td>
      <td>3.800879</td>
      <td>2.837396</td>
      <td>1334</td>
      <td>79.37197</td>
      <td>0.084544</td>
      <td>12</td>
      <td>2</td>
      <td>0.033755</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>Shillong Group</td>
      <td>4.819228</td>
      <td>4.366581</td>
      <td>3.095202</td>
      <td>1369</td>
      <td>77.77071</td>
      <td>-1.212026</td>
      <td>12</td>
      <td>4</td>
      <td>0.028571</td>
    </tr>
  </tbody>
</table>
</div>




```python
mt_2.Lithology = pd.Categorical(mt_2.Lithology)
mt_2.rename(columns = {'Land Cover':'Land_Cover'}, inplace = True)
mt_2.Land_Cover = pd.Categorical(mt_2.Land_Cover)
mt_2.Watershed = pd.Categorical(mt_2.Watershed)
mt_2.dtypes
```




    Lithology                      category
    Distance from Shillong          float64
    Distance from Nearest River     float64
    Distance from Nearest Fault     float64
    Elevation                         int64
    Slope                           float64
    Aspect                          float64
    Land_Cover                     category
    Watershed                      category
    VARI                            float64
    dtype: object




```python
mt_3 = pd.get_dummies(mt_2, columns = ["Lithology"])
mt_3a = pd.get_dummies(mt_3, columns = ["Land_Cover"])
mt_4 = pd.get_dummies(mt_3a, columns = ["Watershed"])
mt_4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Lithology_Shillong Group</th>
      <th>Land_Cover_9</th>
      <th>Land_Cover_11</th>
      <th>Land_Cover_12</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
    <tr>
      <th>Sr. No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>9.281485</td>
      <td>0.828798</td>
      <td>0.183555</td>
      <td>952</td>
      <td>61.41991</td>
      <td>-0.197396</td>
      <td>0.004255</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>8.899308</td>
      <td>0.203190</td>
      <td>0.453964</td>
      <td>929</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>-0.008197</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>8.306454</td>
      <td>0.595944</td>
      <td>0.190703</td>
      <td>1020</td>
      <td>79.37641</td>
      <td>1.190290</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>8.019259</td>
      <td>0.838066</td>
      <td>0.281174</td>
      <td>1050</td>
      <td>79.59367</td>
      <td>1.562532</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>7.786250</td>
      <td>1.081203</td>
      <td>0.514782</td>
      <td>1082</td>
      <td>80.53983</td>
      <td>-0.188572</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>5.801278</td>
      <td>3.175267</td>
      <td>0.949701</td>
      <td>1234</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>5.465077</td>
      <td>3.514135</td>
      <td>1.236325</td>
      <td>1240</td>
      <td>76.33635</td>
      <td>0.175940</td>
      <td>0.033473</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>5.379050</td>
      <td>3.635879</td>
      <td>1.077322</td>
      <td>1229</td>
      <td>59.50669</td>
      <td>-0.558599</td>
      <td>0.025532</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>5.278350</td>
      <td>3.676023</td>
      <td>1.267916</td>
      <td>1239</td>
      <td>76.98981</td>
      <td>-0.440396</td>
      <td>0.033333</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>5.194040</td>
      <td>3.664602</td>
      <td>1.755296</td>
      <td>1302</td>
      <td>77.35306</td>
      <td>-0.813962</td>
      <td>0.033473</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>5.381736</td>
      <td>3.800879</td>
      <td>2.837396</td>
      <td>1334</td>
      <td>79.37197</td>
      <td>0.084544</td>
      <td>0.033755</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>4.819228</td>
      <td>4.366581</td>
      <td>3.095202</td>
      <td>1369</td>
      <td>77.77071</td>
      <td>-1.212026</td>
      <td>0.028571</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Saving the columns in a list
cols = train_scaled.columns.tolist()
cols.remove("Landslide")
cols
```




    ['Distance from Shillong',
     'Distance from Nearest River',
     'Distance from Nearest Fault',
     'Elevation',
     'Slope',
     'Aspect',
     'VARI',
     'Lithology_Gneissic_Complex',
     'Lithology_Granite_Pluton',
     'Lithology_Shillong_Group',
     'Land_Cover_1',
     'Land_Cover_2',
     'Land_Cover_9',
     'Land_Cover_11',
     'Land_Cover_12',
     'Land_Cover_13',
     'Land_Cover_22',
     'Land_Cover_20',
     'Watershed_1',
     'Watershed_2',
     'Watershed_3',
     'Watershed_4',
     'Watershed_5',
     'Watershed_6',
     'Watershed_7',
     'Watershed_8']




```python
mtp_add_on = pd.DataFrame(columns = ['Lithology_Gneissic_Complex','Lithology_Granite_Pluton','Land_Cover_1','Land_Cover_2','Land_Cover_13','Land_Cover_22','Land_Cover_20','Watershed_5'])
mtp = pd.concat([mt_4, mtp_add_on], axis = 1)
mtp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Lithology_Shillong Group</th>
      <th>Land_Cover_9</th>
      <th>Land_Cover_11</th>
      <th>...</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
      <th>Lithology_Gneissic_Complex</th>
      <th>Lithology_Granite_Pluton</th>
      <th>Land_Cover_1</th>
      <th>Land_Cover_2</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>9.281485</td>
      <td>0.828798</td>
      <td>0.183555</td>
      <td>952</td>
      <td>61.41991</td>
      <td>-0.197396</td>
      <td>0.004255</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>8.899308</td>
      <td>0.203190</td>
      <td>0.453964</td>
      <td>929</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>-0.008197</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>8.306454</td>
      <td>0.595944</td>
      <td>0.190703</td>
      <td>1020</td>
      <td>79.37641</td>
      <td>1.190290</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>8.019259</td>
      <td>0.838066</td>
      <td>0.281174</td>
      <td>1050</td>
      <td>79.59367</td>
      <td>1.562532</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>7.786250</td>
      <td>1.081203</td>
      <td>0.514782</td>
      <td>1082</td>
      <td>80.53983</td>
      <td>-0.188572</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>5.801278</td>
      <td>3.175267</td>
      <td>0.949701</td>
      <td>1234</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>5.465077</td>
      <td>3.514135</td>
      <td>1.236325</td>
      <td>1240</td>
      <td>76.33635</td>
      <td>0.175940</td>
      <td>0.033473</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>5.379050</td>
      <td>3.635879</td>
      <td>1.077322</td>
      <td>1229</td>
      <td>59.50669</td>
      <td>-0.558599</td>
      <td>0.025532</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>5.278350</td>
      <td>3.676023</td>
      <td>1.267916</td>
      <td>1239</td>
      <td>76.98981</td>
      <td>-0.440396</td>
      <td>0.033333</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>5.194040</td>
      <td>3.664602</td>
      <td>1.755296</td>
      <td>1302</td>
      <td>77.35306</td>
      <td>-0.813962</td>
      <td>0.033473</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>5.381736</td>
      <td>3.800879</td>
      <td>2.837396</td>
      <td>1334</td>
      <td>79.37197</td>
      <td>0.084544</td>
      <td>0.033755</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>4.819228</td>
      <td>4.366581</td>
      <td>3.095202</td>
      <td>1369</td>
      <td>77.77071</td>
      <td>-1.212026</td>
      <td>0.028571</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 26 columns</p>
</div>




```python
print(mtp.columns)
```

    Index(['Distance from Shillong', 'Distance from Nearest River',
           'Distance from Nearest Fault', 'Elevation', 'Slope', 'Aspect', 'VARI',
           'Lithology_Shillong Group', 'Land_Cover_9', 'Land_Cover_11',
           'Land_Cover_12', 'Watershed_1', 'Watershed_2', 'Watershed_3',
           'Watershed_4', 'Watershed_6', 'Watershed_7', 'Watershed_8',
           'Lithology_Gneissic_Complex', 'Lithology_Granite_Pluton',
           'Land_Cover_1', 'Land_Cover_2', 'Land_Cover_13', 'Land_Cover_22',
           'Land_Cover_20', 'Watershed_5'],
          dtype='object')
    


```python
new_order = [0,1,2,3,4,5,6,18,19,7,20,21,8,9,10,22,23,24,11,12,13,14,25,15,16,17]
mtp_ordered = (mtp[mtp.columns[new_order]]) 
mtp_ordered.rename(columns = {'Lithology_Shillong Group':'Lithology_Shillong_Group'}, inplace = True)
mtp_ordered
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Lithology_Gneissic_Complex</th>
      <th>Lithology_Granite_Pluton</th>
      <th>Lithology_Shillong_Group</th>
      <th>...</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>9.281485</td>
      <td>0.828798</td>
      <td>0.183555</td>
      <td>952</td>
      <td>61.41991</td>
      <td>-0.197396</td>
      <td>0.004255</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>8.899308</td>
      <td>0.203190</td>
      <td>0.453964</td>
      <td>929</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>-0.008197</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>8.306454</td>
      <td>0.595944</td>
      <td>0.190703</td>
      <td>1020</td>
      <td>79.37641</td>
      <td>1.190290</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>8.019259</td>
      <td>0.838066</td>
      <td>0.281174</td>
      <td>1050</td>
      <td>79.59367</td>
      <td>1.562532</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>7.786250</td>
      <td>1.081203</td>
      <td>0.514782</td>
      <td>1082</td>
      <td>80.53983</td>
      <td>-0.188572</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>5.801278</td>
      <td>3.175267</td>
      <td>0.949701</td>
      <td>1234</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>5.465077</td>
      <td>3.514135</td>
      <td>1.236325</td>
      <td>1240</td>
      <td>76.33635</td>
      <td>0.175940</td>
      <td>0.033473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>5.379050</td>
      <td>3.635879</td>
      <td>1.077322</td>
      <td>1229</td>
      <td>59.50669</td>
      <td>-0.558599</td>
      <td>0.025532</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>5.278350</td>
      <td>3.676023</td>
      <td>1.267916</td>
      <td>1239</td>
      <td>76.98981</td>
      <td>-0.440396</td>
      <td>0.033333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>5.194040</td>
      <td>3.664602</td>
      <td>1.755296</td>
      <td>1302</td>
      <td>77.35306</td>
      <td>-0.813962</td>
      <td>0.033473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>5.381736</td>
      <td>3.800879</td>
      <td>2.837396</td>
      <td>1334</td>
      <td>79.37197</td>
      <td>0.084544</td>
      <td>0.033755</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>4.819228</td>
      <td>4.366581</td>
      <td>3.095202</td>
      <td>1369</td>
      <td>77.77071</td>
      <td>-1.212026</td>
      <td>0.028571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 26 columns</p>
</div>




```python
mtp_complete = mtp_ordered.fillna(0)
mtp_complete
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Lithology_Gneissic_Complex</th>
      <th>Lithology_Granite_Pluton</th>
      <th>Lithology_Shillong_Group</th>
      <th>...</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>9.281485</td>
      <td>0.828798</td>
      <td>0.183555</td>
      <td>952</td>
      <td>61.41991</td>
      <td>-0.197396</td>
      <td>0.004255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>8.899308</td>
      <td>0.203190</td>
      <td>0.453964</td>
      <td>929</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>-0.008197</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>8.306454</td>
      <td>0.595944</td>
      <td>0.190703</td>
      <td>1020</td>
      <td>79.37641</td>
      <td>1.190290</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>8.019259</td>
      <td>0.838066</td>
      <td>0.281174</td>
      <td>1050</td>
      <td>79.59367</td>
      <td>1.562532</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>7.786250</td>
      <td>1.081203</td>
      <td>0.514782</td>
      <td>1082</td>
      <td>80.53983</td>
      <td>-0.188572</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>5.801278</td>
      <td>3.175267</td>
      <td>0.949701</td>
      <td>1234</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>5.465077</td>
      <td>3.514135</td>
      <td>1.236325</td>
      <td>1240</td>
      <td>76.33635</td>
      <td>0.175940</td>
      <td>0.033473</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>5.379050</td>
      <td>3.635879</td>
      <td>1.077322</td>
      <td>1229</td>
      <td>59.50669</td>
      <td>-0.558599</td>
      <td>0.025532</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>5.278350</td>
      <td>3.676023</td>
      <td>1.267916</td>
      <td>1239</td>
      <td>76.98981</td>
      <td>-0.440396</td>
      <td>0.033333</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>5.194040</td>
      <td>3.664602</td>
      <td>1.755296</td>
      <td>1302</td>
      <td>77.35306</td>
      <td>-0.813962</td>
      <td>0.033473</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>5.381736</td>
      <td>3.800879</td>
      <td>2.837396</td>
      <td>1334</td>
      <td>79.37197</td>
      <td>0.084544</td>
      <td>0.033755</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>4.819228</td>
      <td>4.366581</td>
      <td>3.095202</td>
      <td>1369</td>
      <td>77.77071</td>
      <td>-1.212026</td>
      <td>0.028571</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 26 columns</p>
</div>




```python
mtp_complete.dtypes
```




    Distance from Shillong         float64
    Distance from Nearest River    float64
    Distance from Nearest Fault    float64
    Elevation                        int64
    Slope                          float64
    Aspect                         float64
    VARI                           float64
    Lithology_Gneissic_Complex       int64
    Lithology_Granite_Pluton         int64
    Lithology_Shillong_Group         uint8
    Land_Cover_1                     int64
    Land_Cover_2                     int64
    Land_Cover_9                     uint8
    Land_Cover_11                    uint8
    Land_Cover_12                    uint8
    Land_Cover_13                    int64
    Land_Cover_22                    int64
    Land_Cover_20                    int64
    Watershed_1                      uint8
    Watershed_2                      uint8
    Watershed_3                      uint8
    Watershed_4                      uint8
    Watershed_5                      int64
    Watershed_6                      uint8
    Watershed_7                      uint8
    Watershed_8                      uint8
    dtype: object




```python
mtp_complete.Lithology_Gneissic_Complex = mtp_complete.Lithology_Gneissic_Complex.astype(np.uint8)
mtp_complete.Lithology_Granite_Pluton = mtp_complete.Lithology_Granite_Pluton.astype(np.uint8)
mtp_complete.Land_Cover_1 = mtp_complete.Land_Cover_1.astype(np.uint8)
mtp_complete.Land_Cover_2 = mtp_complete.Land_Cover_2.astype(np.uint8)
mtp_complete.Land_Cover_13 = mtp_complete.Land_Cover_13.astype(np.uint8)
mtp_complete.Land_Cover_22 = mtp_complete.Land_Cover_22.astype(np.uint8)
mtp_complete.Land_Cover_20 = mtp_complete.Land_Cover_20.astype(np.uint8)
```


```python
mtp_complete
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Lithology_Gneissic_Complex</th>
      <th>Lithology_Granite_Pluton</th>
      <th>Lithology_Shillong_Group</th>
      <th>...</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>9.281485</td>
      <td>0.828798</td>
      <td>0.183555</td>
      <td>952</td>
      <td>61.41991</td>
      <td>-0.197396</td>
      <td>0.004255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>8.899308</td>
      <td>0.203190</td>
      <td>0.453964</td>
      <td>929</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>-0.008197</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>8.306454</td>
      <td>0.595944</td>
      <td>0.190703</td>
      <td>1020</td>
      <td>79.37641</td>
      <td>1.190290</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>8.019259</td>
      <td>0.838066</td>
      <td>0.281174</td>
      <td>1050</td>
      <td>79.59367</td>
      <td>1.562532</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>7.786250</td>
      <td>1.081203</td>
      <td>0.514782</td>
      <td>1082</td>
      <td>80.53983</td>
      <td>-0.188572</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>5.801278</td>
      <td>3.175267</td>
      <td>0.949701</td>
      <td>1234</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>5.465077</td>
      <td>3.514135</td>
      <td>1.236325</td>
      <td>1240</td>
      <td>76.33635</td>
      <td>0.175940</td>
      <td>0.033473</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>5.379050</td>
      <td>3.635879</td>
      <td>1.077322</td>
      <td>1229</td>
      <td>59.50669</td>
      <td>-0.558599</td>
      <td>0.025532</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>5.278350</td>
      <td>3.676023</td>
      <td>1.267916</td>
      <td>1239</td>
      <td>76.98981</td>
      <td>-0.440396</td>
      <td>0.033333</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>5.194040</td>
      <td>3.664602</td>
      <td>1.755296</td>
      <td>1302</td>
      <td>77.35306</td>
      <td>-0.813962</td>
      <td>0.033473</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>5.381736</td>
      <td>3.800879</td>
      <td>2.837396</td>
      <td>1334</td>
      <td>79.37197</td>
      <td>0.084544</td>
      <td>0.033755</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>4.819228</td>
      <td>4.366581</td>
      <td>3.095202</td>
      <td>1369</td>
      <td>77.77071</td>
      <td>-1.212026</td>
      <td>0.028571</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 25 columns</p>
</div>




```python
mt_drop = mtp_complete.drop(["Lithology_Gneissic_Complex","Lithology_Granite_Pluton","Lithology_Shillong_Group", "Land_Cover_1", "Land_Cover_2", "Land_Cover_9", "Land_Cover_11", "Land_Cover_12", "Land_Cover_13", "Land_Cover_22", "Land_Cover_20", "Watershed_1", "Watershed_2", "Watershed_3", "Watershed_4", "Watershed_5", "Watershed_6", "Watershed_7", "Watershed_8"],axis=1)
mt_drop.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>9.281485</td>
      <td>0.828798</td>
      <td>0.183555</td>
      <td>952</td>
      <td>61.41991</td>
      <td>-0.197396</td>
      <td>0.004255</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>8.899308</td>
      <td>0.203190</td>
      <td>0.453964</td>
      <td>929</td>
      <td>40.56959</td>
      <td>-1.518213</td>
      <td>-0.008197</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>8.306454</td>
      <td>0.595944</td>
      <td>0.190703</td>
      <td>1020</td>
      <td>79.37641</td>
      <td>1.190290</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>8.019259</td>
      <td>0.838066</td>
      <td>0.281174</td>
      <td>1050</td>
      <td>79.59367</td>
      <td>1.562532</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>7.786250</td>
      <td>1.081203</td>
      <td>0.514782</td>
      <td>1082</td>
      <td>80.53983</td>
      <td>-0.188572</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>5.801278</td>
      <td>3.175267</td>
      <td>0.949701</td>
      <td>1234</td>
      <td>32.47252</td>
      <td>1.428899</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>5.465077</td>
      <td>3.514135</td>
      <td>1.236325</td>
      <td>1240</td>
      <td>76.33635</td>
      <td>0.175940</td>
      <td>0.033473</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>5.379050</td>
      <td>3.635879</td>
      <td>1.077322</td>
      <td>1229</td>
      <td>59.50669</td>
      <td>-0.558599</td>
      <td>0.025532</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>5.278350</td>
      <td>3.676023</td>
      <td>1.267916</td>
      <td>1239</td>
      <td>76.98981</td>
      <td>-0.440396</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>5.194040</td>
      <td>3.664602</td>
      <td>1.755296</td>
      <td>1302</td>
      <td>77.35306</td>
      <td>-0.813962</td>
      <td>0.033473</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>5.381736</td>
      <td>3.800879</td>
      <td>2.837396</td>
      <td>1334</td>
      <td>79.37197</td>
      <td>0.084544</td>
      <td>0.033755</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>4.819228</td>
      <td>4.366581</td>
      <td>3.095202</td>
      <td>1369</td>
      <td>77.77071</td>
      <td>-1.212026</td>
      <td>0.028571</td>
    </tr>
  </tbody>
</table>
</div>




```python
scaler = StandardScaler()
mt_drop_scaled_array = scaler.fit_transform(mt_drop)
mt_drop_scaled_array
```




    array([[ 1.65795092, -1.07718831, -1.03402507, -1.47244795, -0.44325736,
            -0.16427172, -0.67753436],
           [ 1.41859156, -1.49328252, -0.74578413, -1.63144468, -1.75945234,
            -1.54788879, -1.43798245],
           [ 1.04728401, -1.23206069, -1.02640624, -1.00237067,  0.69026265,
             1.28939275, -0.93740738],
           [ 0.86741177, -1.07102428, -0.92996897, -0.79498364,  0.70397738,
             1.67933361, -0.93740738],
           [ 0.72147707, -0.90931351, -0.68095573, -0.5737708 ,  0.76370458,
            -0.15502887, -0.93740738],
           [-0.52172205,  0.48345419, -0.21735656,  0.47699018, -2.27058708,
             1.53934703, -0.93740738],
           [-0.73228636,  0.70883595,  0.08816833,  0.51846759,  0.49835617,
             0.22681421,  1.10678205],
           [-0.78616554,  0.78980848, -0.08131993,  0.44242568, -0.56403108,
            -0.54264924,  0.62183073],
           [-0.84923431,  0.81650821,  0.12184211,  0.51155469,  0.53960641,
            -0.41882549,  1.09826459],
           [-0.90203836,  0.80891208,  0.64136178,  0.94706746,  0.56253689,
            -0.81015295,  1.10678205],
           [-0.7844834 ,  0.89955007,  1.79481902,  1.1682803 ,  0.68998238,
             0.13107298,  1.12403259],
           [-1.13678531,  1.27580033,  2.06962538,  1.41023184,  0.58890141,
            -1.22714352,  0.80745431]])




```python
mt_drop_scaled_df = pd.DataFrame(mt_drop_scaled_array, columns = ["Distance from Shillong","Distance from Nearest River","Distance from Nearest Fault","Elevation","Slope","Aspect", "VARI"])
type(mt_drop_scaled_df)
```




    pandas.core.frame.DataFrame




```python
mt_add = pd.DataFrame(mtp_complete[['Lithology_Gneissic_Complex','Lithology_Granite_Pluton','Lithology_Shillong_Group','Land_Cover_1', 'Land_Cover_2', 'Land_Cover_9', 'Land_Cover_11','Land_Cover_12', 'Land_Cover_13','Land_Cover_22', 'Land_Cover_20', 'Watershed_1', 'Watershed_2', 'Watershed_3', 'Watershed_4', 'Watershed_5', 'Watershed_6', 'Watershed_7', 'Watershed_8']].copy())
mt_add.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lithology_Gneissic_Complex</th>
      <th>Lithology_Granite_Pluton</th>
      <th>Lithology_Shillong_Group</th>
      <th>Land_Cover_1</th>
      <th>Land_Cover_2</th>
      <th>Land_Cover_9</th>
      <th>Land_Cover_11</th>
      <th>Land_Cover_12</th>
      <th>Land_Cover_13</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
mt_add.to_csv('mt_add_for_standardise.csv')
mt_drop_scaled_df.to_csv('mt_drop_scaled_df.csv')
```


```python
mt_scaled = pd.read_csv('mt_full_to_use_for_standardise.csv')
mt_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance from Shillong</th>
      <th>Distance from Nearest River</th>
      <th>Distance from Nearest Fault</th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Aspect</th>
      <th>VARI</th>
      <th>Lithology_Gneissic_Complex</th>
      <th>Lithology_Granite_Pluton</th>
      <th>Lithology_Shillong_Group</th>
      <th>...</th>
      <th>Land_Cover_22</th>
      <th>Land_Cover_20</th>
      <th>Watershed_1</th>
      <th>Watershed_2</th>
      <th>Watershed_3</th>
      <th>Watershed_4</th>
      <th>Watershed_5</th>
      <th>Watershed_6</th>
      <th>Watershed_7</th>
      <th>Watershed_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.657951</td>
      <td>-1.077188</td>
      <td>-1.034025</td>
      <td>-1.472448</td>
      <td>-0.443257</td>
      <td>-0.164272</td>
      <td>-0.677534</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.418592</td>
      <td>-1.493283</td>
      <td>-0.745784</td>
      <td>-1.631445</td>
      <td>-1.759452</td>
      <td>-1.547889</td>
      <td>-1.437982</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.047284</td>
      <td>-1.232061</td>
      <td>-1.026406</td>
      <td>-1.002371</td>
      <td>0.690263</td>
      <td>1.289393</td>
      <td>-0.937407</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.867412</td>
      <td>-1.071024</td>
      <td>-0.929969</td>
      <td>-0.794984</td>
      <td>0.703977</td>
      <td>1.679334</td>
      <td>-0.937407</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.721477</td>
      <td>-0.909314</td>
      <td>-0.680956</td>
      <td>-0.573771</td>
      <td>0.763705</td>
      <td>-0.155029</td>
      <td>-0.937407</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
mt_pred = model_r_imp.predict(mt_scaled)
print(mt_pred)
```

    [0 0 1 1 1 0 0 0 0 0 0 0]
    


```python
lp = {'Sr.No.': ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12'],
        'Landslide': ['No','No','Yes','No','No','No','No','No','No','No','Yes','No']}

# create a DataFrame from the dictionary
Landslide_Pred = pd.DataFrame(lp)
Landslide_Prediction = Landslide_Pred.set_index('Sr.No.')
Landslide_Prediction
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Landslide</th>
    </tr>
    <tr>
      <th>Sr.No.</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



# Bagging


```python
from sklearn.ensemble import BaggingClassifier
```


```python
model_w_bagging = BaggingClassifier(base_estimator = model_r_imp, n_estimators = 100, n_jobs = -1, random_state = 10)
```


```python
model_w_bagging.fit(x_train, y_train)
```




    BaggingClassifier(base_estimator=SVC(C=5, gamma=0.1, random_state=10),
                      n_estimators=100, n_jobs=-1, random_state=10)




```python
predictions_w_bagging = model_w_bagging.predict(x_test)
```


```python
print(confusion_matrix(y_test, predictions_w_bagging))
```

    [[22  2]
     [ 4 16]]
    


```python
print(classification_report(y_test, predictions_w_bagging))
#Precision - 80% of the predicted landslides were actual landslides
#Recall - 80% of the actual landslides were predicted correctly
#F1 score is 0.80 - the model can be improved further (close to 1 good model)
#Support - 24 in the dataset were not landslides and 20 were lanslides
```

                  precision    recall  f1-score   support
    
               0       0.85      0.92      0.88        24
               1       0.89      0.80      0.84        20
    
        accuracy                           0.86        44
       macro avg       0.87      0.86      0.86        44
    weighted avg       0.87      0.86      0.86        44
    
    


```python
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
```


```python
# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, random_state=10)
# evaluate the model and collect the scores
n_scores = cross_val_score(model_w_bagging, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance, training accuracy
print('Train Accuracy:\t', (mean(n_scores)))
print('Train SD:\t', (std(n_scores)))
#test accuracy
acc = accuracy_score(y_true=y_test, y_pred=model_w_bagging.predict(x_test))
print("Test accuracy:\t", acc)

```

    Train Accuracy:	 0.7505228758169933
    Train SD:	 0.08990102826058342
    Test accuracy:	 0.8636363636363636
    


```python
mt_pred_w_bagging = model_w_bagging.predict(mt_scaled)
print(mt_pred_w_bagging)
```

    [0 0 1 1 0 0 0 0 0 0 0 0]
    


```python
lp_wb = {'Sr.No.': ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12'],
        'Landslide': ['No','No','Yes','Yes','No','No','No','No','No','No','No','No']}

# create a DataFrame from the dictionary
Landslide_Pred_wb = pd.DataFrame(lp_wb)
Landslide_Prediction_wb = Landslide_Pred_wb.set_index('Sr.No.')
Landslide_Prediction_wb
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Landslide</th>
    </tr>
    <tr>
      <th>Sr.No.</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T1</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T3</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>Yes</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T11</th>
      <td>No</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


