# Fast-and-Frugal-Transfer-Learning
This repository provides a **lightweight and energy-aware transfer learning strategy** designed for environments with limited hardware resources. Unlike traditional fine-tuning, the method **decouples feature extraction from classifier training**, avoiding unnecessary iterations through the backbone.

## Key Features 
- **Precomputed Features:** Extract features once from the backbone to drastically reduce training time.  
- **Thresholded Batch Normalization Adaptation:** Efficiently adapts batch normalization statistics to new domains.  
- **Redesigned Classifier Head:** Improves generalization with negligible overhead.  
- **Margin-based Weighted Training:** Focuses on ambiguous samples to improve accuracy.  

Results on **Brain Cancer MRI**, **BreakHis** and **PatchCamelyon** datasets show up to **22× faster training** and **25× lower CO₂ emissions**, while maintaining or improving accuracy compared to full fine-tuning.

---

## Setup
1) Clone the repository
    ```bash
    git clone https://github.com/DanielVilaCruz/Fast-and-Frugal-Transfer-Learning.git
    cd Fast-and-Frugal-Transfer-Learning
    ```
2) Install PyTorch

   This project requires PyTorch (with CUDA or CPU support).

   Example (CUDA 12.1):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Example (CPU):
   ```bash
   pip install torch torchvision torchaudio
   ```
   
   
4) Install project dependencies
    ```bash
    pip install -r requirements.txt
    ```
4) Download datasets and place them under the /data folder.
- Brain Cancer MRI: https://data.mendeley.com/datasets/mk56jw9rns/1
- BreakHis: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- PatchCamyleon:https://github.com/basveeling/pcam

  

## Usage

Experiments are configured in **`experiments.yaml`**. Each entry corresponds to a single experiment, and multiple ones can be added to execute them sequentially.

```yaml
experiments:
  - method: "ours"
    base_backbone: "resnet18"
    dataset: "brain_cancer"
    n_real_samples: 0
    num_runs: 3
    lr: 0.001
    amplify_factor: 10.0
    patience: 5
    unfreeze_layers: 2
    simple_head: false

  - method: "fine-tune"
    base_backbone: "densenet121"
    dataset: "breakhis"
    n_real_samples: 1000
    num_runs: 1
```

### Parameters
- **`method`**: training mode  
  - `"ours"`: precomputed features + lightweight classifier  
  - `"fine-tune"`: standard fine-tuning  
- **`base_backbone`**: model architecture  
  - Options: `"resnet18"`, `"resnet50"`, `"mobilenet_v3_large"`, `"densenet121"`  
- **`dataset`**: dataset to use  
  - Options: `"brain_cancer"`, `"breakhis"`, `"pcam"`
- **`n_real_samples`**: maximum number of samples to use  
  - `0`: use all available samples  
- **`num_runs`**: number of times to repeat the experiment (averages results)  
- **`lr`**: learning rate  
- **`amplify_factor`**: weight assigned to ambiguous samples in margin-based training  
- **`patience`**: early stopping patience 
- **`unfreeze_layers`** *(only for fine-tuning method)*: number of last layers to unfreeze in the backbone  
- **`simple_head`**: whether to use a minimal classifier head (`true`) or the improved one (`false`)  

Then launch experiments with:
```bash
python launch_exp.py
```

By default, results (metrics, timings, CO₂ emissions) are stored in results.json and exps/.

---

## Results

### Brain Cancer MRI Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Architecture</th>
      <th colspan="3" style="background-color:#fdd;">Fine-tuning</th>
      <th colspan="3" style="background-color:#dfd;">Ours</th>
    </tr>
    <tr>
      <th>Acc. (%)</th><th>Time (s)</th><th>CO₂ (kg)</th>
      <th>Acc. (%)</th><th>Time (s)</th><th>CO₂ (kg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet18</td><td>99.16</td><td>194.672</td><td>0.0038</td>
      <td>98.42</td><td>32.486</td><td>0.0006</td>
    </tr>
    <tr>
      <td>ResNet50</td><td>99.25</td><td>311.550</td><td>0.0064</td>
      <td>97.75</td><td>57.407</td><td>0.0011</td>
    </tr>
    <tr>
      <td>MobileNetV3</td><td>93.93</td><td>770.210</td><td>0.0151</td>
      <td>99.71</td><td>34.780</td><td>0.0006</td>
    </tr>
    <tr>
      <td>DenseNet121</td><td>99.58</td><td>625.140</td><td>0.0127</td>
      <td>96.76</td><td>58.204</td><td>0.0011</td>
    </tr>
  </tbody>
</table>

### BreakHis Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Architecture</th>
      <th colspan="3" style="background-color:#fdd;">Fine-tuning</th>
      <th colspan="3" style="background-color:#dfd;">Ours</th>
    </tr>
    <tr>
      <th>Acc. (%)</th><th>Time (s)</th><th>CO₂ (kg)</th>
      <th>Acc. (%)</th><th>Time (s)</th><th>CO₂ (kg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet18</td><td>99.33</td><td>367.299</td><td>0.0069</td>
      <td>98.50</td><td>42.621</td><td>0.0008</td>
    </tr>
    <tr>
      <td>ResNet50</td><td>97.66</td><td>513.383</td><td>0.0104</td>
      <td>98.33</td><td>59.743</td><td>0.0012</td>
    </tr>
    <tr>
      <td>MobileNetV3</td><td>99.00</td><td>430.983</td><td>0.0087</td>
      <td>98.67</td><td>57.147</td><td>0.0010</td>
    </tr>
    <tr>
      <td>DenseNet121</td><td>93.46</td><td>3970.622</td><td>0.0794</td>
      <td>94.52</td><td>184.374</td><td>0.0038</td>
    </tr>
  </tbody>
</table>


### PatchCamelyon (PCam) Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Architecture</th>
      <th colspan="3" style="background-color:#fdd;">Fine-tuning</th>
      <th colspan="3" style="background-color:#dfd;">Ours</th>
    </tr>
    <tr>
      <th>Acc. (%)</th><th>Time (s)</th><th>CO₂ (kg)</th>
      <th>Acc. (%)</th><th>Time (s)</th><th>CO₂ (kg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet18</td>
      <td><b>79.78</b></td><td>5953.25</td><td>0.1202</td>
      <td>79.03</td><td><b>841.32</b></td><td><b>0.0163</b></td>
    </tr>
    <tr>
      <td>ResNet50</td>
      <td><b>83.54</b></td><td>10191.44</td><td>0.2140</td>
      <td>81.17</td><td><b>1114.45</b></td><td><b>0.0229</b></td>
    </tr>
    <tr>
      <td>MobileNetV3</td>
      <td><b>82.96</b></td><td>8009.09</td><td>0.1665</td>
      <td>82.64</td><td><b>695.83</b></td><td><b>0.0134</b></td>
    </tr>
    <tr>
      <td>DenseNet121</td>
      <td><b>84.22</b></td><td>20530.34</td><td>0.4382</td>
      <td>80.67</td><td><b>1287.82</b></td><td><b>0.0264</b></td>
    </tr>
  </tbody>
</table>




  
  
