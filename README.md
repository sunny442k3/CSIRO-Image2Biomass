# CSIRO - Image2Biomass Prediction

## Introduction
Rank: **34/3906 (Silver Medal)**

**Summary of Method:**
Our solution leverages a strong foundation model (**DINOv3**) combined with **DoRA** (Weight-Decomposed Low-Rank Adaptation) for efficient fine-tuning. To address the unique 2:1 aspect ratio of the input images, we employed a split-and-fuse strategy using **Mamba** (State Space Models) to merge features effectively. Finally, we utilized **Tweedie Loss** to handle the zero-inflated target distribution and a **Balanced R2** metric strategy to ensure uniform performance across all biomass types.

---

## 1. Preprocessing & Augmentation

### Data Normalization
The raw target values vary significantly in range. To ensure training stability and faster convergence, we did not train on the raw mass values. Instead, we normalized the target variables to the range $[0, 1]$ using Max Scaling based on the maximum value of each respective column in the training set:

$$y_{norm}^{(i)} = \frac{y_{raw}^{(i)}}{\max(Y_{col})}$$

### Augmentations
To prevent overfitting and improve generalization on the test set, we applied a robust augmentation pipeline focusing on geometric and color invariance:
* **Geometric Transforms:** We used Resize, Horizontal Flip ($p=0.5$), Vertical Flip ($p=0.5$), Random Rotate 90, and slight Rotations ($\pm 5^\circ$).
* **Color Transforms:** Color Jitter (adjusting Brightness, Contrast, Saturation, and Hue) was applied to simulate different lighting conditions and sensor variations.
* **Test-Time Augmentation (TTA):** During inference, we averaged predictions from the original image and its horizontally flipped version to smooth out noise.

---

## 2. Model Architecture

### Backbone & Input Strategy
We utilized **`facebook/dinov3-vitl16-pretrain-lvd1689m`** as the feature extractor. Since the input images have a 2:1 aspect ratio, resizing them directly to a square would distort the biomass features. We adopted a split strategy:
1.  The image is split into two equal square crops.
2.  Each crop is passed independently through the DINOv3 backbone to obtain feature embeddings.

### Feature Fusion with Mamba
To effectively model the spatial relationship and continuity between the two split crops, we used a **LocalMambaBlock**. Unlike simple concatenation, Mamba (State Space Models) allows for efficient token mixing with linear complexity, effectively "stitching" the information from the two halves together to form a coherent global representation.

### Efficient Fine-tuning: DoRA
Full fine-tuning of the ViT-Large backbone is computationally expensive. We employed **DoRA (Weight-Decomposed Low-Rank Adaptation)**, which decomposes weights into magnitude and direction components, offering better learning capacity than standard LoRA.
* **Rank:** 32
* **Target Modules:** Query ($q$), Key ($k$), Value ($v$), and Output ($o$) projection layers.

### Regression Heads
The model predicts the 3 atomic components (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`) using 3 separate MLP heads.
* **Activation:** We used **SoftPlus** at the final layer to enforce the physical constraint that biomass mass cannot be negative ($y \ge 0$).
    $$f(x) = \ln(1 + e^x)$$

The composite targets (`GDM_g` and `Dry_Total_g`) are calculated post-inference by summing the predicted atomic components.

---

## 3. Objective: Loss Function

One of the critical challenges in this dataset was the distribution of the target variables. They exhibit a **Compound Poisson-Gamma distribution**:
1.  **Zero-inflation:** A significant number of samples have exactly 0g biomass.
2.  **Right-skewness:** Non-zero values are highly skewed with a long tail.

Standard losses like MSE or L1 are unsuitable here because MSE is overly sensitive to outliers, and L1 does not structurally account for the probability mass at zero.

**Selected Loss: Tweedie Loss**
We adopted the Tweedie Loss with power parameter $p \in (1, 2)$. The variance of the Tweedie distribution is given by:
$$\text{Var}(Y) = \phi \mu^p$$
Where $1 < p < 2$ characterizes a compound Poisson-Gamma process. This allows the model to simultaneously learn:
* The probability of biomass being zero (Poisson component).
* The quantity of biomass if positive (Gamma component).

---

## 4. Metric & Balanced Optimization

### The Bias Problem
The competition metric is the mean $R^2$ across targets. However, we observed that simply optimizing for the global mean $R^2$ caused the model to overfit the "easier" targets (typically `Dry_Green`, which has more consistent visual cues) while neglecting the harder ones (like `Dry_Clover`, which is sparse and difficult to detect).

This creates a scenario where the mean score looks good, but the model fails significantly on specific minority classes.

### Balanced R2 Strategy
To counter this, we designed a **Balanced Metric** for model monitoring and checkpoint selection. The core idea is to treat the $R^2$ scores of the three targets as a set and penalize the variance within that set.

Let $S$ be the set of $R^2$ scores for the three individual targets:

$$S = \{R^{2}_{\text{Dry\_Green}}, R^{2}_{\text{Dry\_Dead}}, R^{2}_{\text{Dry\_Clover}}\}$$

We define our custom optimization objective as:

$$\text{Score}_{balanced} = \mu_{S} - \lambda \cdot \sigma_{S}$$

Where:
* $\mu_{S} = \text{Mean}(S)$: The average $R^2$ performance.
* $\sigma_{S} = \text{Std}(S)$: The standard deviation of the scores in set $S$.
* $\lambda = 0.1$: The penalty factor.

**Intuition:**
If the model cheats by maximizing only `Dry_Green` (e.g., $R^2=0.8$) but failing on `Dry_Clover` (e.g., $R^2=0.2$), the standard deviation $\sigma_{S}$ will be large, significantly lowering the final $\text{Score}_{balanced}$. This penalty forces the model to "pull up" the performance of the weakest link (`Dry_Clover`) to reduce variance, ensuring a robust and uniform prediction capability across all biomass types.