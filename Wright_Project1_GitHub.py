# %% [markdown]
# # 1.1 Exercise

# %% [markdown]
# **1.1.2 Part 1**

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
data = np.load('fruits.npz')

# %%
# get input X and y
X = data['X']
y = data['y']

# %%
indices = np.random.choice(X.shape[0], 5, replace=False)

# %%
index_to_fruit = {
1: 'Apple',
2: 'Avocado',
3: 'Banana',
4: 'Cherry',
5: 'Corn',
6: 'Mango',
7: 'Orange',
8: 'Pear',
9: 'Plum',
10: 'Strawberry',
}

# %%
fig, axs = plt.subplots(1, 5, figsize=(12, 3))

for i, idx in enumerate(indices):
    img = X[idx].reshape(100, 100)

    axs[i].imshow(img, cmap="gray")
    axs[i].axis("off")
    name = index_to_fruit[y[idx]]
    axs[i].set_title(f"{name}")
    
plt.tight_layout()
plt.show()

# %% [markdown]
# ------------------------

# %% [markdown]
# **1.1.3 Part 2**

# %%
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# %%
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
scaler = StandardScaler()

# %%
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# %%
model = LinearRegression()

model.fit(X_train_std, y_train)

y_pred = model.predict(X_test_std)

# %%
# 6. Evaluate the model without PCA
score = r2_score(y_test, y_pred)

print("R2 Score:", score)

# %% [markdown]
# -------------------

# %% [markdown]
# **1.1.4 Part 3**

# %%
pca = PCA(n_components=50)

# %%
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# %%
model_pca = LinearRegression()

model_pca.fit(X_train_pca, y_train)

# %%
y_pred_pca = model_pca.predict(X_test_pca)

# %%
score_pca = r2_score(y_test, y_pred_pca)

print('PCA R2 Score:', score_pca)

# %% [markdown]
# **Comparison**
# 
# The baseline accuracy model has an R2 score of around 0.952 and the PCA accuracy model has an R2 score of around 0.548. So, the PCA accuracy is slightly lower than the baseline which means the baseline performed alightly better than the PCA.

# %% [markdown]
# **Did PCA improve or reduce performance?**
# 
# PCA reduced the performance a little because the R2 score went down after PCA.

# %% [markdown]
# **What is the trade-off between dimensionality reduction and accuracy?**
# 
# The trade-off between dimensionality reduction and accuracy is that PCA makes data smaller which, in turn, makes the data process faster. But, it has the potential to lose information, so the model might not perform as well. So, while the PCA helps with speed and efficieny, it also slightly reduces the accuracy.

# %% [markdown]
# -------------------

# %% [markdown]
# **1.1.5 Part 4**

# %%
pca_full = PCA()
pca_full.fit(X_train_std)

# %%
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# %%
k_90 = np.argmax(cum_var >= 0.90) + 1

print('k for 90% variance:', k_90)

# %% [markdown]
# **How many features *k* are required to retain 90% of the score?**
# 
# About 63 principal components are needed to retain 90% of the score.

# %% [markdown]
# **How does this compare to k = 50 used earlier?**
# 
# This is slightly bigger than the $k = 50$ that was used before. It shows that 50 components wasn't enough to keep the majority of information in the data. This can help to explain why the performance decreased.

# %% [markdown]
# **Would you choose this *k* instead? Why or why not?**
# 
# I'd choose $k = 63$ because it keeps more information from the data. This helps to improve the performance of the model. But, since the number of features also increases a little, the model could take a little longer to run. But, overall I think it's better for accuracy and efficienty purposes than $k = 50$.


