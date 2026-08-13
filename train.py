# %% 
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100)

# %%
# TÉCNICA UTILIZADA PARA GUIAR: SEMMA
# S: SAMPLE
# E: EXPLORE
# M: MODIFY
# M: MODEL
# A: ASSES

# %%
df = pd.read_csv("data/abt_churn.csv")
df.head()

# %%

df["dtRef"].value_counts().sort_index()

# %%

# => OUT OF TIME: select the most actual safra of the data to test it to see 
# if it will work in the "future"

oot = df[(df["dtRef"] == df["dtRef"].max())].copy()
print(oot.shape)

oot.head()

# %%

df_train = df[(df["dtRef"] < df["dtRef"].max())].copy()
print(df_train.shape)
df_train.head(2)

# %%

# Variáveis das features:
features = df_train.columns[2:-1] # id_usuario até 1 antes da variavel alvo

# Variável target:
target = df_train.columns[-1] # id_usuario até 1 antes da variavel alvo

X, y = df_train[features], df_train[target]

# %%

# 1: SAMPLE

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                                                X, y,
                                                                random_state=42,
                                                                test_size=0.2
                                                                )

# %%

# Verificar se as amostras são "parecidas"

print("Taxa variável resposta [Treino]:", y_train.mean().round(4))
print("Taxa variável resposta [Teste]:", y_test.mean().round(4))

# %%

# Vamos estratificar: garantir que a mesma quantidade de resposta (0/1) proporcional
# em ambos os datasets Treino/Teste

X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                                                X, y,
                                                                random_state = 42,
                                                                test_size = 0.2,
                                                                stratify = y
                                                                )

print("Taxa variável resposta [Treino]:", y_train.mean().round(4))
print("Taxa variável resposta [Teste]:", y_test.mean().round(4))

# %%

# 2: EXPLORE (EDA -> Explore Data Analysis)

X_train.isna().sum().sort_values(ascending=False)

# %%

df_analise = X_train.copy()
df_analise["target"] = y_train
df_analise.head(2)

# %%

sumario = df_analise.groupby(by="target").agg(["mean", "median"]).T
sumario

# %%

sumario["diff_abs"] = sumario[0] - sumario[1]
sumario["diff_rel"] = sumario[0] / sumario[1]
sumario.round(3).sort_values(by=["diff_rel"], ascending=False)

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state = 42, max_depth = 5)
arvore.fit(X_train, y_train)

# %%

plt.figure(dpi=800)

tree.plot_tree(
    arvore,
    feature_names=X_train.columns,
    filled=True,
    class_names=[str(i) for i in arvore.classes_]
)

plt.show()

# %%

# Vamos remover o max_depth para pegar todas as features (mas sem plot)
arvore = tree.DecisionTreeClassifier(random_state = 42)
arvore.fit(X_train, y_train)

# %%

feature_importance = (
                    pd.Series(arvore.feature_importances_, index=X_train.columns)
                    .sort_values(ascending=False)
                    .reset_index()
                    )

feature_importance["acum."] = feature_importance[0].cumsum()
feature_importance

# Pegar talvez o que vai até 95%?
# Pegar apenas quem contribui pelo menos 1%?
# Ou a combinação dos dois?

# %%
feature_importance[feature_importance[0] > 0.01]

# %%
feature_importance[feature_importance["acum."] <= 0.95]

# %%
feature_importance[(feature_importance[0] > 0.01) & (feature_importance["acum."] <= 0.95)]