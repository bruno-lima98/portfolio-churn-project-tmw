# %% 
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 200)

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
# %%

# 3. MODIFY

# -> Padronização: normalização, padronização min-max, etc
# -> Imputação de missings: treinar um modelo sem a variavel missing para decidir
# -> Binning: dividir uma variável contínua em faixas
# -> OneHot Encoding: criar colunas baseada nas possibilidades de categoria 
# -> Mean Encoder: calcular % da variavel resposta para cada categoria e criar uma
# coluna com essa informação no lugar
# -> Agrupar Categorias: pode agrupar as categorias, para ter menos, pode ser manual
# ou até mesmo algum algorítimo de clustering
# -> Tranformação de logaritimo, tranformação de exponecial: pesquisar sobre
# -> Combinação de variáveis

best_features = feature_importance[feature_importance["acum."] <= 0.95]["index"].to_list()
best_features

# %%
from feature_engine import discretisation, encoding
from sklearn import pipeline

# V1: Utilizando um passo a passo mais bruto

# Discretizar com binning
# tree_discretization = discretisation.DecisionTreeDiscretiser(
#                                                         variables = best_features,
#                                                         regression = False,
#                                                         bin_output = "bin_number",
#                                                         cv = 3
#                                                         )

# tree_discretization.fit(X_train[best_features], y_train)
# X_train_transform = tree_discretization.transform(X_train[best_features])

# # OneHot
# onehot = encoding.OneHotEncoder(
#                             variables = best_features,
#                             ignore_format = True
#                             )
# onehot.fit(X_train_transform, y_train)

# X_train_transform = onehot.transform(X_train_transform)
# X_train_transform

# from sklearn import linear_model

# reg = linear_model.LogisticRegression(
#                                     penalty = None,
#                                     random_state = 42,
#                                     max_iter = 1000000)

# reg.fit(X_train_transform, y_train)

# V2: Utilizando um pipeline de transformação

tree_discretization = discretisation.DecisionTreeDiscretiser(
                                                        variables = best_features,
                                                        regression = False,
                                                        bin_output = "bin_number",
                                                        cv = 3
                                                        )

# OneHot
onehot = encoding.OneHotEncoder(
                            variables = best_features,
                            ignore_format = True
                            )

# %%

# 4. MODEL

from sklearn import linear_model

reg = linear_model.LogisticRegression(
                                    penalty = None,
                                    random_state = 42,
                                    max_iter = 1000000)

model_pipeline = pipeline.Pipeline(
    steps = [
        ("Discretizar", tree_discretization),
        ("OneHot", onehot),
        ("Model", reg)
    ]
)

model_pipeline.fit(X_train, y_train)

# %%

# 5. ASSES

from sklearn import metrics

# y_train_predict = reg.predict(X_train_transform)
# y_train_proba = reg.predict_proba(X_train_transform)[:,1]

# acc_train = metrics.accuracy_score(y_train, y_train_predict)
# auc_train = metrics.roc_auc_score(y_train, y_train_proba)

# print("Acurácia [Treino] =", round(acc_train,4))
# print("AUC [Treino] =", round(auc_train,4))

# # %%

# X_test_transform = tree_discretization.transform(X_test[best_features])
# X_test_transform = onehot.transform(X_test_transform)

# y_test_predict = reg.predict(X_test_transform)
# y_test_proba = reg.predict_proba(X_test_transform)[:,1]

# acc_test = metrics.accuracy_score(y_test, y_test_predict)
# auc_test = metrics.roc_auc_score(y_test, y_test_proba)

# print("Acurácia [Teste] =", round(acc_test,4))
# print("AUC [Teste] =", round(auc_test,4))

# X_oot_transform = tree_discretization.transform(oot[best_features])
# X_oot_transform = onehot.transform(X_oot_transform)

# y_oot_predict = reg.predict(X_oot_transform)
# y_oot_proba = reg.predict_proba(X_oot_transform)[:,1]

# acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
# auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)

# print("Acurácia [OOT] =", round(acc_oot,4))
# print("AUC [OOT] =", round(auc_oot,4))

# V2:

y_train_predict = model_pipeline.predict(X_train)
y_train_proba = model_pipeline.predict_proba(X_train)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)

print("Acurácia [Treino] =", round(acc_train,4))
print("AUC [Treino] =", round(auc_train,4))

# %%

y_test_predict = model_pipeline.predict(X_test)
y_test_proba = model_pipeline.predict_proba(X_test)[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print("Acurácia [Teste] =", round(acc_test,4))
print("AUC [Teste] =", round(auc_test,4))

# %%
y_oot_predict = model_pipeline.predict(oot[features])
y_oot_proba = model_pipeline.predict_proba(oot[features])[:,1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)

print("Acurácia [OOT] =", round(acc_oot,4))
print("AUC [OOT] =", round(auc_oot,4))

# %%
