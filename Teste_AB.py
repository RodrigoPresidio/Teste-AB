#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# In[ ]:


# Carregando os dados
df = pd.read_csv('dataset.csv')


# In[ ]:


# Shape
df.shape


# In[ ]:


# Amostra
df.head()


# In[ ]:


# Calculando a taxa de conversão por grupo
taxa_conversao = df.groupby('grupo')['conversao'].mean()


# In[ ]:


# Exibindo taxas de conversão
print("Taxa de Conversão por Grupo:")
print(taxa_conversao)


# In[ ]:


# Separação dos grupos
grupo_A = df[df['grupo'] == 'A']
grupo_B = df[df['grupo'] == 'B']


# In[ ]:


# Cálculo das taxas médias de conversão
taxa_conversao_A = grupo_A['conversao'].mean()
taxa_conversao_B = grupo_B['conversao'].mean()


# In[ ]:


taxa_conversao_A


# In[ ]:


taxa_conversao_B


# In[ ]:


# Teste de Shapiro-Wilk para normalidade
shapiro_A = stats.shapiro(grupo_A['conversao'])
shapiro_B = stats.shapiro(grupo_B['conversao'])
print(f'Teste de Shapiro-Wilk Grupo A: {shapiro_A}')
print(f'Teste de Shapiro-Wilk Grupo B: {shapiro_B}')


# In[ ]:


# Verificação da homogeneidade das variâncias
levene_test = stats.levene(grupo_A['conversao'], grupo_B['conversao'])
print(f'Teste de Levene para homogeneidade das variâncias:\n{levene_test}')


# In[ ]:


# Teste t de Student para comparar as médias de conversão
t_stat, p_val = stats.ttest_ind(grupo_A['conversao'], grupo_B['conversao'])


# In[ ]:


print(f'Taxa de conversão do Grupo A: {grupo_A["conversao"].mean()}')
print(f'Taxa de conversão do Grupo B: {grupo_B["conversao"].mean()}')
print(f'Estatística t: {t_stat}')
print(f'Valor p: {p_val}')


# In[ ]:


# Teste de Mann-Whitney U
u_stat, p_val_mw = stats.mannwhitneyu(grupo_A['conversao'], grupo_B['conversao'])


# In[ ]:


print(f'Estatística U: {u_stat}')
print(f'Valor p (Mann-Whitney): {p_val_mw}')


# In[ ]:


# Interpretação dos resultados
alpha = 0.05
if p_val_mw < alpha:
    print("Rejeitamos a hipótese nula. Diferença estatisticamente significativa entre os grupos A e B")
else:
    print("Não rejeitamos a hipótese nula. Não há diferença estatisticamente significativa entre os grupos A e B")


# ## Distribuição Geográfica dos Grupos A e B

# In[ ]:


# Plot
plt.scatter(grupo_A['longitude'], grupo_A['latitude'], c='blue', label='Grupo A')
plt.scatter(grupo_B['longitude'], grupo_B['latitude'], c='red', label='Grupo B')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Distribuição Geográfica dos Grupos A e B')
plt.show()


# ## Conclusão
# 
# Com base nos dados, a utilização de testes paramétircos não é viável, pois os dados não estão normalmente distribuidos, apesar de haver homegeneidade das variâncias, apresentados nos resultados dos testes de Shapiro-Wilk e Levene respectivamente.
# 
# Foi utilizado, então, o teste não paramétrico de Mann-Whitney U, pois as suposições foram atendidas.
# 
# Com base na análise dos dados, não temos evidências estatísticas para afirmar que a campanha de Marketing apresenta diferença entre as regiões geográficas dos usuários. 
# 
# A região geográfica não tem influência na taxa média de conversão nos dados analisados.
