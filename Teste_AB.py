# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Carregando os dados
df = pd.read_csv('dataset.csv')
# Shape
df.shape
# Amostra
df.head()

# Calculando a taxa de conversão por grupo
taxa_conversao = df.groupby('grupo')['conversao'].mean()

# Exibindo taxas de conversão
print("Taxa de Conversão por Grupo:")
print(taxa_conversao)

# Separação dos grupos
grupo_A = df[df['grupo'] == 'A']
grupo_B = df[df['grupo'] == 'B']

# Cálculo das taxas médias de conversão
taxa_conversao_A = grupo_A['conversao'].mean()
taxa_conversao_B = grupo_B['conversao'].mean()
taxa_conversao_A
taxa_conversao_B

# Teste de Shapiro-Wilk para normalidade
shapiro_A = stats.shapiro(grupo_A['conversao'])
shapiro_B = stats.shapiro(grupo_B['conversao'])
print(f'Teste de Shapiro-Wilk Grupo A: {shapiro_A}')
print(f'Teste de Shapiro-Wilk Grupo B: {shapiro_B}')

# Verificação da homogeneidade das variâncias
levene_test = stats.levene(grupo_A['conversao'], grupo_B['conversao'])
print(f'Teste de Levene para homogeneidade das variâncias:\n{levene_test}')

# Teste t de Student para comparar as médias de conversão
t_stat, p_val = stats.ttest_ind(grupo_A['conversao'], grupo_B['conversao'])

print(f'Taxa de conversão do Grupo A: {grupo_A["conversao"].mean()}')
print(f'Taxa de conversão do Grupo B: {grupo_B["conversao"].mean()}')
print(f'Estatística t: {t_stat}')
print(f'Valor p: {p_val}')

# Teste de Mann-Whitney U
u_stat, p_val_mw = stats.mannwhitneyu(grupo_A['conversao'], grupo_B['conversao'])

print(f'Estatística U: {u_stat}')
print(f'Valor p (Mann-Whitney): {p_val_mw}')

# Interpretação dos resultados
alpha = 0.05
if p_val_mw < alpha:
    print("Rejeitamos a hipótese nula. Diferença estatisticamente significativa entre os grupos A e B")
else:
    print("Não rejeitamos a hipótese nula. Não há diferença estatisticamente significativa entre os grupos A e B")

# Plot
plt.scatter(grupo_A['longitude'], grupo_A['latitude'], c='blue', label='Grupo A')
plt.scatter(grupo_B['longitude'], grupo_B['latitude'], c='red', label='Grupo B')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Distribuição Geográfica dos Grupos A e B')
plt.show()

# ## Conclusão
# Com base nos dados, a utilização de testes paramétircos não é viável, pois os dados não estão normalmente distribuidos, apesar de haver homegeneidade das variâncias, apresentados nos resultados dos testes de Shapiro-Wilk e Levene respectivamente.
# Foi utilizado, então, o teste não paramétrico de Mann-Whitney U, pois as suposições foram atendidas.
# Com base na análise dos dados, não temos evidências estatísticas para afirmar que a campanha de Marketing apresenta diferença entre as regiões geográficas dos usuários. 
# A região geográfica não tem influência na taxa média de conversão nos dados analisados.
