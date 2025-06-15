import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np # Usado para a raiz quadrada (sqrt)
from sklearn.metrics import mean_squared_error # Usado para calcular o erro

endereco = kagglehub.dataset_download("jeanmidev/smart-meters-in-london")
caminho_do_arquivo = f'{endereco}/daily_dataset.csv'
df = pd.read_csv(caminho_do_arquivo)
df['day'] = pd.to_datetime(df['day'])
df_total = df.groupby('day').agg({'energy_sum': 'sum'})

df_total['Dia da Semana'] = df_total.index.dayofweek
df_total['Trimestre'] = df_total.index.quarter
df_total['Mês'] = df_total.index.month
df_total['Ano'] = df_total.index.year
df_total['Dia do ano'] = df_total.index.dayofyear

train = df_total.loc[df_total.index < '2014-01-01']
test = df_total.loc[df_total.index >= '2014-01-01']
FEATURES = ['Dia da Semana', 'Trimestre', 'Mês', 'Ano', 'Dia do ano']
TARGET = 'energy_sum'

reg = xgb.XGBRegressor(
    n_estimators=850,
    early_stopping_rounds=50,
    learning_rate=0.01
    )
reg.fit(train[FEATURES], train[TARGET],
        eval_set=[(train[FEATURES], train[TARGET]), (test[FEATURES], test[TARGET])],
        verbose=100)

test['Previsão'] = reg.predict(test[FEATURES])
df_total = df_total.merge(test[['Previsão']], how='inner', left_index=True, right_index=True)

grafico = df_total['energy_sum'].plot(figsize=(15, 5))
df_total['Previsão'].plot(ax=grafico, style='.', ms=7) 

plt.legend(['Valor Verdadeiro', 'Previsão do Modelo'])
plt.title('Previsão do Consumo x Valor Real')
plt.savefig('grafico_previsao_vs_real.png')
print("Gráfico final 'grafico_previsao_vs_real.png' foi salvo.")

score = np.sqrt(mean_squared_error(test[TARGET], test['Previsão']))

# Linha 8: Imprime a nota final. Este número nos diz, em média, quantos kWh nosso modelo errou.
print(f'O erro médio da nossa previsão (RMSE) é de: {score:0.2f} kWh')
