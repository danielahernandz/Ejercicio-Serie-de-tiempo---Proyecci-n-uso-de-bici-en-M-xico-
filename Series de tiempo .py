#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid") # estilo de salida de las gráficas
from datetime import datetime


# In[8]:


import os
os.chdir('C:\\Users\\isabe\\OneDrive\Escritorio\python')


# In[10]:


# Leyendo los datos
viajes = pd.read_csv('2018-10.csv')
viajes.head()


# In[11]:


# concatenar Hora_Retiro y Fecha_Retiro
viajes['fecha_hora_retiro'] = viajes.Fecha_Retiro + ' ' + viajes.Hora_Retiro

# cambiar de str a datetime
viajes['fecha_hora'] = viajes.fecha_hora_retiro                              .map(lambda x : datetime.strptime(x, '%d/%m/%Y %H:%M:%S'))
# reindexar el dataframe
viajes.index = viajes.fecha_hora

# limpiar valores de otros años
viajes = viajes.loc['2018-10']
viajes.head()


# In[12]:


# resample y agregacion por dia de mes
viajes_resample_day = viajes.Bici.resample('H').count()

# asignar día de la semana
df_resample = pd.concat([viajes_resample_day], axis=1)
df_resample['dayofweek'] = df_resample.index.dayofweek # 0 es lunes

# lunes a viernes
df_mon_to_fri = df_resample[df_resample.dayofweek.isin([0,1,2,3,4])].Bici
df_mon_to_fri.head()


# In[13]:


df_mon_to_fri[0:(24*5)].plot()
plt.show()

"""Los datos empiezan a contar una historia. Para cada día ocurre un primer pico en en las mañanas, 
posiblemente sea gente que va a trabajar. Hay otro pico al rededor de las 2pm, que suele ser la hora de la comida 
y el descanso. Por último, la mayor cantidad de viajes ocurre a partir de las 6pm, 
cuando ya vamos de regreso a casa y alguno que otro sale a pasear en bici en la noche. 
Este patrón se repite con cierta regularidad cada 24 horas. """


# In[15]:


get_ipython().system('pip3 install statsmodels')
from statsmodels.tsa.statespace.sarimax import SARIMAX
#SARIMAX es esencialmente un modelo de regresión lineal que utiliza un modelo de tipo ARIMA estacional de residuos.


# In[16]:


# definir conjunto de datos
x = df_mon_to_fri

# instanciar modelo
sarima_model = SARIMAX(x, order=(2,0,1), seasonal_order=(2, 1, 0, 24))

# ajustar modelo
results = sarima_model.fit()

# mirar el AIC #El criterio de información de Akaike (AIC) es una medida de la calidad relativa de un modelo estadístico, para un conjunto dado de datos.
results.aic


# In[ ]:


"""Modelos para series de tiempo
Queremos pronosticar la cantidad de viajes por hora del día. Para esto, utilizamos la variable dependiente o target (la cantidad de viajes) en un tiempo anterior como variable independiente
Por ejemplo, la cantidad de viajes que se realiza un día a las 13 horas puede estar relacionada con la cantidad de viajes que se realizaron a las 13 horas de uno o dos días antes.
El tipo de modelos que sigue esta filosofía se llaman AutoRegresivos o AR. Utilizaremos una variante más avanzada llamada SARIMA (Seasonal AutoRegressive Integrated Moving Average). 
Estos modelos consideran además la estacionalidad, lo que permite incorporar el patrón repetitivo de cada día.
"""


# In[17]:


# tomar de datos originales dias 29-oct, 30-oct, y 31-oct
df_29_31 = df_mon_to_fri.loc['2018-10-29':'2018-10-31']
df_29_31.plot()

# agregar bandas de confianza
pred_1_2_conf = results.get_forecast(steps=24*2).conf_int()
pred_1_2_conf.index = pd.date_range(start='11/1/2018', end='11/3/2018', freq='H')[:-1]
x = pd.date_range(start='11/1/2018', end='11/3/2018', freq='H')[:-1]
y1 = pred_1_2_conf['lower Bici']
y2 = pred_1_2_conf['upper Bici']
plt.fill_between(x, y1, y2, alpha=0.6)

# predecir para 1-nov y 2-nov
pred_1_2 = results.get_forecast(steps=24*2).predicted_mean
pred_1_2.index = pd.date_range(start='11/1/2018', end='11/3/2018', freq='H')[:-1]
pred_1_2.plot()

# formato de la grafica final
plt.title('Pronóstico de viajes')
plt.ylabel('Cantidad de viajes')
# plt.xlabel('Semana lun-29-oct al vie-02-nov')
plt.xlabel(' ')
plt.legend(('Datos originales octubre', 'Pronóstico noviembre'),
           loc='lower left')
plt.savefig('pronostico.png', dpi=200)
plt.show()

"""Pronósticos
Para observar los pronósticos generamos una gráfica con la estimación puntual y de intervalos para el resto de la semana. 
Como el 31 de octubre de 2018 fue miércoles, se presentan las estimaciones para los primeros dos días de noviembre, jueves y viernes.
De esta forma se puede apreciar como el modelo captura el patrón del comportamiento de los viajes en bici.
 
"""


# In[ ]:




