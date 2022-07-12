import json
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from PIL                      import Image
from plotly                   import express as px
from folium.plugins           import MarkerCluster
from streamlit_folium         import folium_static
from matplotlib.pyplot        import figimage
from distutils.fancy_getopt   import OptionDummy



st.set_page_config(page_title='Venta de casas',
                    layout="wide", 
                    page_icon='🌏',  
                    initial_sidebar_state="expanded")




st.title('Inmobiliaria en King County')
st.header('Propuesto por Jenny Fuquen ')


# @st.cache
def get_data():
     url = 'https://raw.githubusercontent.com/Jenfuor24/PROY_cien_datos/main/ProyectoPreciosCasas/data/kc_house_data.csv'
     return pd.read_csv(url)
    
data = get_data()
data_ref = data.copy()

st.sidebar.markdown("# Parámetros")
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d').dt.date
data['yr_built']= pd.to_datetime(data['yr_built'], format = '%Y').dt.year
# data['yr_renovated'] = data['yr_renovated'].apply(lambda x: pd.to_datetime(x, format ='%Y') if x >0 else x )
# data['id'] = data['id'].astype(str)

#llenar la columna anterior con new_house para fechas anteriores a 2015-01-01
data['house_age'] = 'NA'
#llenar la columna anterior con new_house para fechas anteriores a 2015-01-01
data.loc[data['yr_built']>1990,'house_age'] = 'new_house' 
#llenar la columna anterior con old_house para fechas anteriores a 2015-01-01
data.loc[data['yr_built']<1990,'house_age'] = 'old_house'

data['zipcode'] = data['zipcode'].astype(str)


data.loc[data['yr_built']>=1990,'house_age'] = 'new_house' 
data.loc[data['yr_built']<1990,'house_age'] = 'old_house'

data.loc[data['bedrooms']<=1, 'dormitory_type'] = 'studio'
data.loc[data['bedrooms']==2, 'dormitory_type'] = 'apartment'
data.loc[data['bedrooms']>2, 'dormitory_type'] = 'house'

data.loc[data['condition']<=2, 'condition_type'] = 'bad'
data.loc[data['condition'].isin([3,4]), 'condition_type'] = 'regular'
data.loc[data['condition']== 5, 'condition_type'] = 'good'

data = get_data()
datta = data.copy()
datta['price/sqft'] = datta['price']/datta['sqft_living']
datta['year_old'] = 2020-datta['yr_built']
datta = datta.drop(columns=['price'])

banhos = st.sidebar.number_input('Número de baños', min_value=0, max_value=6, value=1, step=1)

### habitaciones info
habitaciones = st.sidebar.number_input('Número de habitaciones', min_value=1, max_value=7, value=1, step=1)

### area total construida
area = st.sidebar.number_input('Area total del inmueble', min_value=1, max_value=3500, value=1, step=10)

### area de terreno
area_lote = st.sidebar.number_input('Area del terreno', min_value=1, max_value=5000, value=1, step=1)

### area de la sala
area_sala = st.sidebar.number_input('Area de la sala', min_value=1, max_value=3500, value=1, step=1)

### pisos info
pisos = st.sidebar.number_input('Pisos', min_value=1, max_value=3, value=1, step=1)


### asiganacion de valores del vector
X = pd.DataFrame()

X.loc[0,'bedrooms'] = habitaciones
X.loc[0,'bathrooms'] = banhos
X.loc[0,'sqft_above'] = area
X.loc[0,'sqft_lot'] = area_lote
X.loc[0,'sqft_living'] = area_sala
X.loc[0,'floors'] = pisos



### informacion por pantalla
st.markdown("""
En esta seccion, un modelo de Machine Learning ha sido disponibilizado para generar pronósticos de precios que se basa en las caracteristicas del inmueble. El usuario debe suministrar las características de su interes utilizando el menú de la barra izquierda. A continuación se definen la información que se requiere:
     
- Número de baños: Número de baños de la propiedad a sugerir precio. 
- Número de habitaciones: Número de habitaciones de la propiedad a sugerir precio
- Área del inmueble: Área en pies cuadrados total de la propiedad a sugerir precio. Desde 3000 metros cuadrados
- Área del terreno: Área en pies cuadrados del terreno a sugerir precio. Desde 3000 pies cuadrados
- Area de la sala: Area en pies cuadrados de la sala a sugerir precio. Desde 1000 pies cuadrados
- Número de pisos: Número de pisos de la propiedad a sugerir precio.
    """)

variables = ['bedrooms', 'bathrooms', 'sqft_living','floors',
             'sqft_above', 'sqft_basement'
             ]

params = {'Habitaciones':['bedrooms', habitaciones],
          'Baños':['bathrooms', banhos],
          'Pisos':['floors', pisos],
          'Area de sala': ['sqft_living',area_sala],
          'Area terreno': ['sqft_basement', area_lote],
          'Area': ['sqft_above',area]
         
         }

OptFiltro = st.multiselect(
     'Variables a incluir en los filtros:',
     ['Habitaciones', 'Baños', 'Pisos','Area de sala','Area terreno','Area'],
     ['Baños'])



col1, col2 = st.columns(2)

with col1:

    data_v2 = datta.copy()
    for filtro in OptFiltro:
        (llave, variable) = params[filtro]
        data_v2 = data_v2[data_v2[llave]==variable]

    data_v2['zipcode'] = data_v2['zipcode'].astype(str)

    st.header("Ubicación y detalles de casas disponibles acorde a los requerimientos del cliente.")
    mapa = folium.Map(location=[data_v2['lat'].mean(), data_v2['long'].mean()], zoom_start=9)
    markercluster = MarkerCluster().add_to(mapa)
    for nombre, fila in data_v2.iterrows():
        folium.Marker([fila['lat'],fila['long']],
                         popup = 'Fecha: {} \n {} habitaciones \n {} baños \n construida en {} \n área de {} pies cuadrados \n Precio por pie cuadrado: {}'.format(
                         fila['date'],
                         fila['bedrooms'],
                         fila['bathrooms'],
                         fila['yr_built'], 
                         fila['sqft_living'], 
                         fila['sqft_above'])
          ).add_to(markercluster)
    folium_static(mapa)

### se carga el model xgboost para la estimacion del valor de la casa
### se muestra por panatalla
if st.sidebar.button('Los parámetros han sido cargados. Calcular precio'):

    modelo_final = pickle.load(open('../xbg_final.sav', 'rb'))
    vector = np.array(list(X.loc[0])).reshape(-1, 1).T
    precio = modelo_final.predict(vector)[0]
    st.balloons()
    st.success('El precio ha sido calculado')
    st.metric("Precio Estimado:", f"${np.float(round(precio, 1))}")

    st.header(f"Un total de {data_v2.shape[0]} casas coinciden con las caracteristicas requeridas por el usuario.")
    st.dataframe(data_v2)  

else:
    st.snow()
    st.error('Por favor, seleccione los parámatros de la propiedad a estimar el precio.')
