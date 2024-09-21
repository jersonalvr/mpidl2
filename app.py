import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos desde archivo CSV local
csv_file = 'resultados_modelo_ventas.csv'
data = pd.read_csv(csv_file)

# Título y subtítulo del dashboard
st.title('Resolución del Caso Propuesto')
st.subheader('Dashboard Interactivo')

# Sección 1: Cargar Datos
st.header('1. Cargar Datos')
st.write('Cargar un archivo CSV que contiene los resultados de un modelo de predicción de ventas en una cadena de tiendas, incluyendo las variables "Precio", "Ventas Predichas", "Categoría de Producto", y "Mes".')

# Mostrar las primeras filas del archivo
st.write("Datos cargados exitosamente:")
st.write(data.head())

# 1. Preparar los datos
data['Mes'] = pd.to_datetime(data['Mes'])  # Convertir la columna 'Mes' a formato de fecha
data['Mes_Num'] = data['Mes'].dt.month     # Extraer el número del mes

# Codificación One-Hot para 'Categoría_Producto'
data_encoded = pd.get_dummies(data, columns=['Categoría_Producto'], drop_first=True)

# Variables independientes (X) y dependiente (y)
X = data_encoded[['Precio', 'Mes_Num'] + [col for col in data_encoded if col.startswith('Categoría_Producto_')]]
y = data_encoded['Ventas_Predichas']

# 2. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados
st.header('2. Resultados del Modelo Predictivo')
st.write(f'Error Cuadrático Medio (MSE): {mse}')
st.write(f'Coeficiente de Determinación (R^2): {r2}')

# Título de la visualización
st.header('3. Visualización de Dispersión (Precio vs Ventas Predichas)')

# Filtrar por categoría de producto
categorias = data['Categoría_Producto'].unique()
categoria_seleccionada = st.selectbox('Selecciona la categoría de producto:', categorias)

# Filtrar los datos según la categoría seleccionada
data_filtrada = data[data['Categoría_Producto'] == categoria_seleccionada]

# Crear el gráfico de dispersión con Plotly Express
fig = px.scatter(data_filtrada, x='Precio', y='Ventas_Predichas',
                 title=f'Dispersión de Precio vs Ventas Predichas ({categoria_seleccionada})',
                 labels={'Precio': 'Precio', 'Ventas_Predichas': 'Ventas Predichas'})

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Título y filtros
st.title('4. Dashboard Interactivo de Ventas Predichas')

# Filtrar por categoría de producto
categorias = ['Todas'] + list(data['Categoría_Producto'].unique())
categoria_seleccionada = st.selectbox('Selecciona la categoría de producto:', categorias)

# Filtrar los datos según la categoría seleccionada
if categoria_seleccionada == 'Todas':
    data_filtrada = data
else:
    data_filtrada = data[data['Categoría_Producto'] == categoria_seleccionada]

# Gráfico de dispersión: Precio vs Ventas Predichas
fig_dispersion = px.scatter(
    data_filtrada,
    x='Precio',
    y='Ventas_Predichas',
    color='Categoría_Producto',
    title=f'Relación entre Precio y Ventas Predichas ({categoria_seleccionada})',
    labels={'Precio': 'Precio', 'Ventas_Predichas': 'Ventas Predichas'}
)
st.plotly_chart(fig_dispersion)

# Gráfico de líneas: Ventas Predichas a lo largo del tiempo
fig_lineas = px.line(
    data_filtrada,
    x='Mes',
    y='Ventas_Predichas',
    color='Producto',
    title=f'Ventas Predichas a lo largo del tiempo ({categoria_seleccionada})',
    labels={'Mes': 'Mes', 'Ventas_Predichas': 'Ventas Predichas'}
)
st.plotly_chart(fig_lineas)

# Sección 5: Análisis de Resultados
st.header('5. Análisis de Resultados')
st.write("""
### Insights Clave:
- **Relación Precio-Ventas:** En la visualización de dispersión, podemos observar que los productos con un precio más alto tienden a tener menos ventas predichas, mientras que los productos más baratos tienen un mayor volumen de ventas predichas. 
- **Tendencia por Categoría:** Según la categoría seleccionada, vemos patrones específicos. Algunas categorías muestran una relación más marcada entre el precio y las ventas, lo que podría indicar que los clientes son más sensibles al precio en esas categorías.
- **Ventas a lo largo del tiempo:** En el gráfico de líneas, podemos observar cómo las ventas predichas fluctúan a lo largo de los meses. Es interesante notar los picos en ciertos meses, lo que podría estar relacionado con factores estacionales o promociones de productos específicos.
""")

# Sección 6: Conclusiones y Recomendaciones
st.header('6. Conclusiones y Recomendaciones')
st.write("""
### Conclusiones:
- La relación entre el precio y las ventas predichas es clara en la mayoría de las categorías, destacando que los productos más asequibles suelen vender más.
- Las tendencias de ventas varían significativamente entre categorías, lo que sugiere que el análisis de los datos por separado para cada categoría puede generar insights más útiles.

### Recomendaciones:
- **Ajuste de precios:** Considerar estrategias de precios diferentes para las categorías donde la sensibilidad al precio es mayor.
- **Promociones:** Analizar los meses donde se observan picos en las ventas predichas para alinear promociones y campañas de marketing.
- **Análisis continuo:** Se recomienda continuar monitoreando las ventas a lo largo del tiempo y ajustar el modelo predictivo con nuevos datos para mejorar la precisión de las predicciones.
""")