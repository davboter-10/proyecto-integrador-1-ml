\# Datos del Proyecto



\## Ubicación



Los datos originales están almacenados en AWS S3:

\- \*\*Bucket:\*\* `s3://ml-reestructuraciones-029885540752`

\- \*\*Ruta:\*\* `data/raw/`



\## Descripción



\*\*Archivo principal:\*\* `reestructuraciones\_711k.csv`

\- \*\*Tamaño:\*\* 711,000 registros

\- \*\*Features:\*\* 40 variables

\- \*\*Target:\*\* `reestructurado` (binario: 0/1)



\### Variables



\#### Financieras

\- `monto\_credito`: Monto del préstamo

\- `plazo\_meses`: Plazo en meses

\- `tasa\_interes`: Tasa de interés anual

\- `cuota\_mensual`: Valor de la cuota

\- \[...]



\#### Demográficas

\- `edad`: Edad del cliente

\- `ingreso\_mensual`: Ingreso mensual declarado

\- `score\_crediticio`: Score de crédito (0-1000)

\- \[...]



\### Target



\- `reestructurado`: 

&nbsp; - 0: Cliente no requirió reestructuración

&nbsp; - 1: Cliente requirió reestructuración



\### Distribución



\- \*\*Clase 0:\*\* 85% (no reestructurado)

\- \*\*Clase 1:\*\* 15% (reestructurado)

\- \*\*Balance:\*\* Desbalanceado → Se aplicó SMOTE



\## Acceso



Para descargar los datos desde S3:

```bash

aws s3 cp s3://ml-reestructuraciones-029885540752/data/raw/data.csv ./data/

```



\## Procesamiento



Ver notebooks:

\- `01\_eda.ipynb`: Análisis exploratorio

\- `02\_preparacion\_datos.ipynb`: Transformaciones

