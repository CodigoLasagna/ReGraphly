import math

# Abre el archivo para lectura
def obtener_datos_de_coordenadas_geograficas(filename='', scale: int = 100000):
    archivo = open(filename, 'r')

    # Inicializa una lista para almacenar los datos
    datos = []

    # Coordenadas de Martir-de-Cuilapan (punto de referencia)
    lat_ref = math.radians(32.20757)
    lon_ref = math.radians(-116.89570)

    # Radio de la Tierra en metros (aproximadamente)
    radio_tierra = 6371000

    # Función para convertir latitud y longitud a coordenadas cartesianas
    def lat_lon_to_xy(lat, lon):
        lat = math.radians(lat)
        lon = math.radians(lon)
        x = (lon - lon_ref) * radio_tierra
        y = (lat - lat_ref) * radio_tierra * math.cos(lat_ref)
        return x / scale, y / scale

    # Lee cada línea del archivo
    for linea in archivo:
        # Ignora las líneas vacías y las que comienzan con #
        if not linea.strip() or linea.strip().startswith('#'):
            continue

        # Divide la línea en partes utilizando el carácter de tabulación como separador
        partes = linea.strip().split('\t')

        # Verifica que haya al menos dos partes (nombre de ubicación y coordenadas)
        if len(partes) >= 2:
            name = partes[0]
            coordenadas = partes[1].split(',')
            if len(coordenadas) == 2:
                latitud = float(coordenadas[0])
                longitud = float(coordenadas[1])
                x, y = lat_lon_to_xy(latitud, longitud)
                datos.append((name, x, y))

    archivo.close()  # Es importante cerrar el archivo después de usarlo
    return datos


def print_coords_list(datos = []):
    for name, x, y in datos:
        print(f'Ubicación: {name}, x: {x}, y: {y}')
