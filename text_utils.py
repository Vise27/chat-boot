import unicodedata
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

COLOR_VARIANTS = {
    # Gris
    "grises": "gris",
    "plateado": "gris",
    "plateada": "gris",
    # Negro
    "negra": "negro",
    "negros": "negro",
    "negras": "negro",
    # Blanco
    "blanca": "blanco",
    "blancos": "blanco",
    "blancas": "blanco",
    # Madera
    "maderas": "madera",
    "natural": "madera",
    "naturales": "madera",
    # Beige
    "beiges": "beige",
    "crema": "beige",
    "cremas": "beige",
    # Azul
    "azules": "azul",
    "celeste": "azul",
    "celestes": "azul",
    # Rojo
    "roja": "rojo",
    "rojos": "rojo",
    "rojas": "rojo",
    # Verde
    "verdes": "verde",
    "esmeralda": "verde",
    # Amarillo
    "amarilla": "amarillo",
    "amarillos": "amarillo",
    "amarillas": "amarillo",
    # Rosa
    "rosas": "rosa",
    "fucsia": "rosa",
    # Dorado
    "dorada": "dorado",
    "dorados": "dorado",
    "doradas": "dorado",
    # Marrón
    "marron": "marrón",
    "cafe": "marrón",
    "café": "marrón",
    "marrones": "marrón",
    "marrónes": "marrón"
}

# Diccionario de errores ortográficos comunes
COMMON_MISTAKES = {
    # Errores de tildes
    "silla": ["sila", "sillas", "sillas", "silla"],
    "mesa": ["mesa", "mesas", "mesa"],
    "sofa": ["sofá", "sofa", "sofas", "sofás"],
    "lampara": ["lámpara", "lampara", "lamparas", "lámparas"],
    "mueble": ["muebles", "mueble"],
    "estanteria": ["estantería", "estanteria", "estanterias", "estanterías"],
    "cama": ["camas", "cama"],
    "armario": ["armarios", "armario"],
    
    # Errores de escritura común
    "silla": ["sila", "silla", "sillas", "silla"],
    "mesa": ["mesa", "mesas", "mesa"],
    "sofa": ["sofa", "sofá", "sofas", "sofás"],
    "lampara": ["lampara", "lámpara", "lamparas", "lámparas"],
    "mueble": ["muebles", "mueble"],
    "estanteria": ["estanteria", "estantería", "estanterias", "estanterías"],
    "cama": ["camas", "cama"],
    "armario": ["armarios", "armario"],
    
    # Errores de letras duplicadas o faltantes
    "silla": ["sila", "silla", "sillas", "silla"],
    "mesa": ["mesa", "mesas", "mesa"],
    "sofa": ["sofa", "sofá", "sofas", "sofás"],
    "lampara": ["lampara", "lámpara", "lamparas", "lámparas"],
    "mueble": ["muebles", "mueble"],
    "estanteria": ["estanteria", "estantería", "estanterias", "estanterías"],
    "cama": ["camas", "cama"],
    "armario": ["armarios", "armario"],
}

# Sinónimos y variaciones de productos
PRODUCT_SYNONYMS = {
    "silla": ["silla", "asiento", "sillón", "butaca", "taburete", "banqueta"],
    "mesa": ["mesa", "tabla", "escritorio", "mesita", "mesilla"],
    "sofa": ["sofá", "sofa", "sillón", "diván", "canapé", "tresillo"],
    "lampara": ["lámpara", "lampara", "luminaria", "foco", "bombilla"],
    "mueble": ["mueble", "mobiliario", "mobiliario", "muebles"],
    "estanteria": ["estantería", "estanteria", "estante", "repisa", "biblioteca"],
    "cama": ["cama", "lecho", "colchón", "litera", "nido"],
    "armario": ["armario", "ropero", "closet", "guardarropa", "vestidor"],
    "escritorio": ["escritorio", "mesa de trabajo", "bureau", "escritorio"],
    "velador": ["velador", "mesita de noche", "mesilla", "lámpara de noche"],
    "biombo": ["biombo", "separador", "divisor", "pantalla"],
    "cortina": ["cortina", "persiana", "estore", "tela"],
    "alfombra": ["alfombra", "tapete", "moqueta", "carpeta"],
    "cojin": ["cojín", "cojin", "almohadón", "cushion"],
    "fundas": ["fundas", "fundas", "cubiertas", "protectores"]
}

def normalize_color(color: str) -> str:
    """Normaliza variantes de colores a su forma canónica"""
    color = normalize_text(color)
    return COLOR_VARIANTS.get(color, color)

def normalize_text(text: str) -> str:
    """
    Normaliza texto completo para búsquedas:
    1. Elimina TODOS los signos de puntuación
    2. Normaliza espacios (incluyendo espacios especiales)
    3. Elimina tildes y diacríticos
    4. Convierte a minúsculas
    """
    if not text:
        return ""

    # Paso 1: Eliminar signos de puntuación (conserva letras, números y espacios)
    text = re.sub(r'[^\w\s]|_', '', text)
    
    # Paso 2: Normalizar espacios (incluye no-break spaces, tabs, etc.)
    text = ' '.join(text.split())  # Esto también elimina espacios duplicados
    
    # Paso 3: Eliminar tildes y convertir a minúsculas
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    return text.strip()

def contains_word(text: str, word: str) -> bool:
    """Verifica si un texto contiene una palabra (sin sensibilidad a tildes)"""
    return normalize_text(word) in normalize_text(text)

def similarity(a: str, b: str) -> float:
    """Calcula la similitud entre dos strings usando SequenceMatcher"""
    return SequenceMatcher(None, a, b).ratio()

def correct_common_mistakes(word: str) -> str:
    """Corrige errores ortográficos comunes"""
    normalized_word = normalize_text(word)
    
    # Buscar en errores comunes
    for correct, mistakes in COMMON_MISTAKES.items():
        if normalized_word in mistakes:
            return correct
    
    return word

def find_product_synonyms(word: str) -> List[str]:
    """Encuentra sinónimos de un producto"""
    normalized_word = normalize_text(word)
    
    for product, synonyms in PRODUCT_SYNONYMS.items():
        if normalized_word in [normalize_text(s) for s in synonyms]:
            return synonyms
    
    return [word]

def fuzzy_search(query: str, product_names: List[str], threshold: float = 0.7) -> List[str]:
    """
    Búsqueda fuzzy que encuentra productos similares
    """
    normalized_query = normalize_text(query)
    matches = []
    
    for product_name in product_names:
        normalized_name = normalize_text(product_name)
        
        # Búsqueda exacta
        if normalized_query in normalized_name or normalized_name in normalized_query:
            matches.append(product_name)
            continue
        
        # Búsqueda fuzzy
        sim = similarity(normalized_query, normalized_name)
        if sim >= threshold:
            matches.append(product_name)
    
    return matches

def normalize_singular_plural(word: str) -> List[str]:
    """Normaliza una palabra para manejar singular/plural"""
    word = word.lower().strip()
    
    # Reglas básicas de singular/plural en español
    if word.endswith('s'):
        # Es plural, agregar singular
        singular = word[:-1]
        return [word, singular]  # [plural, singular]
    else:
        # Es singular, agregar plural
        plural = word + 's'
        return [word, plural]  # [singular, plural]

def smart_product_search(query: str, product_names: List[str]) -> List[str]:
    """
    Búsqueda inteligente que maneja singular/plural y sinónimos
    """
    results = set()
    
    # Normalizar query
    normalized_query = normalize_text(query)
    
    # 1. Búsqueda directa con singular/plural
    query_variations = normalize_singular_plural(normalized_query)
    for variation in query_variations:
        for product_name in product_names:
            normalized_name = normalize_text(product_name)
            if variation in normalized_name or normalized_name in variation:
                results.add(product_name)
    
    # 2. Buscar sinónimos con singular/plural
    synonyms = find_product_synonyms(normalized_query)
    for synonym in synonyms:
        synonym_variations = normalize_singular_plural(normalize_text(synonym))
        for variation in synonym_variations:
            for product_name in product_names:
                normalized_name = normalize_text(product_name)
                if variation in normalized_name or normalized_name in variation:
                    results.add(product_name)
    
    # 3. Búsqueda por palabras individuales
    words = normalized_query.split()
    for word in words:
        if len(word) > 2:  # Solo palabras de más de 2 caracteres
            word_variations = normalize_singular_plural(word)
            for variation in word_variations:
                for product_name in product_names:
                    normalized_name = normalize_text(product_name)
                    if variation in normalized_name:
                        results.add(product_name)
    
    return list(results)

def improve_product_search(query: str, product_names: List[str]) -> List[str]:
    """
    Mejora la búsqueda de productos con múltiples estrategias:
    1. Corrección de errores ortográficos
    2. Búsqueda de sinónimos
    3. Búsqueda fuzzy
    4. Manejo de singular/plural
    """
    results = set()
    
    # Normalizar query
    normalized_query = normalize_text(query)
    
    # 1. Búsqueda inteligente con singular/plural
    smart_matches = smart_product_search(query, product_names)
    results.update(smart_matches)
    
    # 2. Búsqueda directa
    direct_matches = fuzzy_search(normalized_query, product_names, threshold=0.8)
    results.update(direct_matches)
    
    # 3. Corregir errores comunes
    corrected_query = correct_common_mistakes(normalized_query)
    if corrected_query != normalized_query:
        corrected_matches = fuzzy_search(corrected_query, product_names, threshold=0.8)
        results.update(corrected_matches)
    
    # 4. Buscar sinónimos
    synonyms = find_product_synonyms(normalized_query)
    for synonym in synonyms:
        synonym_matches = fuzzy_search(normalize_text(synonym), product_names, threshold=0.7)
        results.update(synonym_matches)
    
    # 5. Búsqueda por palabras individuales
    words = normalized_query.split()
    for word in words:
        if len(word) > 2:  # Solo palabras de más de 2 caracteres
            word_matches = fuzzy_search(word, product_names, threshold=0.6)
            results.update(word_matches)
    
    return list(results)

def clean_query(text: str) -> str:
    """Limpia el texto removiendo signos de puntuación y normalizando"""
    if not text:
        return ""
    # Remover signos de puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Normalizar como antes
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) 
              if unicodedata.category(c) != 'Mn')
    return text    

def detect_query_type(query: str) -> str:
    """Detecta el tipo de consulta basado en palabras clave"""
    query = normalize_text(query)
    
    # Palabras clave para atributos
    attribute_keywords = [
        "color", "material", "tamaño", "medida", "dimension", 
        "alto", "ancho", "largo", "estilo", "acabado"
    ]
    
    # Lista simplificada de colores base (sin variantes)
    color_keywords = ["gris", "negro", "blanco", "madera", "beige", "azul", 
                     "rojo", "verde", "amarillo", "rosa", "dorado", "marrón"]
    
    # 1. Primero verificar colores (incluyendo variantes)
    query_words = query.split()
    for word in query_words:
        normalized_color = normalize_color(word)
        if normalized_color in color_keywords:
            return "attributes"
    
    # 2. Luego verificar otros atributos
    if any(attr in query for attr in attribute_keywords):
        return "attributes"
    
    # 3. Verificar espacios
    spaces = ["comedor", "sala", "living", "dormitorio", "habitacion", "cocina", 
              "oficina", "jardin", "terraza", "baño", "cuarto", "espacio"]
    if any(espacio in query for espacio in spaces):
        return "style"
    
    # 4. Verificar palabras de estilo
    style_keywords = ["para", "estilo", "ambiente", "look", "diseño", "decoración"]
    if any(kw in query for kw in style_keywords):
        return "style"
    
    # 5. Verificar productos específicos
    product_keywords = ["lámpara", "silla", "sofá", "mesa", "mueble"]
    if any(re.search(rf"\b{kw}\b", query) for kw in product_keywords):
        return "product"
    
    # 6. Si no coincide con nada anterior
    return "generic"   