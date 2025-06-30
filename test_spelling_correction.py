#!/usr/bin/env python3
"""
Script de prueba para demostrar la corrección de errores ortográficos
y la búsqueda mejorada de productos
"""

from text_utils import (
    improve_product_search, 
    correct_common_mistakes, 
    find_product_synonyms,
    fuzzy_search,
    normalize_text
)

def test_spelling_correction():
    """Prueba las funciones de corrección de errores"""
    
    # Productos de ejemplo
    sample_products = [
        "Silla Ergonómica Ejecutiva",
        "Silla de Comedor Acapulco", 
        "Silla de Oficina Giratoria",
        "Mesa de Comedor Moderna",
        "Mesa de Centro Elegante",
        "Sofá de 3 Plazas",
        "Lámpara de Pie Moderna",
        "Estantería de Madera",
        "Cama Matrimonial",
        "Armario de Dormitorio"
    ]
    
    # Casos de prueba con errores ortográficos
    test_cases = [
        # Errores de tildes
        "tienes silas??",           # silla sin tilde
        "busco mesas",              # mesa plural
        "quiero sofas",             # sofá sin tilde
        "necesito lamparas",        # lámpara sin tilde
        
        # Errores de escritura
        "tienes silas",             # silla mal escrita
        "busco mesas",              # mesa bien escrita
        "quiero sofas",             # sofá bien escrito
        "necesito lamparas",        # lámpara bien escrita
        
        # Sinónimos
        "tienes asientos??",        # sinónimo de silla
        "busco tablas",             # sinónimo de mesa
        "quiero sillones",          # sinónimo de sofá
        "necesito luminarias",      # sinónimo de lámpara
        
        # Errores mixtos
        "tienes silas ergonomicas", # múltiples errores
        "busco mesas de comedor",   # combinación correcta
        "quiero sofas modernos",    # combinación correcta
    ]
    
    print("=== PRUEBA DE CORRECCIÓN DE ERRORES ORTOGRÁFICOS ===\n")
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"Prueba {i}: '{test_query}'")
        
        # Normalizar query
        normalized = normalize_text(test_query)
        print(f"  Normalizado: '{normalized}'")
        
        # Corregir errores comunes
        corrected = correct_common_mistakes(normalized)
        print(f"  Corregido: '{corrected}'")
        
        # Buscar sinónimos
        synonyms = find_product_synonyms(normalized)
        print(f"  Sinónimos: {synonyms}")
        
        # Búsqueda mejorada
        results = improve_product_search(test_query, sample_products)
        print(f"  Productos encontrados: {len(results)}")
        for result in results[:3]:  # Mostrar solo los primeros 3
            print(f"    - {result}")
        
        print()

def test_fuzzy_search():
    """Prueba la búsqueda fuzzy"""
    
    print("=== PRUEBA DE BÚSQUEDA FUZZY ===\n")
    
    products = [
        "Silla Ergonómica Ejecutiva",
        "Silla de Comedor Acapulco",
        "Mesa de Comedor Moderna",
        "Sofá de 3 Plazas"
    ]
    
    fuzzy_queries = [
        "sila",           # Error común
        "mesa",           # Correcto
        "sofa",           # Sin tilde
        "silla ergonomica", # Sin tilde
        "mesa comedor",   # Combinación
    ]
    
    for query in fuzzy_queries:
        print(f"Buscando: '{query}'")
        results = fuzzy_search(query, products, threshold=0.6)
        print(f"  Resultados: {results}")
        print()

def test_real_scenarios():
    """Prueba escenarios reales de uso"""
    
    print("=== ESCENARIOS REALES ===\n")
    
    # Simular productos de la base de datos
    real_products = [
        "Silla Ergonómica Ejecutiva",
        "Silla de Comedor Acapulco",
        "Silla de Oficina Giratoria",
        "Mesa de Comedor Moderna",
        "Mesa de Centro Elegante",
        "Sofá de 3 Plazas",
        "Lámpara de Pie Moderna",
        "Estantería de Madera",
        "Cama Matrimonial",
        "Armario de Dormitorio"
    ]
    
    # Escenarios reales con errores típicos
    scenarios = [
        {
            "query": "tienes silas??",
            "description": "Usuario escribe 'sila' en lugar de 'silla'"
        },
        {
            "query": "busco mesas de comedor",
            "description": "Usuario busca mesa de comedor (correcto)"
        },
        {
            "query": "quiero sofas modernos",
            "description": "Usuario escribe 'sofas' sin tilde"
        },
        {
            "query": "necesito lamparas para sala",
            "description": "Usuario escribe 'lamparas' sin tilde"
        },
        {
            "query": "tienes asientos ergonomicos",
            "description": "Usuario usa sinónimo 'asientos' y sin tilde"
        }
    ]
    
    for scenario in scenarios:
        print(f"Escenario: {scenario['description']}")
        print(f"Query: '{scenario['query']}'")
        
        results = improve_product_search(scenario['query'], real_products)
        
        if results:
            print(f"✅ Encontrados {len(results)} productos:")
            for result in results[:3]:
                print(f"   - {result}")
        else:
            print("❌ No se encontraron productos")
        
        print()

if __name__ == "__main__":
    test_spelling_correction()
    test_fuzzy_search()
    test_real_scenarios()
    
    print("=== RESUMEN ===")
    print("✅ Corrección de errores ortográficos implementada")
    print("✅ Búsqueda fuzzy con umbral configurable")
    print("✅ Sinónimos de productos")
    print("✅ Normalización de texto mejorada")
    print("✅ Múltiples estrategias de búsqueda")
    print("\n¡El chatbot ahora es más tolerante a errores de escritura! 🎉") 