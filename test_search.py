#!/usr/bin/env python3
"""
Script de prueba para verificar la búsqueda singular/plural
"""

from text_utils import smart_product_search, normalize_text

# Productos de ejemplo (simulando tu BD)
test_products = [
    "Silla de Oficina Ergonómica",
    "Silla de Comedor Moderna", 
    "Silla Plegable para Eventos",
    "Sillas Plegables para Eventos",
    "Mesa de Comedor",
    "Mesas de Comedor",
    "Sofá Moderno",
    "Sofás Modernos"
]

def test_search():
    print("🧪 Probando búsqueda singular/plural...")
    print("=" * 50)
    
    # Casos de prueba
    test_cases = [
        "silla",
        "sillas", 
        "mesa",
        "mesas",
        "sofá",
        "sofás"
    ]
    
    for query in test_cases:
        print(f"\n🔍 Buscando: '{query}'")
        results = smart_product_search(query, test_products)
        print(f"✅ Encontrados: {len(results)} productos")
        for result in results:
            print(f"   - {result}")
    
    print("\n" + "=" * 50)
    print("✅ Prueba completada!")

if __name__ == "__main__":
    test_search() 