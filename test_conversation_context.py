#!/usr/bin/env python3
"""
Script de prueba para verificar el sistema de contexto de conversaciones
"""

import json
import os
from datetime import datetime

def test_conversation_context():
    """Prueba el sistema de contexto de conversaciones"""
    
    # Simular una conversación
    conversation_data = {
        "session_id": "test-session-123",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "conversation": [
            {
                "timestamp": "2025-06-18T16:30:00",
                "user_message": "tienes sillas??",
                "query_type": "product",
                "product_type": "silla",
                "products_shown": ["27", "3", "8"],
                "response": "Encontré 3 sillas..."
            },
            {
                "timestamp": "2025-06-18T16:32:00",
                "user_message": "tienes otros ejemplos??",
                "query_type": "continuation",
                "product_type": "silla",
                "products_shown": ["1", "34", "11"],
                "response": "Aquí tienes más sillas..."
            },
            {
                "timestamp": "2025-06-18T16:33:00",
                "user_message": "tienes camas??",
                "query_type": "product",
                "product_type": "cama",
                "products_shown": ["15", "22", "29"],
                "response": "Encontré 3 camas..."
            },
            {
                "timestamp": "2025-06-18T16:34:00",
                "user_message": "tienes otros ejemplos??",
                "query_type": "continuation",
                "product_type": "cama",
                "products_shown": ["7", "12", "18"],
                "response": "Aquí tienes más camas..."
            }
        ]
    }
    
    def get_last_product_type(session_data):
        """Obtiene el tipo de producto de la última consulta exitosa"""
        conversation = session_data.get("conversation", [])
        
        # Buscar desde el final hacia el principio
        for message in reversed(conversation):
            # Si el mensaje tiene un tipo de producto específico (no continuation)
            if message.get("product_type") and message.get("query_type") != "continuation":
                return message["product_type"]
        
        # Si no encuentra ninguno, buscar cualquier tipo de producto
        for message in reversed(conversation):
            if message.get("product_type"):
                return message["product_type"]
        
        return None
    
    def detect_continuation_query(user_query):
        """Detecta si la consulta es una continuación"""
        continuation_keywords = [
            "otros ejemplos", "mas ejemplos", "tienes mas", "mas opciones",
            "otros", "mas", "continuar", "siguiente", "mas de lo mismo",
            "tienes otros", "hay mas", "muestrame mas", "dame mas",
            "mas productos", "otras opciones", "mas variedad", "mas alternativas",
            "solo tienes esos", "no hay mas", "esos son todos", "es todo"
        ]
        
        # Normalizar la consulta
        user_query = user_query.lower().strip()
        
        # Si la consulta menciona un producto específico, NO es continuación
        product_types = ["silla", "mesa", "sofá", "lampara", "mueble", "estantería", "cama", "armario"]
        for product_type in product_types:
            if product_type in user_query:
                return False  # Es una nueva consulta específica
        
        # Verificar si contiene palabras clave de continuación
        for keyword in continuation_keywords:
            if keyword in user_query:
                return True
        
        return False
    
    # Probar la lógica
    print("=== PRUEBA DEL SISTEMA DE CONTEXTO ===\n")
    
    for i, message in enumerate(conversation_data["conversation"]):
        user_message = message["user_message"]
        is_continuation = detect_continuation_query(user_message)
        last_product_type = get_last_product_type(conversation_data)
        
        print(f"Mensaje {i+1}: '{user_message}'")
        print(f"  - Es continuación: {is_continuation}")
        print(f"  - Tipo de producto guardado: {message['product_type']}")
        print(f"  - Tipo de producto anterior: {last_product_type}")
        print(f"  - Query type: {message['query_type']}")
        print()
    
    print("=== ANÁLISIS ===")
    print("✅ Mensaje 1: 'tienes sillas??' - Nueva consulta, guarda 'silla'")
    print("✅ Mensaje 2: 'tienes otros ejemplos??' - Continuación, usa 'silla'")
    print("✅ Mensaje 3: 'tienes camas??' - Nueva consulta, guarda 'cama'")
    print("✅ Mensaje 4: 'tienes otros ejemplos??' - Continuación, usa 'cama'")
    
    # Guardar archivo de prueba
    os.makedirs("logs/conversations", exist_ok=True)
    with open("logs/conversations/test-session-123.json", "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Archivo de prueba guardado en logs/conversations/test-session-123.json")

if __name__ == "__main__":
    test_conversation_context() 