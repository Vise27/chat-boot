import httpx
import os 
import asyncio
import json
import unicodedata
from llama_sanitizer import sanitize_llama_response
import logging
from text_utils import normalize_text, contains_word, normalize_color, improve_product_search
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv() 
import re
from typing import Tuple

async def ask_llama(prompt: str, max_retries: int = 3) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    logging.info(f"API Key cargada: {'Sí' if api_key else 'No'}")
    
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-70b-8192",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                logging.error(f"Error en LLM: {e.response.text}")
                if attempt == max_retries - 1:
                    return "[]"
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Error inesperado: {str(e)}")
                return "[]"
    return "[]"

async def ask_llama_for_products(products: list[dict], user_message: str) -> list[str]:
    """Versión mejorada que maneja la nueva estructura de productos con corrección de errores"""
    try:
        # Obtener nombres de productos
        product_names = [p.get('name', '') for p in products if isinstance(p, dict)]
        
        # Usar búsqueda mejorada
        matched_names = improve_product_search(user_message, product_names)
        
        if matched_names:
            return matched_names
        
        # Si no hay resultados con búsqueda mejorada, usar LLM como respaldo
        normalized_query = normalize_text(user_message)
        
        product_list = []
        for p in products:
            if not isinstance(p, dict):
                continue
            name = p.get('name', '')
            product_list.append({
                "original_name": name,
                "normalized_name": normalize_text(name),
                "product_data": p
            })

        # Búsqueda directa sin LLM
        direct_matches = [
            p["original_name"] for p in product_list 
            if normalized_query in p["normalized_name"]
        ]

        if direct_matches:
            return direct_matches

        # Preparar datos para LLM
        products_str = "\n".join(
            f"- {p['original_name']} (SKU: {p['product_data'].get('sku', '')}, Tipo: {p['product_data'].get('type', '')})"
            for p in product_list
        )

        prompt = f"""
Productos disponibles:
{products_str}

El usuario busca: "{user_message}"

Identifica los productos más relevantes considerando:
1. Coincidencias exactas o similares
2. Sin diferenciar tildes
3. SKU y tipo como contexto adicional
4. Devuelve SOLO un JSON con los nombres exactos

Ejemplo: ["Sillón Elegante", "Silla Moderna"]
"""
        response = await ask_llama(prompt)
        return sanitize_llama_response(response, expected_type="list_str")

    except Exception as e:
        logging.error(f"Error en ask_llama_for_products: {e}")
        return []

async def ask_llama_summary_for_products(products: list[dict]) -> str:
    """Genera resumen profesional con todos los campos disponibles"""
    try:
        if not products:
            return "No encontré productos que coincidan exactamente con tu búsqueda."
            
        if len(products) == 1:
            product = products[0]
            prompt = f"""
Eres un experto en ventas de muebles. Crea una descripción precisa y atractiva en español para:

Nombre: {product.get('name', 'Producto')}
Tipo: {product.get('type', '')}
Descripción: {product.get('description', '')}
Precio: ${product.get('base_price', '')}
Stock: {'Disponible' if not product.get('stock_alert') else 'Últimas unidades'}
Ventas: {product.get('sales_count', 0)} unidades vendidas

REGLAS:
1. Responde SIEMPRE en español
2. Mantén el nombre del producto exactamente como está
3. Sé preciso con los datos
4. Destaca disponibilidad y precio
5. Menciona el ambiente ideal para este producto
6. Máximo 2 oraciones
7. Usa un tono amigable y profesional
8. NO traduzcas el nombre del producto
"""
        
        else:
            # Agrupar productos por tipo
            productos_por_tipo = {}
            for p in products:
                tipo = p.get('type', 'Otros')
                if tipo not in productos_por_tipo:
                    productos_por_tipo[tipo] = []
                productos_por_tipo[tipo].append(p)

            # Crear resumen por tipo
            productos_info = []
            newline = '\n'
            for tipo, productos in productos_por_tipo.items():
                productos_info.append(f"{newline}**{tipo.upper()}**{newline}")
                for p in productos:
                    productos_info.append(
                        f"* {p.get('name', 'Producto')}: ${p.get('base_price', '')}"
                    )

            prompt = f"""
Resume estos productos agrupados por tipo, destacando sus características y usos. Responde SIEMPRE en español:

{''.join(productos_info)}

REGLAS:
1. Responde SIEMPRE en español
2. Mantén la agrupación por tipo de producto
3. Menciona solo el nombre y precio para cada producto
4. Mantén los nombres de productos exactamente como están
5. NO traduzcas los nombres de los productos
6. Destaca las características clave de cada tipo
7. Sugiere el mejor uso para cada grupo de productos
8. Usa un tono amigable y profesional
9. Máximo 3 oraciones por grupo de productos
10. Destaca las características únicas de cada producto
11. Asegúrate de que cada producto sea relevante para el ambiente solicitado
"""

        response = await ask_llama(prompt)
        return response if response.strip() else f"Encontré {len(products)} opciones relevantes."
        
    except Exception as e:
        logging.error(f"Error al generar resumen: {str(e)}")
        return f"Encontré {len(products)} opciones relevantes."

async def ask_llama_for_style_recommendations(products: list[dict], user_message: str) -> list[str]:
    """Recomienda productos basados en ambientes interiores"""
    try:
        AMBIENTE_KEYWORDS = {
            "dormitorio": {
                "keywords": ["dormitorio", "cuarto", "habitación", "recámara", "noche", "descanso", "velador", "cama", "mesita"],
                "description": "un espacio para descansar y relajarse",
                "productos_tipicos": ["cama", "velador", "armario", "mesita de noche", "lámpara de noche"],
                "productos_no_relevantes": ["sofá", "mesa de comedor", "taburete", "cortina de baño", "escritorio", "archivador"]
            },
            "oficina": {
                "keywords": ["oficina", "escritorio", "trabajo", "estudio", "ergonómica", "archivador"],
                "description": "un espacio para trabajar y concentrarse",
                "productos_tipicos": ["escritorio", "silla ergonómica", "archivador", "lámpara de escritorio", "estantería"],
                "productos_no_relevantes": ["sofá", "cama", "velador", "cortina de baño", "mesa de comedor", "biombo", "fundas"]
            },
            "sala": {
                "keywords": ["sala", "estar", "living", "sofá", "centro", "decorativa", "recibidor"],
                "description": "un espacio para socializar y recibir visitas",
                "productos_tipicos": ["sofá", "mesa de centro", "sillón", "lámpara de pie", "estantería decorativa"],
                "productos_no_relevantes": ["escritorio", "archivador", "velador", "cortina de baño", "silla ergonómica"]
            },
            "comedor": {
                "keywords": ["comedor", "dining", "mesa", "silla", "banqueta"],
                "description": "un espacio para compartir comidas",
                "productos_tipicos": ["mesa de comedor", "sillas de comedor", "banqueta", "lámpara colgante"],
                "productos_no_relevantes": ["escritorio", "archivador", "sofá", "cortina de baño", "silla ergonómica"]
            },
            "cocina": {
                "keywords": ["cocina", "taburete", "isla", "banqueta", "desayunador"],
                "description": "un espacio para preparar y disfrutar comidas",
                "productos_tipicos": ["taburete", "banqueta", "mesa de desayunador", "estantería de cocina"],
                "productos_no_relevantes": ["sofá", "escritorio", "cama", "cortina de baño", "silla ergonómica"]
            }
        }

        normalized_query = normalize_text(user_message)
        
        # Detectar ambiente
        ambiente = None
        for amb, data in AMBIENTE_KEYWORDS.items():
            if amb in normalized_query or any(kw in normalized_query for kw in data["keywords"]):
                ambiente = amb
                break

        if not ambiente:
            return []

        ambiente_data = AMBIENTE_KEYWORDS[ambiente]
        
        # Preparar prompt más detallado
        newline = '\n'
        prompt = f"""
Eres un experto en decoración de interiores y análisis de productos. Tu tarea es analizar cuidadosamente cada producto y determinar si es realmente relevante para {ambiente_data['description']}.

ANÁLISIS REQUERIDO:
1. Para cada producto, debes considerar:
   - ¿El producto es funcionalmente útil en {ambiente}?
   - ¿El producto es típicamente usado en {ambiente}?
   - ¿El producto tiene sentido en el contexto de {ambiente}?
   - ¿El producto podría ser confundido con otro tipo de mueble?

2. Productos que DEBEN ser descartados:
   - Productos claramente diseñados para otros ambientes
   - Productos que no tienen una función clara en {ambiente}
   - Productos que podrían confundir al usuario
   - Productos que no son típicos de {ambiente}

3. Productos que DEBEN ser priorizados:
   - Productos típicos de {ambiente}: {', '.join(ambiente_data['productos_tipicos'])}
   - Productos que mencionan {ambiente} en su descripción
   - Productos que tienen una función clara en {ambiente}

Productos a analizar:
{newline.join(
    f"- {p['name']} (SKU: {p.get('sku', '')}, Precio: ${p.get('base_price', '')}, Descripción: {p.get('description', '')})"
    for p in products
)}

INSTRUCCIONES:
1. Analiza CADA producto individualmente
2. Para cada producto, escribe una breve justificación de por qué SÍ o por qué NO es relevante
3. Selecciona SOLO los productos que son CLARAMENTE relevantes para {ambiente}
4. NO incluyas productos que generen dudas
5. Devuelve un JSON con:
   - Lista de productos seleccionados
   - Explicación de por qué cada producto es relevante
   - Lista de productos descartados y por qué

Ejemplo de respuesta:
{{
    "productos_seleccionados": ["Silla Ergonómica X", "Escritorio Moderno Y"],
    "explicacion": "Estos productos son ideales para tu oficina porque...",
    "productos_descartados": [
        {{
            "nombre": "Cortina de Baño",
            "razon": "Este producto está diseñado específicamente para baños y no tiene función en una oficina"
        }}
    ]
}}
"""
        response = await ask_llama(prompt)
        try:
            result = json.loads(response)
            productos_seleccionados = result.get("productos_seleccionados", [])
            
            # Verificación final de relevancia
            productos_finales = []
            for producto in productos_seleccionados:
                nombre = normalize_text(producto).lower()
                if not any(no_relevante in nombre for no_relevante in ambiente_data['productos_no_relevantes']):
                    productos_finales.append(producto)
            
            return productos_finales
        except:
            return sanitize_llama_response(response, expected_type="list_str")

    except Exception as e:
        logging.error(f"Error en recomendaciones por ambiente: {e}")
        return []

async def ask_llama_for_attributes_query(products: list[dict], user_query: str) -> list[dict]:
    """Busca productos basados en atributos con soporte para VARIABLE"""
    try:
        normalized_query = normalize_text(user_query)
        
        color_keywords = ["gris", "negro", "blanco", "madera", "beige", "azul", 
                         "rojo", "verde", "amarillo", "rosa", "dorado", "marrón"]
        product_types = ["silla", "mesa", "sofá", "lámpara", "mueble"]
        
        colors_in_query = []
        for word in normalized_query.split():
            norm_color = normalize_color(word)
            if norm_color in color_keywords:
                colors_in_query.append(norm_color)
        
        product_type_in_query = next(
            (pt for pt in product_types if pt in normalized_query),
            None
        )

        matched_products = []
        for product in products:
            if not isinstance(product, dict):
                continue
                
            attributes = product.get('attributes_normalized', {})
            if isinstance(attributes, str):
                try:
                    attributes = json.loads(attributes)
                except:
                    attributes = {}
            
            match = True
            
            # Manejo de productos variables
            if product.get('type') == 'VARIABLE':
                if colors_in_query and 'variantes' in attributes and 'color' in attributes['variantes']:
                    # Producto tiene variantes de color que podrían coincidir
                    pass
                else:
                    match = False
            
            # Verificación de color para no variables
            elif colors_in_query:
                product_color = normalize_text(str(attributes.get("color", "")))
                if not any(color == product_color for color in colors_in_query):
                    match = False
            
            # Verificación del tipo
            if match and product_type_in_query:
                product_name = normalize_text(product.get('name', ''))
                if product_type_in_query not in product_name:
                    match = False
            
            if match:
                matched_products.append(product)
        
        if matched_products:
            matched_products.sort(
                key=lambda p: (
                    p.get('sales_count', 0),
                    p.get('base_price', 0)
                ),
                reverse=True
            )
            return matched_products
        
        # Respaldo con LLM
        products_data = []
        for p in products:  # Analizar todos los productos
            attrs = p.get('attributes_normalized', {})
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except:
                    attrs = {}
            
            products_data.append({
                "id": p.get("id"),
                "name": p.get("name", ""),
                "sku": p.get("sku", ""),
                "type": p.get("type", ""),
                "attributes": attrs
            })

        prompt = f"""
Consulta: "{user_query}"

Productos:
{json.dumps(products_data, indent=2)}

INSTRUCCIONES:
1. Busca coincidencias con:
   - Color: {colors_in_query or 'No especificado'}
   - Tipo: {product_type_in_query or 'No especificado'}
2. Considera variantes para productos VARIABLE
3. Devuelve SOLO un JSON con los IDs
"""
        response = await ask_llama(prompt)
        product_ids = sanitize_llama_response(response, expected_type="list_int")
        
        return [p for p in products if str(p.get("id")) in map(str, product_ids)]
    
    except Exception as e:
        logging.error(f"Error en ask_llama_for_attributes_query: {e}")
        return []

def extract_quantity_from_query(query: str) -> Tuple[int, str]:
    """Extrae la cantidad solicitada del query y retorna la cantidad y el query limpio"""
    # Patrones comunes para detectar cantidades
    patterns = [
        r'(\d+)\s*(?:opciones|productos|muebles|sillas|mesas|sofás|lamparas|muebles|estanterías|camas|armarios)',
        r'quiero\s*(\d+)',
        r'necesito\s*(\d+)',
        r'busco\s*(\d+)',
        r'mostrar\s*(\d+)',
        r'ver\s*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            quantity = int(match.group(1))
            # Remover la parte de la cantidad del query
            clean_query = re.sub(pattern, '', query.lower(), flags=re.IGNORECASE).strip()
            return quantity, clean_query
    
    return 4, query  # Valor por defecto

async def ask_llama_for_template_recommendations(templates: list[dict], template_products: list[dict], user_message: str) -> list[dict]:
    """Recomienda plantillas de diseño basadas en la consulta del usuario"""
    try:
        ROOM_TYPES = {
            "dormitorio": ["dormitorio", "cuarto", "habitación", "recámara", "noche", "descanso"],
            "oficina": ["oficina", "escritorio", "trabajo", "estudio", "ergonómica"],
            "sala": ["sala", "estar", "living", "sofá", "centro", "decorativa", "recibidor"],
            "comedor": ["comedor", "dining", "mesa", "silla", "banqueta"],
            "cocina": ["cocina", "taburete", "isla", "banqueta", "desayunador"]
        }

        STYLES = {
            "moderno": ["moderno", "contemporáneo", "minimalista", "sencillo"],
            "clásico": ["clásico", "tradicional", "elegante", "formal"],
            "industrial": ["industrial", "rústico", "vintage", "loft"],
            "escandinavo": ["escandinavo", "nórdico", "simple", "natural"],
            "bohemio": ["bohemio", "boho", "artístico", "colorido"]
        }

        normalized_query = normalize_text(user_message)
        logging.info(f"Buscando plantillas para consulta normalizada: {normalized_query}")
        
        # Detectar tipo de habitación
        room_type = None
        for rt, keywords in ROOM_TYPES.items():
            if rt in normalized_query or any(kw in normalized_query for kw in keywords):
                room_type = rt
                logging.info(f"Tipo de habitación detectado: {room_type}")
                break

        # Detectar estilo
        style = None
        for st, keywords in STYLES.items():
            if st in normalized_query or any(kw in normalized_query for kw in keywords):
                style = st
                logging.info(f"Estilo detectado: {style}")
                break

        if not room_type:
            logging.warning("No se detectó tipo de habitación en la consulta")
            return []

        # Filtrar plantillas por tipo de habitación primero
        filtered_templates = [
            t for t in templates 
            if t.get('room_type', '').lower() == room_type.lower()
        ]
        logging.info(f"Plantillas filtradas por tipo de habitación: {len(filtered_templates)}")

        # Si hay estilo, filtrar también por estilo
        if style:
            filtered_templates = [
                t for t in filtered_templates 
                if t.get('style', '').lower() == style.lower()
            ]
            logging.info(f"Plantillas filtradas por estilo: {len(filtered_templates)}")

        # Si no hay resultados después del filtrado básico, usar LLM
        if not filtered_templates:
            logging.info("Usando LLM para búsqueda de plantillas")
            templates_data = []
            for t in templates:
                products = [
                    tp for tp in template_products 
                    if str(tp.get('template_id')) == str(t.get('id'))
                ]
                templates_data.append({
                    "id": t.get("id"),
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "style": t.get("style", ""),
                    "room_type": t.get("room_type", ""),
                    "total_price": t.get("total_price", 0),
                    "discount": t.get("discount", 0),
                    "sales_count": t.get("sales_count", 0),
                    "products": [
                        {
                            "id": p.get("product_id"),
                            "quantity": p.get("quantity", 1),
                            "optional": p.get("is_optional", False),
                            "notes": p.get("notes", "")
                        }
                        for p in products
                    ]
                })

            prompt = f"""
Consulta: "{user_message}"

Plantillas disponibles:
{json.dumps(templates_data, indent=2)}

INSTRUCCIONES:
1. Busca coincidencias con:
   - Tipo de habitación: {room_type}
   - Estilo: {style or 'No especificado'}
2. Considera:
   - Precio total
   - Descuentos disponibles
   - Popularidad (ventas)
   - Productos incluidos
3. Devuelve SOLO un JSON con los IDs de las plantillas más relevantes
"""
            response = await ask_llama(prompt)
            template_ids = sanitize_llama_response(response, expected_type="list_int")
            logging.info(f"IDs de plantillas encontrados por LLM: {template_ids}")
            
            return [t for t in templates if str(t.get("id")) in map(str, template_ids)]
        
        # Ordenar por popularidad y precio
        sorted_templates = sorted(
            filtered_templates,
            key=lambda x: (x.get('sales_count', 0), -x.get('total_price', 0)),
            reverse=True
        )
        logging.info(f"Plantillas ordenadas por popularidad: {len(sorted_templates)}")
        
        return sorted_templates

    except Exception as e:
        logging.error(f"Error en recomendaciones de plantillas: {e}")
        return []

async def ask_llama_template_summary(template: dict, template_products: list[dict]) -> str:
    """Genera un resumen detallado de una plantilla de diseño"""
    try:
        products = [
            tp for tp in template_products 
            if tp.get('template_id') == template.get('id')
        ]

        prompt = f"""
Eres un experto en diseño de interiores. Crea una descripción atractiva para esta plantilla:

Nombre: {template.get('name', 'Plantilla')}
Descripción: {template.get('description', '')}
Tipo de Habitación: {template.get('room_type', '')}
Estilo: {template.get('style', '')}
Precio Total: ${template.get('total_price', 0):,.2f}
Descuento: {template.get('discount', 0)}%
Ventas: {template.get('sales_count', 0)} unidades vendidas

Productos incluidos:
{json.dumps([{
    'id': p.get('product_id'),
    'quantity': p.get('quantity', 1),
    'optional': p.get('is_optional', False),
    'notes': p.get('notes', '')
} for p in products], indent=2)}

REGLAS:
1. Responde SIEMPRE en español
2. Mantén el nombre de la plantilla exactamente como está
3. Sé preciso con los datos
4. Destaca el estilo y tipo de habitación
5. Menciona los productos más importantes
6. Máximo 3 oraciones
7. Usa un tono amigable y profesional
8. NO traduzcas los nombres de los productos
"""
        response = await ask_llama(prompt)
        return response if response.strip() else "No se pudo generar una descripción para esta plantilla."
        
    except Exception as e:
        logging.error(f"Error al generar resumen de plantilla: {str(e)}")
        return "No se pudo generar una descripción para esta plantilla."