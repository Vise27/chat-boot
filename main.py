from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from db import get_all_products, get_all_templates, get_all_template_products, init_db, close_db
from llama_utils import (
    ask_llama_for_products, 
    ask_llama_summary_for_products,
    ask_llama_for_attributes_query,
    ask_llama_for_style_recommendations,
    ask_llama_for_template_recommendations,
    ask_llama_template_summary,
    ask_llama
)
from product_analyzer import ProductAnalyzer
from design_template_analyzer import DesignTemplateAnalyzer, generate_template_summary
import logging
from typing import Dict, List, Optional, Tuple
from text_utils import normalize_text, detect_query_type, normalize_color, improve_product_search, smart_product_search
import re
import json
import sys
from datetime import datetime, timedelta
import os
import time
from llama_sanitizer import sanitize_llama_response
import asyncio
import uuid
import glob
import threading


# Configuración de la aplicación
app = FastAPI(
    title="API de Asistente de Decoración",
    description="API para búsqueda inteligente de productos y plantillas de decoración",
    version="1.0.0"
)

# Configurar CORS para permitir conexiones desde cualquier frontend
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
allow_origins = [origin.strip() for origin in cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"--- Nueva petición ---")
    print(f"URL: {request.url}")
    print(f"Método: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        print(f"Body: {body.decode('utf-8')}")
    response = await call_next(request)
    print(f"--- Respuesta ---")
    print(f"Status code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    return response

# Configuración del logging
def setup_logging():
    logger = logging.getLogger('product_search')
    logger.setLevel(logging.INFO)
    
    # Formato del log
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo
    file_handler = logging.FileHandler(
        f'logs/product_search_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()
logging.basicConfig(level=logging.INFO)

# Tipos de productos disponibles (incluyendo singular y plural)
PRODUCT_TYPES = [
    "silla", "sillas", 
    "mesa", "mesas", 
    "sofá", "sofás", "sofa", "sofas",
    "lámpara", "lámparas", "lampara", "lamparas", 
    "mueble", "muebles", 
    "estantería", "estanterías", "estanteria", "estanterias",
    "cama", "camas", 
    "armario", "armarios"
]

# Directorio para conversaciones
CONVERSATIONS_DIR = "logs/conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

# Inicializar analizadores
product_analyzer = ProductAnalyzer()

# Configuración del sistema de conversaciones
CLEANUP_INTERVAL = 30 * 60  # 30 minutos
SESSION_MAX_AGE = 2 * 60 * 60  # 2 horas
MAX_SESSIONS = 1000
MAX_CONVERSATIONS_SIZE_MB = 50

class ProductSearchError(Exception):
    """Error personalizado para búsqueda de productos"""
    pass

async def search_products_by_type(products: List[dict], query: str) -> List[dict]:
    """Busca productos por tipo de producto mencionado"""
    mentioned_type = next((pt for pt in PRODUCT_TYPES if pt in query), None)
    if not mentioned_type:
        return products
    
    # Obtener nombres de productos para búsqueda inteligente
    product_names = [p.get('name', '') for p in products if isinstance(p, dict)]
    
    # Usar búsqueda inteligente que maneja singular/plural y sinónimos
    matched_names = smart_product_search(mentioned_type, product_names)
    
    if matched_names:
        # Encontrar los productos que coinciden con los nombres encontrados
        filtered_products = [
            p for p in products 
            if p.get('name', '') in matched_names
        ]
        logger.debug(f"Búsqueda por tipo inteligente '{mentioned_type}': {len(filtered_products)} productos encontrados")
        return filtered_products
    
    # Fallback a búsqueda original
    filtered_products = [
        p for p in products
        if mentioned_type in normalize_text(p.get('name', '')) or 
           mentioned_type in normalize_text(p.get('type', ''))
    ]
    
    logger.debug(f"Búsqueda por tipo '{mentioned_type}': {len(filtered_products)} productos encontrados")
    return filtered_products

async def search_products_by_query(products: List[dict], query: str) -> List[dict]:
    """Busca productos usando diferentes estrategias"""
    # Si el query está vacío o solo contiene números, no hacer búsqueda por texto
    if not query.strip() or query.strip().isdigit():
        logger.debug(f"Query vacío o solo números: '{query}', retornando lista vacía")
        return []
    
    query_words = query.split()
    
    # Filtrar palabras que son solo números (para evitar búsquedas por ID)
    query_words = [word for word in query_words if not word.isdigit()]
    
    # Si no quedan palabras después de filtrar números, no hacer búsqueda
    if not query_words:
        logger.debug(f"No quedan palabras válidas después de filtrar números")
        return []
    
    # Obtener nombres de productos para búsqueda inteligente
    product_names = [p.get('name', '') for p in products if isinstance(p, dict)]
    
    # Usar búsqueda inteligente que maneja singular/plural y sinónimos
    matched_names = smart_product_search(query, product_names)
    
    if matched_names:
        # Encontrar los productos que coinciden con los nombres encontrados
        matched_products = [
            p for p in products 
            if p.get('name', '') in matched_names
        ]
        logger.debug(f"Búsqueda inteligente: {len(matched_products)} productos encontrados")
        return matched_products
    
    # Fallback: búsqueda por palabras individuales
    word_matches = []
    for word in query_words:
        if len(word) > 2:  # Solo palabras de más de 2 caracteres
            for p in products:
                normalized_name = normalize_text(p.get('name', ''))
                if word in normalized_name:
                    word_matches.append(p)
    
    # Remover duplicados
    seen_ids = set()
    unique_matches = []
    for p in word_matches:
        if p.get('id') not in seen_ids:
            seen_ids.add(p.get('id'))
            unique_matches.append(p)
    
    logger.debug(f"Búsqueda por palabras: {len(unique_matches)} productos encontrados")
    return unique_matches

def extract_quantity_from_query(query: str) -> Tuple[int, str]:
    """Extrae la cantidad solicitada del query y retorna la cantidad y el query limpio"""
    # Patrones comunes para detectar cantidades
    patterns = [
        r'(\d+)\s*(?:opciones|productos|muebles|sillas|mesas|sofás|lamparas|muebles|estanterías|camas|armarios)',
        r'quiero\s*(\d+)',
        r'necesito\s*(\d+)',
        r'busco\s*(\d+)',
        r'mostrar\s*(\d+)',
        r'ver\s*(\d+)',
        r'dame\s*(\d+)',
        r'necesito\s*(\d+)',
        r'quiero\s*ver\s*(\d+)',
        r'mostrar\s*(\d+)',
        r'opciones\s*(\d+)',
        r'productos\s*(\d+)',
        r'unos?\s*(\d+)',  # "dame unos 5"
        r'algunos?\s*(\d+)',  # "dame algunos 5"
        r'(\d+)\s*(?:mas|más)',  # "5 más"
        r'(\d+)\s*(?:ejemplos|opciones|productos)',  # "5 ejemplos"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            quantity = int(match.group(1))
            # Remover la parte de la cantidad del query
            clean_query = re.sub(pattern, '', query.lower(), flags=re.IGNORECASE).strip()
            return quantity, clean_query
    
    # Si no se especifica cantidad, usar un valor por defecto entre 4 y 5
    return 4, query  # Valor por defecto

@app.on_event("startup")
async def startup_event():
    """Inicializa la base de datos al arrancar la aplicación"""
    await init_db()
    # Iniciar el programador de limpieza automática
    start_cleanup_scheduler()

@app.on_event("shutdown")
async def shutdown_event():
    """Cierra la conexión a la base de datos al detener la aplicación"""
    await close_db()

@app.post("/chat")
async def chat(request: Request):
    try:
        # 1. Obtener y validar datos de entrada
        data = await request.json()
        raw_query = data.get("message", "").strip()
        session_id = data.get("session_id", "")  # ID de sesión para el historial
        # last_products = data.get("last_products", [])  # Productos mostrados anteriormente

        if not raw_query:
            logger.warning("Intento de búsqueda con mensaje vacío")
            raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")

        # 2. Cargar o crear conversación de sesión
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session_data = load_conversation(session_id)

        # Obtener productos ya mostrados: si el frontend los manda, úsalos; si no, reconstruye desde el historial
        if "last_products" in data:
            last_products = data.get("last_products", [])
        else:
            last_products = get_all_shown_products(session_data)
        
        # 3. Normalizar consulta y extraer cantidad
        requested_quantity, clean_query = extract_quantity_from_query(raw_query)
        user_query = normalize_text(clean_query)
        
        # 4. Detectar si es una consulta de continuación
        is_continuation = detect_continuation_query(user_query)
        if is_continuation:
            # Obtener el tipo de producto de la consulta anterior
            last_product_type = get_last_product_type(session_data)
            if last_product_type:
                # Modificar la consulta para buscar el mismo tipo de producto
                user_query = last_product_type
                logger.info(f"Consulta de continuación detectada. Buscando más: {last_product_type}")
        
        query_type = detect_query_type(user_query)
        
        # Determinar el tipo de producto de manera más inteligente
        product_type = None
        if is_continuation:
            # Si es continuación, usar el tipo de la consulta anterior
            product_type = get_last_product_type(session_data)
        else:
            # Si no es continuación, buscar en la consulta actual
            product_type = next((pt for pt in PRODUCT_TYPES if pt in user_query), None)
        
        logger.info(f"Búsqueda iniciada - Tipo: {query_type} | Original: '{raw_query}' | Normalizada: '{user_query}' | Continuación: {is_continuation} | Product Type: {product_type}")

        # 5. Obtener datos
        all_products = await get_all_products()
        all_templates = await get_all_templates()
        all_template_products = await get_all_template_products()
        
        logger.info(f"Total productos obtenidos: {len(all_products)}")
        logger.info(f"Total plantillas obtenidas: {len(all_templates)}")

        # 6. Detectar si la consulta es sobre plantillas
        template_keywords = ["plantilla", "diseño", "decoración", "estilo", "conjunto", "pack"]
        is_template_query = any(kw in user_query for kw in template_keywords)

        if is_template_query:
            # Buscar plantillas relevantes
            matched_templates = await ask_llama_for_template_recommendations(
                all_templates,
                all_template_products,
                user_query
            )
            
            # Limitar cantidad
            available_quantity = min(requested_quantity, len(matched_templates))
            templates_to_show = matched_templates[:available_quantity]

            # Generar resumen usando all_products
            response_text = ""
            for template in templates_to_show:
                template_summary = generate_template_summary(template, all_template_products, all_products)
                response_text += f"\n{template_summary}\n"

            # Preparar mensaje de cantidad
            quantity_message = ""
            if available_quantity < requested_quantity:
                if available_quantity == 0:
                    quantity_message = "\n\nLo siento, no encontré plantillas que coincidan con tu búsqueda."
                elif available_quantity == 1:
                    quantity_message = f"\n\nSolo encontré 1 plantilla que coincide con tu búsqueda."
                else:
                    quantity_message = f"\n\nSolo encontré {available_quantity} plantillas que coinciden con tu búsqueda de {requested_quantity} solicitadas."

            response_text += quantity_message

            # Guardar en conversación
            add_message_to_conversation(
                session_data, raw_query, "template", 
                product_type=product_type,
                response=response_text
            )

            return {
                "response": response_text,
                "templates": [{
                    "id": str(t.get("id")),
                    "name": t.get("name"),
                    "description": t.get("description"),
                    "room_type": t.get("room_type"),
                    "style": t.get("style"),
                    "total_price": t.get("total_price"),
                    "discount": t.get("discount")
                } for t in templates_to_show],
                "query_type": "template",
                "requested_quantity": requested_quantity,
                "available_quantity": available_quantity,
                "session_id": session_id
            }

        # 7. Si no es consulta de plantillas, continuar con búsqueda normal de productos
        # Detectar ambiente y analizar productos
        ambiente = product_analyzer.detect_ambiente(user_query)
        if ambiente:
            productos_agrupados = product_analyzer.analizar_productos(all_products, ambiente)
            response_text = product_analyzer.generar_resumen(productos_agrupados, ambiente)
            
            # Preparar lista plana de productos para la respuesta
            productos_planos = []
            for categoria, productos in productos_agrupados.items():
                productos_planos.extend(productos)
            
            # Filtrar productos ya mostrados
            productos_planos = [p for p in productos_planos if str(p.get("id")) not in last_products]
            
            # Limitar a la cantidad solicitada
            productos_planos = productos_planos[:requested_quantity]
            
            # Preparar mensaje de cantidad
            quantity_message = ""
            if len(productos_planos) < requested_quantity:
                if len(productos_planos) == 0:
                    quantity_message = "\n\nLo siento, no encontré más productos que coincidan con tu búsqueda."
                elif len(productos_planos) == 1:
                    quantity_message = f"\n\nSolo encontré 1 producto más que coincide con tu búsqueda."
                else:
                    quantity_message = f"\n\nSolo encontré {len(productos_planos)} productos más que coinciden con tu búsqueda de {requested_quantity} solicitados."
            
            response_text += quantity_message
            
            # Actualizar lista de productos mostrados
            productos_mostrados = last_products + [str(p.get("id")) for p in productos_planos]
            
            # Guardar en conversación
            add_message_to_conversation(
                session_data, raw_query, query_type, 
                product_type=product_type,
                products_shown=[str(p.get("id")) for p in productos_planos],
                response=response_text
            )
            
            return {
                "response": response_text,
                "products": [{
                    "id": str(p.get("id")),
                    "sku": p.get("sku"),
                    "slug": p.get("slug"),
                    "name": p.get("name"),
                    "description": p.get("description")
                } for p in productos_planos],
                "query_type": query_type,
                "mentioned_product_type": product_type,
                "requested_quantity": requested_quantity,
                "available_quantity": len(productos_planos),
                "session_id": session_id,
                "last_products": productos_mostrados
            }

        # 8. Si no se detectó ambiente, usar búsqueda normal
        matched_products = []
        try:
            if query_type == "attributes":
                logger.info("Iniciando búsqueda por atributos")
                matched_products = await ask_llama_for_attributes_query(all_products, user_query)
                logger.info(f"Búsqueda por atributos completada: {len(matched_products)} productos encontrados")
            else:
                # Si es una consulta de continuación y el query está vacío, buscar por tipo de producto
                if is_continuation and (not user_query.strip() or user_query.strip().isdigit()):
                    logger.info("Consulta de continuación con query vacío, buscando por tipo de producto")
                    if product_type:
                        matched_products = await search_products_by_type(all_products, product_type)
                        logger.info(f"Búsqueda por tipo '{product_type}' completada: {len(matched_products)} productos encontrados")
                else:
                    # Búsqueda por texto
                    logger.info("Iniciando búsqueda por texto")
                    matched_products = await search_products_by_query(all_products, user_query)
                    
                    # Si no hay resultados, intentar búsqueda semántica
                    if not matched_products:
                        logger.info("Iniciando búsqueda semántica")
                        product_names = await ask_llama_for_products(all_products, user_query)
                        matched_products = [
                            p for p in all_products 
                            if normalize_text(p.get('name', '')) in 
                            [normalize_text(n) for n in product_names]
                        ]
                        logger.info(f"Búsqueda semántica completada: {len(matched_products)} productos encontrados")
                    
                    # Si aún no hay resultados, buscar por estilo
                    if not matched_products:
                        logger.info("Iniciando búsqueda por estilo")
                        style_products = await ask_llama_for_style_recommendations(all_products, user_query)
                        matched_products = [
                            p for p in all_products 
                            if p['name'] in style_products
                        ]
                        logger.info(f"Búsqueda por estilo completada: {len(matched_products)} productos encontrados")
        except Exception as e:
            logger.error(f"Error en búsqueda principal: {str(e)}")
            raise ProductSearchError("Error al buscar productos")

        # Filtrar productos ya mostrados
        matched_products = [p for p in matched_products if str(p.get("id")) not in last_products]

        # 9. Limitar y responder
        available_quantity = min(requested_quantity, len(matched_products))
        products_to_show = matched_products[:available_quantity]

        # Preparar mensaje de cantidad
        quantity_message = ""
        if available_quantity < requested_quantity:
            if available_quantity == 0:
                quantity_message = "\n\nLo siento, no encontré más productos que coincidan con tu búsqueda."
            elif available_quantity == 1:
                quantity_message = f"\n\nSolo encontré 1 producto más que coincide con tu búsqueda."
            else:
                quantity_message = f"\n\nSolo encontré {available_quantity} productos más que coinciden con tu búsqueda de {requested_quantity} solicitados."

        response_text = await ask_llama_summary_for_products(products_to_show) + quantity_message
        logger.info(f"Búsqueda exitosa - {len(products_to_show)} productos mostrados de {requested_quantity} solicitados")

        # Actualizar lista de productos mostrados
        productos_mostrados = last_products + [str(p.get("id")) for p in products_to_show]
        
        # Guardar en conversación
        add_message_to_conversation(
            session_data, raw_query, query_type, 
            product_type=product_type,
            products_shown=[str(p.get("id")) for p in products_to_show],
            response=response_text
        )

        return {
            "response": response_text,
            "products": [{
                "id": str(p.get("id")),
                "sku": p.get("sku"),
                "slug": p.get("slug"),
                "name": p.get("name"),
                "description": p.get("description")
            } for p in products_to_show],
            "query_type": query_type,
            "mentioned_product_type": product_type,
            "requested_quantity": requested_quantity,
            "available_quantity": available_quantity,
            "session_id": session_id,
            "last_products": productos_mostrados
        }

    except ProductSearchError as e:
        logger.error(f"Error en búsqueda de productos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error inesperado en /chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error al procesar tu solicitud")

@app.get("/health")
async def health_check():
    """Endpoint de health check para verificar que la API está funcionando"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Asistente de Decoración",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat - POST - Búsqueda de productos y plantillas",
            "health": "/health - GET - Estado de la API"
        },
        "documentation": "/docs - Documentación automática de la API"
    }

def load_conversation(session_id: str) -> Dict:
    """Carga la conversación de una sesión desde archivo"""
    try:
        file_path = os.path.join(CONVERSATIONS_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar conversación {session_id}: {e}")
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "conversation": []
    }

def save_conversation(session_data: Dict):
    """Guarda la conversación de una sesión en archivo"""
    try:
        session_id = session_data["session_id"]
        session_data["last_updated"] = datetime.now().isoformat()
        file_path = os.path.join(CONVERSATIONS_DIR, f"{session_id}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error al guardar conversación {session_data.get('session_id')}: {e}")

def add_message_to_conversation(session_data: Dict, user_message: str, query_type: str, 
                               product_type: str = None, products_shown: List[str] = None, 
                               response: str = None):
    """Agrega un mensaje a la conversación"""
    # Determinar si es una consulta de continuación
    is_continuation = detect_continuation_query(user_message)
    
    message = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "query_type": "continuation" if is_continuation else query_type,
        "product_type": product_type,
        "products_shown": products_shown or [],
        "response": response
    }
    session_data["conversation"].append(message)
    save_conversation(session_data)

def detect_continuation_query(user_query: str) -> bool:
    """Detecta si la consulta es una continuación de la conversación anterior"""
    continuation_keywords = [
        "otros ejemplos", "mas ejemplos", "tienes mas", "mas opciones",
        "otros", "mas", "continuar", "siguiente", "mas de lo mismo",
        "tienes otros", "hay mas", "muestrame mas", "dame mas",
        "mas productos", "otras opciones", "mas variedad", "mas alternativas",
        "solo tienes esos", "no hay mas", "esos son todos", "es todo"
    ]
    
    # Normalizar la consulta
    normalized_query = normalize_text(user_query)
    
    # Si la consulta menciona un producto específico, NO es continuación
    product_types = ["silla", "mesa", "sofá", "lampara", "mueble", "estantería", "cama", "armario"]
    for product_type in product_types:
        if product_type in normalized_query:
            return False  # Es una nueva consulta específica
    
    # Verificar si contiene palabras clave de continuación
    for keyword in continuation_keywords:
        if keyword in normalized_query:
            return True
    
    # Verificar patrones específicos
    continuation_patterns = [
        r'\b(otros?|mas?)\s+(ejemplos?|opciones?|productos?)\b',
        r'\b(tienes?|hay)\s+(mas?|otros?)\b',
        r'\b(muestrame?|dame?)\s+(mas?|otros?)\b',
        r'\b(solo|unicamente?)\s+(tienes?|hay)\s+(esos?|estos?)\b',
        r'\b(esos?|estos?)\s+(son|es)\s+(todos?|todo)\b'
    ]
    
    for pattern in continuation_patterns:
        if re.search(pattern, normalized_query):
            return True
    
    return False

def get_last_product_type(session_data: Dict) -> Optional[str]:
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

def cleanup_old_conversations():
    """Limpia conversaciones antiguas y excesivas"""
    try:
        current_time = datetime.now()
        deleted_count = 0
        total_size = 0
        
        # Obtener todos los archivos de conversación
        conversation_files = glob.glob(os.path.join(CONVERSATIONS_DIR, "*.json"))
        
        for file_path in conversation_files:
            try:
                # Verificar edad del archivo
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_seconds = (current_time - file_time).total_seconds()
                
                # Borrar si es muy antiguo
                if age_seconds > SESSION_MAX_AGE:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"Conversación eliminada por edad: {os.path.basename(file_path)}")
                    continue
                
                # Calcular tamaño total
                total_size += os.path.getsize(file_path)
                
            except Exception as e:
                logger.error(f"Error procesando archivo {file_path}: {e}")
        
        # Si hay demasiados archivos, borrar los más antiguos
        if len(conversation_files) > MAX_SESSIONS:
            files_with_time = []
            for file_path in conversation_files:
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    files_with_time.append((file_path, file_time))
                except:
                    continue
            
            # Ordenar por tiempo y borrar los más antiguos
            files_with_time.sort(key=lambda x: x[1])
            files_to_delete = files_with_time[:len(files_with_time) - MAX_SESSIONS]
            
            for file_path, _ in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"Conversación eliminada por límite: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error eliminando archivo {file_path}: {e}")
        
        # Si el tamaño total es muy grande, borrar archivos antiguos
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > MAX_CONVERSATIONS_SIZE_MB:
            files_with_time = []
            for file_path in conversation_files:
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    files_with_time.append((file_path, file_time))
                except:
                    continue
            
            files_with_time.sort(key=lambda x: x[1])
            target_size = MAX_CONVERSATIONS_SIZE_MB * 0.7  # Reducir al 70%
            
            for file_path, _ in files_with_time:
                if total_size_mb <= target_size:
                    break
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    total_size_mb -= file_size / (1024 * 1024)
                    deleted_count += 1
                    logger.info(f"Conversación eliminada por tamaño: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error eliminando archivo {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Limpieza completada: {deleted_count} conversaciones eliminadas")
            
    except Exception as e:
        logger.error(f"Error en limpieza de conversaciones: {e}")

def start_cleanup_scheduler():
    """Inicia el programador de limpieza automática"""
    def cleanup_loop():
        while True:
            try:
                cleanup_old_conversations()
                time.sleep(CLEANUP_INTERVAL)
            except Exception as e:
                logger.error(f"Error en loop de limpieza: {e}")
                time.sleep(60)  # Esperar 1 minuto antes de reintentar
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Programador de limpieza automática iniciado")

def get_all_shown_products(session_data: dict) -> list:
    """Devuelve una lista de todos los IDs de productos ya mostrados en la conversación."""
    shown = []
    for message in session_data.get("conversation", []):
        shown.extend([str(pid) for pid in message.get("products_shown", [])])
    # Elimina duplicados manteniendo el orden
    return list(dict.fromkeys(shown))

if __name__ == "__main__":
    asyncio.run(main())
