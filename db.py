import asyncpg
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging

load_dotenv()

# Configuración del logging
logger = logging.getLogger('db')
logger.setLevel(logging.INFO)

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/your_database")

# Pool de conexiones
pool = None

async def init_db():
    """Inicializa el pool de conexiones a la base de datos"""
    global pool
    try:
        pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Conexión a la base de datos establecida")
    except Exception as e:
        logger.error(f"Error al conectar con la base de datos: {str(e)}")
        raise

async def close_db():
    """Cierra el pool de conexiones"""
    if pool:
        await pool.close()
        logger.info("Conexión a la base de datos cerrada")

async def get_db_connection():
    return await asyncpg.connect(DATABASE_URL)

async def get_all_products() -> List[Dict]:
    conn = await get_db_connection()
    try:
        rows = await conn.fetch("""
            SELECT 
                id,
                name,
                description,
                type,
                base_price,
                sku,
                slug,
                is_active,
                sales_count,
                stock_alert
            FROM "Product"
            WHERE is_active = true;
        """)
        return [dict(row) for row in rows]
    finally:
        await conn.close()

async def get_all_templates() -> List[Dict]:
    """Obtiene todas las plantillas de diseño activas"""
    try:
        conn = await get_db_connection()
        query = """
            SELECT 
                id, name, slug, description, cover_image_url,
                discount, room_type, style, total_price,
                is_active, sales_count, featured
            FROM "DesignTemplate"
            WHERE is_active = true AND deleted_at IS NULL
            ORDER BY sales_count DESC, created_at DESC
        """
        rows = await conn.fetch(query)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error al obtener plantillas: {str(e)}")
        return []
    finally:
        await conn.close()

async def get_all_template_products() -> List[Dict]:
    """Obtiene todos los productos asociados a plantillas"""
    try:
        conn = await get_db_connection()
        query = """
            SELECT 
                template_id, product_id, quantity,
                is_optional, notes
            FROM "DesignTemplateProduct"
        """
        rows = await conn.fetch(query)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error al obtener productos de plantillas: {str(e)}")
        return []
    finally:
        await conn.close()

async def get_template_by_id(template_id: str) -> Optional[Dict]:
    """Obtiene una plantilla específica por su ID"""
    try:
        async with pool.acquire() as conn:
            query = """
                SELECT 
                    id, name, slug, description, cover_image_url,
                    discount, room_type, style, total_price,
                    is_active, sales_count, featured
                FROM "DesignTemplate"
                WHERE id = $1 AND is_active = true AND deleted_at IS NULL
            """
            row = await conn.fetchrow(query, template_id)
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error al obtener plantilla por ID: {str(e)}")
        return None

async def get_template_products_by_template_id(template_id: str) -> List[Dict]:
    """Obtiene los productos asociados a una plantilla específica"""
    try:
        async with pool.acquire() as conn:
            query = """
                SELECT 
                    template_id, product_id, quantity,
                    is_optional, notes
                FROM "DesignTemplateProduct"
                WHERE template_id = $1
            """
            rows = await conn.fetch(query, template_id)
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error al obtener productos de plantilla: {str(e)}")
        return []
