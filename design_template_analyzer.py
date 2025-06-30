from typing import List, Dict, Optional
import json
import logging
from datetime import datetime

class DesignTemplateAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('design_template_analyzer')
        
    def get_templates_by_room_type(self, templates: List[Dict], room_type: str) -> List[Dict]:
        """Filtra plantillas por tipo de habitación"""
        return [
            template for template in templates 
            if template.get('room_type', '').lower() == room_type.lower()
        ]
    
    def get_templates_by_style(self, templates: List[Dict], style: str) -> List[Dict]:
        """Filtra plantillas por estilo"""
        return [
            template for template in templates 
            if template.get('style', '').lower() == style.lower()
        ]
    
    def get_featured_templates(self, templates: List[Dict]) -> List[Dict]:
        """Obtiene plantillas destacadas"""
        return [
            template for template in templates 
            if template.get('featured', False)
        ]
    
    def get_active_templates(self, templates: List[Dict]) -> List[Dict]:
        """Obtiene plantillas activas"""
        return [
            template for template in templates 
            if template.get('is_active', True) and not template.get('deleted_at')
        ]
    
    def get_templates_by_price_range(self, templates: List[Dict], min_price: float, max_price: float) -> List[Dict]:
        """Filtra plantillas por rango de precio"""
        return [
            template for template in templates 
            if min_price <= template.get('total_price', 0) <= max_price
        ]
    
    def get_templates_with_discount(self, templates: List[Dict]) -> List[Dict]:
        """Obtiene plantillas con descuento"""
        return [
            template for template in templates 
            if template.get('discount', 0) > 0
        ]
    
    def get_templates_by_popularity(self, templates: List[Dict], limit: Optional[int] = None) -> List[Dict]:
        """Obtiene las plantillas más populares basadas en ventas"""
        sorted_templates = sorted(
            templates,
            key=lambda x: x.get('sales_count', 0),
            reverse=True
        )
        return sorted_templates[:limit] if limit else sorted_templates
    
    def get_template_products(self, template: Dict, template_products: List[Dict]) -> List[Dict]:
        """Obtiene los productos asociados a una plantilla"""
        return [
            tp for tp in template_products 
            if tp.get('template_id') == template.get('id')
        ]
    
    def get_recommended_templates(self, templates: List[Dict], room_type: str, style: Optional[str] = None) -> List[Dict]:
        """Obtiene plantillas recomendadas basadas en tipo de habitación y estilo"""
        filtered_templates = self.get_templates_by_room_type(templates, room_type)
        
        if style:
            filtered_templates = self.get_templates_by_style(filtered_templates, style)
        
        # Ordenar por popularidad y precio
        sorted_templates = sorted(
            filtered_templates,
            key=lambda x: (x.get('sales_count', 0), -x.get('total_price', 0)),
            reverse=True
        )
        return sorted_templates 

# --- FUNCIÓN GLOBAL, FUERA DE LA CLASE ---
def generate_template_summary(template: Dict, template_products: List[Dict], all_products: List[Dict]) -> str:
    """Genera un resumen detallado de una plantilla"""
    logging.info(f"Generando resumen para plantilla: {template.get('name')} - ID: {template.get('id')}")
    summary = f"Plantilla: {template['name']}\nDescripción: {template.get('description', 'Sin descripción')}\nProductos incluidos:\n"
    template_id = str(template.get('id'))
    
    # Filtrar productos de la plantilla
    template_products_filtered = [
        tp for tp in template_products 
        if str(tp.get('template_id')) == template_id
    ]
    
    logging.info(f"Productos filtrados para plantilla {template_id}: {len(template_products_filtered)}")
    
    for tp in template_products_filtered:
        # Buscar el producto en la lista de productos
        product = next(
            (p for p in all_products if str(p.get('id')) == str(tp.get('product_id'))), 
            None
        )
        if product:
            logging.info(f"Producto encontrado: {product.get('product_name')} - ID: {product.get('id')}")
            optional = " [Opcional]" if tp.get('is_optional') else ""
            notes = f" ({tp['notes']})" if tp.get('notes') else ""
            summary += f"- {product.get('product_name', 'Producto sin nombre')} x{tp.get('quantity', 1)}{notes}{optional}\n"
        else:
            logging.warning(f"Producto no encontrado para ID: {tp.get('product_id')}")
    
    return summary 