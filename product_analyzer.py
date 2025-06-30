import json
from typing import List, Dict, Any, Tuple
import re
from text_utils import normalize_text

class ProductAnalyzer:
    def __init__(self):
        # Palabras clave para ambientes
        self.ambiente_keywords = {
            "oficina": ["oficina", "trabajo", "escritorio", "laboral", "profesional"],
            "dormitorio": ["dormitorio", "habitación", "recámara", "cuarto", "cama"],
            "sala": ["sala", "living", "estar", "recibidor", "social"],
            "comedor": ["comedor", "dining", "mesa", "comida"],
            "cocina": ["cocina", "kitchen", "cooking", "preparación"],
            "baño": ["baño", "bathroom", "wc", "sanitario"],
            "exterior": ["exterior", "jardín", "terraza", "patio", "balcón"]
        }
        
        # Palabras clave para necesidades
        self.necesidad_keywords = {
            "iluminación": ["luz", "iluminación", "lámpara", "iluminar", "brillo"],
            "confort": ["confort", "comodidad", "ergonomía", "descanso"],
            "organización": ["organización", "almacenamiento", "orden", "guardar"],
            "decoración": ["decoración", "estético", "diseño", "estilo"],
            "funcionalidad": ["funcional", "práctico", "útil", "servicio"]
        }
        
        # Reglas de exclusión por ambiente
        self.exclusion_rules = {
            "oficina": ["cama", "colchón", "dormitorio", "recámara"],
            "dormitorio": ["oficina", "escritorio", "laboral"],
            "baño": ["sofá", "sillón", "mesa", "escritorio"],
            "cocina": ["sofá", "sillón", "cama", "colchón"],
            "exterior": ["sofá", "sillón", "cama", "colchón", "escritorio"]
        }
    
    def get_relevant_products(self, products: List[Dict], ambiente: str, necesidades: List[str]) -> List[Dict]:
        """
        Filtra y ordena productos según su relevancia para el ambiente y necesidades especificadas.
        """
        scored_products = []
        
        # Normalizar ambiente y necesidades
        ambiente = normalize_text(ambiente)
        necesidades = [normalize_text(n) for n in necesidades]
        
        for product in products:
            # Calcular puntuación base
            score = self._calculate_base_score(product, ambiente, necesidades)
            
            # Aplicar reglas de exclusión
            if self._should_exclude_product(product, ambiente):
                continue
            
            # Aplicar reglas de relevancia
            score = self._apply_relevance_rules(score, product, ambiente, necesidades)
            
            # Solo incluir productos con puntuación positiva
            if score > 0:
                product['relevance_score'] = score
                scored_products.append(product)
        
        # Ordenar por puntuación de relevancia
        scored_products.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_products
    
    def _calculate_base_score(self, product: Dict, ambiente: str, necesidades: List[str]) -> float:
        """
        Calcula la puntuación base de un producto basado en su descripción y características.
        """
        score = 0.0
        
        # Obtener texto normalizado
        description = normalize_text(product.get('description', ''))
        characteristics = normalize_text(product.get('characteristics', ''))
        
        # Verificar coincidencias en ambiente
        for keyword in self.ambiente_keywords.get(ambiente, []):
            if keyword in description or keyword in characteristics:
                score += 2.0
        
        # Verificar coincidencias en necesidades
        for necesidad in necesidades:
            for keyword in self.necesidad_keywords.get(necesidad, []):
                if keyword in description or keyword in characteristics:
                    score += 1.5
        
        return score
    
    def _should_exclude_product(self, product: Dict, ambiente: str) -> bool:
        """
        Verifica si un producto debe ser excluido basado en reglas específicas del ambiente.
        """
        if ambiente not in self.exclusion_rules:
            return False
            
        description = normalize_text(product.get('description', ''))
        characteristics = normalize_text(product.get('characteristics', ''))
        
        for keyword in self.exclusion_rules[ambiente]:
            if keyword in description or keyword in characteristics:
                return True
                
        return False
    
    def _apply_relevance_rules(self, score: float, product: Dict, ambiente: str, necesidades: List[str]) -> float:
        """
        Aplica reglas adicionales de relevancia para ajustar la puntuación.
        """
        # Ajustar por tipo de producto
        product_type = normalize_text(product.get('type', ''))
        
        # Reglas específicas por ambiente
        if ambiente == "oficina" and "escritorio" in product_type:
            score *= 1.5
        elif ambiente == "dormitorio" and "cama" in product_type:
            score *= 1.5
        elif ambiente == "sala" and "sofá" in product_type:
            score *= 1.5
            
        # Reglas específicas por necesidad
        for necesidad in necesidades:
            if necesidad == "iluminación" and "lámpara" in product_type:
                score *= 1.3
            elif necesidad == "organización" and "mueble" in product_type:
                score *= 1.3
                
        return score

    def detect_ambiente(self, query: str) -> str:
        """Detecta el ambiente basado en la consulta del usuario"""
        query = normalize_text(query)
        for ambiente, data in self.ambiente_keywords.items():
            if any(kw in query for kw in data):
                return ambiente
        return ""

    def calcular_relevancia(self, product: Dict, ambiente: str) -> Tuple[float, str]:
        """Calcula la relevancia de un producto para un ambiente específico"""
        if not ambiente or ambiente not in self.ambiente_keywords:
            return 0.0, ""

        keywords = self.ambiente_keywords[ambiente]
        nombre = normalize_text(product.get('name', '')).lower()
        descripcion = normalize_text(product.get('description', '')).lower()
        
        # Verificar exclusiones
        for exclusion in self.exclusion_rules.get(ambiente, []):
            if exclusion in nombre:
                return 0.0, ""
        
        # Calcular puntuación
        score = 0.0
        categoria_principal = "mobiliario_principal"  # Categoría por defecto
        
        # Verificar palabras clave en nombre y descripción
        for keyword in keywords:
            if keyword in nombre:
                score += 0.5
            if keyword in descripcion:
                score += 0.3
        
        # Determinar categoría basada en el tipo de producto
        tipo = normalize_text(product.get('type', '')).lower()
        if any(kw in tipo for kw in ['lámpara', 'luz', 'iluminación']):
            categoria_principal = "iluminacion"
        elif any(kw in tipo for kw in ['estantería', 'armario', 'cajonera']):
            categoria_principal = "organizacion"
        elif any(kw in tipo for kw in ['cuadro', 'alfombra', 'cortina']):
            categoria_principal = "decoracion"
        
        return score, categoria_principal

    def analizar_productos(self, products: List[Dict], ambiente: str) -> List[Dict]:
        """Analiza y filtra productos según su relevancia para el ambiente"""
        if not ambiente or ambiente not in self.ambiente_keywords:
            return products

        productos_analizados = []
        for product in products:
            score, categoria = self.calcular_relevancia(product, ambiente)
            if score > 0.3:  # Umbral de relevancia
                product['relevancia_score'] = score
                product['categoria'] = categoria
                productos_analizados.append(product)
        
        # Ordenar por relevancia
        productos_analizados.sort(key=lambda x: x['relevancia_score'], reverse=True)
        
        # Agrupar por categoría
        productos_agrupados = {}
        for product in productos_analizados:
            categoria = product['categoria']
            if categoria not in productos_agrupados:
                productos_agrupados[categoria] = []
            productos_agrupados[categoria].append(product)
        
        return productos_agrupados

    def generar_resumen(self, productos_agrupados: Dict[str, List[Dict]], ambiente: str) -> str:
        """Genera un resumen organizado de los productos"""
        if not productos_agrupados:
            return "No encontré productos relevantes para este ambiente."

        resumen = []
        
        # Mapeo de categorías a emojis
        emojis = {
            "mobiliario_principal": "🪑",
            "iluminacion": "💡",
            "organizacion": "🗄️",
            "decoracion": "🎨"
        }

        for categoria, productos in productos_agrupados.items():
            if not productos:
                continue
                
            # Obtener nombre de categoría en español
            nombre_categoria = categoria.replace('_', ' ').title()
            emoji = emojis.get(categoria, "📦")
            
            resumen.append(f"\n**{emoji} {nombre_categoria}**")
            
            for product in productos:
                nombre = product.get('name', '')
                precio = product.get('base_price', 0)
                resumen.append(f"* {nombre}: ${precio:.2f}")
            
            # Agregar descripción de la categoría
            keywords = self.ambiente_keywords[ambiente]
            resumen.append(f"\nProductos ideales para {ambiente} que ofrecen {', '.join(keywords)}.")

        return "\n".join(resumen) 