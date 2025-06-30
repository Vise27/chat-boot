# Sistema de Conversaciones con Contexto

## Descripción
Este sistema permite mantener el contexto de las conversaciones por sesión, resolviendo el problema de "otros ejemplos" y mejorando la experiencia del usuario.

## ✅ **PROBLEMA RESUELTO**

**Antes (con el bug):**
```
Usuario: "tienes sillas??" → [Muestra sillas] + Guarda: "silla"
Usuario: "tienes otros ejemplos??" → [Muestra más sillas] ✅
Usuario: "tienes camas??" → [Muestra camas] + Guarda: "silla" ❌ (Bug)
Usuario: "tienes otros ejemplos??" → [Muestra sillas] ❌ (Problema)
```

**Ahora (corregido):**
```
Usuario: "tienes sillas??" → [Muestra sillas] + Guarda: "silla"
Usuario: "tienes otros ejemplos??" → [Muestra más sillas] ✅
Usuario: "tienes camas??" → [Muestra camas] + Guarda: "cama" ✅
Usuario: "tienes otros ejemplos??" → [Muestra más camas] ✅
```

## Características

### ✅ Funcionalidades Implementadas
- **Conversaciones por sesión**: Cada sesión mantiene su historial completo
- **Detección de continuación**: Detecta cuando el usuario pide "otros ejemplos"
- **Contexto inteligente**: Recuerda el tipo de producto de la consulta anterior
- **Limpieza automática**: Elimina conversaciones antiguas automáticamente
- **Persistencia JSON**: Guarda conversaciones en archivos JSON
- **Debugging**: Fácil revisión de conversaciones para debugging
- **Detección mejorada**: No confunde consultas específicas con continuaciones

### 🔧 Configuración
```python
# Configuración en main.py
CONVERSATIONS_DIR = "logs/conversations"      # Directorio de conversaciones
CLEANUP_INTERVAL = 30 * 60                   # Limpieza cada 30 minutos
SESSION_MAX_AGE = 2 * 60 * 60                # Sesiones expiran en 2 horas
MAX_SESSIONS = 1000                          # Máximo 1000 archivos de sesión
MAX_CONVERSATIONS_SIZE_MB = 50               # Máximo 50MB total
```

## Estructura de Archivos

```
logs/
├── conversations/
│   ├── a8c48ed5-2e92-4a42-8200-92dee800ea60.json
│   ├── b7c59fe6-3a03-5b53-9311-a3eff901fb71.json
│   └── ...
└── cleanup_log.txt
```

## Estructura de Conversación

```json
{
  "session_id": "a8c48ed5-2e92-4a42-8200-92dee800ea60",
  "created_at": "2025-06-18T16:30:00",
  "last_updated": "2025-06-18T16:35:00",
  "conversation": [
    {
      "timestamp": "2025-06-18T16:30:00",
      "user_message": "tienes sillas??",
      "query_type": "product",
      "product_type": "silla",
      "products_shown": ["27", "3", "8"],
      "response": "Encontré 3 sillas perfectas..."
    },
    {
      "timestamp": "2025-06-18T16:32:00",
      "user_message": "tienes otros ejemplos??",
      "query_type": "continuation",
      "product_type": "silla",
      "products_shown": ["1", "34", "11"],
      "response": "Aquí tienes más sillas..."
    }
  ]
}
```

## Cómo Funciona

### 1. Detección de Continuación Mejorada
El sistema detecta frases como:
- "tienes otros ejemplos??"
- "mas opciones"
- "tienes mas"
- "otros"
- "mas de lo mismo"

**IMPORTANTE**: Si la consulta menciona un producto específico (ej: "tienes camas??"), NO se considera continuación.

### 2. Contexto Inteligente
Cuando se detecta una consulta de continuación:
1. Busca el tipo de producto de la última consulta **específica** (no continuation)
2. Modifica la consulta para buscar el mismo tipo
3. Filtra productos ya mostrados
4. Devuelve nuevos productos del mismo tipo

### 3. Limpieza Automática
- **Por tiempo**: Sesiones > 2 horas se eliminan
- **Por cantidad**: Si hay > 1000 archivos, elimina los más antiguos
- **Por tamaño**: Si la carpeta > 50MB, reduce al 70%

## Ejemplo de Uso

```
Usuario: "tienes sillas??"
Sistema: [Muestra 3 sillas] + Guarda contexto: product_type="silla", query_type="product"

Usuario: "tienes otros ejemplos??"
Sistema: [Detecta continuación] + Busca más sillas + [Muestra 3 sillas más] + query_type="continuation"

Usuario: "tienes alguna cama??"
Sistema: [Muestra 3 camas] + Actualiza contexto: product_type="cama", query_type="product"

Usuario: "tienes otros ejemplos??"
Sistema: [Detecta continuación] + Busca más camas + [Muestra 3 camas más] + query_type="continuation"
```

## Mejoras Implementadas

### ✅ **Detección de Continuación Mejorada**
```python
def detect_continuation_query(user_query: str) -> bool:
    # Si la consulta menciona un producto específico, NO es continuación
    product_types = ["silla", "mesa", "sofá", "lampara", "mueble", "estantería", "cama", "armario"]
    for product_type in product_types:
        if product_type in normalized_query:
            return False  # Es una nueva consulta específica
```

### ✅ **Contexto Inteligente**
```python
def get_last_product_type(session_data: Dict) -> Optional[str]:
    # Buscar desde el final hacia el principio
    for message in reversed(conversation):
        # Si el mensaje tiene un tipo de producto específico (no continuation)
        if message.get("product_type") and message.get("query_type") != "continuation":
            return message["product_type"]
```

### ✅ **Determinación de Tipo de Producto**
```python
# Determinar el tipo de producto de manera más inteligente
product_type = None
if is_continuation:
    # Si es continuación, usar el tipo de la consulta anterior
    product_type = get_last_product_type(session_data)
else:
    # Si no es continuación, buscar en la consulta actual
    product_type = next((pt for pt in PRODUCT_TYPES if pt in user_query), None)
```

## Ventajas para Hosting

✅ **Ligero**: Solo archivos JSON pequeños
✅ **Sin DB**: No requiere base de datos adicional
✅ **Limpieza automática**: Se mantiene ligero
✅ **Debugging**: Fácil revisión de problemas
✅ **Escalable**: Funciona bien hasta miles de usuarios
✅ **Backup**: Fácil backup de conversaciones importantes
✅ **Contexto preciso**: No confunde consultas específicas con continuaciones

## Logs de Limpieza

El sistema registra todas las operaciones de limpieza:
```
2025-06-18 16:30:00 - INFO - Conversación eliminada por edad: a8c48ed5-2e92-4a42-8200-92dee800ea60.json
2025-06-18 16:30:00 - INFO - Limpieza completada: 5 conversaciones eliminadas
```

## Configuración Avanzada

Para modificar los tiempos de limpieza, edita las constantes en `main.py`:

```python
# Limpieza más frecuente (cada 15 minutos)
CLEANUP_INTERVAL = 15 * 60

# Sesiones más largas (4 horas)
SESSION_MAX_AGE = 4 * 60 * 60

# Más archivos permitidos
MAX_SESSIONS = 2000
```

## Pruebas

Ejecuta el script de prueba para verificar el funcionamiento:
```bash
python test_conversation_context.py
```

Este script simula una conversación completa y verifica que el contexto se mantenga correctamente. 