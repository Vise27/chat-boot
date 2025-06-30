# Asistente de Decoración Inteligente

Sistema de IA para encontrar productos de decoración y muebles con conversaciones inteligentes que mantienen el contexto.

## Características

- Búsqueda inteligente de productos por tipo y ambiente
- **Corrección automática de errores ortográficos**
- **Búsqueda fuzzy tolerante a errores de escritura**
- **Reconocimiento de sinónimos de productos**
- Conversaciones con contexto (recuerda productos mostrados)
- Detección de continuaciones ("tienes más ejemplos?")
- Limpieza automática de conversaciones antiguas
- Interfaz web con FastAPI
- Integración con GROQ API

## Instalación

1. **Clonar repositorio**
```bash
git clone https://github.com/Vise27/chat_boot.git
cd chat_boot
```

2. **Crear entorno virtual**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
Crear archivo `.env`:
```env
DATABASE_URL=postgresql://usuario:password@localhost:5432/nombre_db
GROQ_API_KEY=tu_api_key_de_groq
```

5. **Ejecutar servidor**
```bash
python -m uvicorn main:app --reload
```

## Uso

- Abrir `http://localhost:8000` en el navegador
- Hacer preguntas como "tienes sillas para oficina?"
- **El sistema corrige automáticamente errores ortográficos**
- **Funciona con sinónimos: "asientos", "tablas", "sillones"**
- Usar "tienes más ejemplos?" para continuar la búsqueda

## API

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "tienes silas?", "session_id": "123"}'
```

## Mejoras de Reconocimiento

### ✅ **Corrección de Errores Ortográficos**
- "sila" → "silla"
- "sofas" → "sofá" 
- "lamparas" → "lámpara"
- "mesas" → "mesa"

### ✅ **Sinónimos Reconocidos**
- "asientos" = "sillas"
- "tablas" = "mesas"
- "sillones" = "sofás"
- "luminarias" = "lámparas"

### ✅ **Búsqueda Fuzzy**
- Tolerante a errores de escritura
- Reconocimiento de palabras similares
- Múltiples estrategias de búsqueda

## Pruebas

Ejecuta el script de prueba para ver las mejoras:
```bash
python test_spelling_correction.py 