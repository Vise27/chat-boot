import json
import re
from typing import Union, List

def sanitize_llama_response(response: str, expected_type: str = "list_int") -> Union[List[int], List[str]]:
    """
    Versión mejorada del sanitizador que maneja más casos edge
    """
    # Limpieza inicial
    response = response.strip().replace('\\"', '"').replace("'", '"')
    
    # Caso especial para respuestas como {"1, 8":null}
    if response.startswith('{') and expected_type == "list_int":
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                keys = list(data.keys())
                if len(keys) == 1 and ',' in keys[0]:
                    return [int(x.strip()) for x in keys[0].split(',') if x.strip().isdigit()]
                return [int(k) for k in keys if str(k).isdigit()]
        except:
            pass

    # Intento estándar de parseo JSON
    try:
        parsed = json.loads(response)
        if expected_type == "list_int":
            if isinstance(parsed, dict):
                return [int(k) for k in parsed.keys() if str(k).isdigit()]
            elif isinstance(parsed, list):
                return [int(x) for x in parsed if str(x).isdigit()]
        elif expected_type == "list_str":
            if isinstance(parsed, dict):
                return [str(k) for k in parsed.keys() if k]
            elif isinstance(parsed, list):
                return [str(x) for x in parsed if x]
    except json.JSONDecodeError:
        pass

    # Extracción mediante regex para respuestas mal formadas
    lista_match = re.search(r'\[.*?\]', response, re.DOTALL)
    if lista_match:
        try:
            snippet = lista_match.group(0)
            snippet = re.sub(r',\s*]', ']', snippet)
            parsed = json.loads(snippet)
            if expected_type == "list_int":
                return [int(x) for x in parsed if str(x).isdigit()]
            return [str(x) for x in parsed if x]
        except:
            pass

    # Último intento para listas de strings
    if expected_type == "list_str":
        keys = re.findall(r'"([^"]+)"', response) or re.findall(r"'([^']+)'", response)
        if keys:
            return keys

    return []