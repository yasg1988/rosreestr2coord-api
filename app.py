"""
Rosreestr2Coord API Server v3.1.1
Полная информация о земельных участках и ОКС по кадастровому номеру
+ Список объектов в кадастровом квартале
"""

from fastapi import FastAPI, HTTPException, Header, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib
import json
import os
import time
import requests
import re
import urllib3

from rosreestr2coord.parser import Area

# Отключаем предупреждения SSL (ПКК использует российские сертификаты)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(
    title="Rosreestr2Coord API",
    description="API для получения полной информации об объектах недвижимости по кадастровому номеру",
    version="3.1.1"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Авторизация
API_TOKEN = os.environ.get("API_TOKEN", "rr2c_api_7f8a9b3c4d5e6f")

# ПКК API URL
PKK_API_URL = "https://pkk.rosreestr.ru/api/features"

# Типы объектов в НСПД (обновлённые)
AREA_TYPES = {
    1: "Объекты недвижимости (ЗУ, здания, сооружения, ОНС)",
    2: "Кадастровое деление (округа, районы, кварталы)",
    4: "Административно-территориальное деление (МО, населённые пункты)",
    5: "Зоны и территории (ОКН, ЗОУИТ, ООПТ, лесничества)",
    7: "Территориальные зоны",
    15: "Комплексы объектов (ЕНК, предприятия)"
}

# Категории земель
LAND_CATEGORIES = {
    "003001000000": "Земли сельскохозяйственного назначения",
    "003002000000": "Земли населённых пунктов",
    "003003000000": "Земли промышленности",
    "003004000000": "Земли особо охраняемых территорий",
    "003005000000": "Земли лесного фонда",
    "003006000000": "Земли водного фонда",
    "003007000000": "Земли запаса"
}

# Системы координат
COORD_SYSTEMS = {
    "4326": "WGS 84 (широта/долгота)",
    "3857": "Web Mercator (метры)"
}

# === КЭШИРОВАНИЕ ===
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 3600  # 1 час


def get_cache_key(cadastral_number: str, area_type: int, center_only: bool) -> str:
    """Генерирует ключ кэша"""
    return hashlib.md5(f"{cadastral_number}:{area_type}:{center_only}".encode()).hexdigest()


def get_from_cache(key: str) -> Optional[dict]:
    """Получает данные из кэша"""
    if key in CACHE:
        entry = CACHE[key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return entry["data"]
        else:
            del CACHE[key]
    return None


def set_cache(key: str, data: dict):
    """Сохраняет данные в кэш"""
    CACHE[key] = {"data": data, "timestamp": time.time()}
    # Очистка старых записей (макс 1000)
    if len(CACHE) > 1000:
        oldest = min(CACHE.items(), key=lambda x: x[1]["timestamp"])
        del CACHE[oldest[0]]


# === МОДЕЛИ ===
class CadastralRequest(BaseModel):
    cadastral_number: str
    area_type: int = 1
    center_only: bool = False
    coord_out: str = "4326"


class BatchRequest(BaseModel):
    cadastral_numbers: List[str]
    area_type: int = 1
    center_only: bool = False
    coord_out: str = "4326"


class SearchRequest(BaseModel):
    query: str
    area_type: int = 1
    limit: int = 10


# === УТИЛИТЫ ===
def verify_token(authorization: Optional[str] = Header(None)):
    """Проверка токена авторизации"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


def parse_area_value(options: dict) -> Optional[float]:
    """Парсит площадь из разных форматов"""
    for key in ["area_value", "build_record_area", "sum_land_area"]:
        val = options.get(key)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def format_cadastral_cost(options: dict) -> Optional[Dict[str, Any]]:
    """Форматирует кадастровую стоимость"""
    cost = options.get("cad_cost") or options.get("cost_value") or options.get("cost_value_total_geom")
    if cost:
        return {
            "value": float(cost) if cost else None,
            "unit": options.get("cad_unit", "руб."),
            "date": options.get("date_cost") or options.get("cost_determination_date", "")
        }
    return None


def get_category_name(category_code: str) -> str:
    """Получает название категории земель по коду"""
    return LAND_CATEGORIES.get(category_code, category_code or "Не указана")


def extract_center(geometry: dict) -> Optional[Dict[str, float]]:
    """Извлекает центр из геометрии"""
    coords = geometry.get("coordinates", [])
    if not coords:
        return None

    geo_type = geometry.get("type", "")

    if geo_type == "Point":
        return {"lon": coords[0], "lat": coords[1]}
    elif geo_type == "Polygon" and coords:
        # Центр bounding box
        all_points = coords[0] if coords else []
        if all_points:
            lons = [p[0] for p in all_points]
            lats = [p[1] for p in all_points]
            return {
                "lon": (min(lons) + max(lons)) / 2,
                "lat": (min(lats) + max(lats)) / 2
            }
    elif geo_type == "MultiPolygon" and coords:
        all_points = []
        for polygon in coords:
            if polygon:
                all_points.extend(polygon[0])
        if all_points:
            lons = [p[0] for p in all_points]
            lats = [p[1] for p in all_points]
            return {
                "lon": (min(lons) + max(lons)) / 2,
                "lat": (min(lats) + max(lats)) / 2
            }
    return None


def extract_full_info(options: dict, area_type: int = 1) -> Dict[str, Any]:
    """Извлекает полную информацию из options"""
    # Универсальные поля
    info = {
        "cadastral_number": options.get("cn") or options.get("cad_num", ""),
        "object_type": options.get("obj_kind_value") or AREA_TYPES.get(area_type, f"Тип {area_type}"),
        "address": options.get("address") or options.get("readable_address", ""),
        "status": options.get("statecd") or options.get("status", ""),
    }

    # Площадь
    area_value = parse_area_value(options)
    if area_value:
        info["area"] = {
            "value": area_value,
            "unit": options.get("area_unit", "кв.м")
        }

    # Кадастровая стоимость
    cad_cost = format_cadastral_cost(options)
    if cad_cost:
        info["cadastral_cost"] = cad_cost

    # Для земельных участков
    if area_type == 1:
        category = options.get("category_type", "")
        if category:
            info["category"] = {"code": category, "name": get_category_name(category)}
        permitted = options.get("util_by_doc") or options.get("permitted_use_name", "")
        if permitted:
            info["permitted_use"] = {"by_document": permitted}

    # Для ОКС/зданий
    building_name = options.get("name") or options.get("building_name", "")
    if building_name or options.get("floors"):
        info["building"] = {
            "name": building_name,
            "purpose": options.get("purpose", ""),
            "floors": options.get("floors", ""),
            "underground_floors": options.get("underground_floors", ""),
            "year_built": options.get("year_built", ""),
            "walls_material": options.get("walls") or options.get("materials", "")
        }

    # Для кадастрового деления
    if options.get("cnt_land"):
        info["statistics"] = {
            "land_plots": options.get("cnt_land", 0),
            "land_plots_with_geometry": options.get("cnt_land_geom", 0),
            "oks_count": options.get("cnt_oks", 0),
            "oks_with_geometry": options.get("cnt_oks_geom", 0),
            "total_land_area": options.get("sum_land_area", 0),
            "total_cost": options.get("cost_value_total_geom", 0)
        }

    # Даты
    dates = {}
    if options.get("cad_record_date") or options.get("build_record_registration_date"):
        dates["registration"] = options.get("cad_record_date") or options.get("build_record_registration_date", "")
    if options.get("adate") or options.get("date_ch"):
        dates["data_update"] = options.get("adate") or options.get("date_ch", "")
    if dates:
        info["dates"] = dates

    # Права
    if options.get("rights_reg") or options.get("right_type"):
        info["rights"] = {
            "registered": bool(options.get("rights_reg")),
            "type": options.get("right_type") or options.get("ownership_type", "")
        }

    # Кадастровый квартал
    quarter = options.get("kvartal_cn") or options.get("quarter_cad_number", "")
    if quarter:
        info["cadastral_quarter"] = quarter

    # Все оригинальные данные
    info["raw_options"] = options
    return info


def geometry_to_kml(geometry: dict, name: str = "", description: str = "") -> str:
    """Конвертирует GeoJSON геометрию в KML"""
    geo_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    kml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'<name>{name}</name>',
        f'<description>{description}</description>',
        '<Placemark>',
        f'<name>{name}</name>'
    ]

    if geo_type == "Point":
        kml_parts.append(f'<Point><coordinates>{coords[0]},{coords[1]},0</coordinates></Point>')
    elif geo_type == "Polygon":
        kml_parts.append('<Polygon><outerBoundaryIs><LinearRing><coordinates>')
        coord_str = ' '.join([f'{p[0]},{p[1]},0' for p in coords[0]])
        kml_parts.append(coord_str)
        kml_parts.append('</coordinates></LinearRing></outerBoundaryIs></Polygon>')
    elif geo_type == "MultiPolygon":
        kml_parts.append('<MultiGeometry>')
        for polygon in coords:
            kml_parts.append('<Polygon><outerBoundaryIs><LinearRing><coordinates>')
            coord_str = ' '.join([f'{p[0]},{p[1]},0' for p in polygon[0]])
            kml_parts.append(coord_str)
            kml_parts.append('</coordinates></LinearRing></outerBoundaryIs></Polygon>')
        kml_parts.append('</MultiGeometry>')

    kml_parts.extend(['</Placemark>', '</Document>', '</kml>'])
    return '\n'.join(kml_parts)


def is_cadastral_quarter(cn: str) -> bool:
    """Проверяет, является ли номер кадастровым кварталом (без номера участка)"""
    # Формат квартала: XX:XX:XXXXXXX (без :XXX в конце)
    pattern = r'^\d{2}:\d{2}:\d{6,7}$'
    return bool(re.match(pattern, cn))


def extract_settlement_from_address(address: str) -> Optional[str]:
    """Извлекает название населённого пункта из адреса"""
    if not address:
        return None

    # Паттерны для извлечения НП
    patterns = [
        r'(?:г\.|город)\s*([^,]+)',  # г. Йошкар-Ола
        r'(?:д\.|деревня)\s*([^,]+)',  # д. Апшакбеляк
        r'(?:с\.|село)\s*([^,]+)',  # с. Семёновка
        r'(?:п\.|посёлок|пос\.)\s*([^,]+)',  # п. Медведево
        r'(?:пгт|пгт\.)\s*([^,]+)',  # пгт Советский
    ]

    for pattern in patterns:
        match = re.search(pattern, address, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


# === ФУНКЦИЯ ПОЛУЧЕНИЯ ОБЪЕКТОВ В КВАРТАЛЕ ===
def get_objects_in_quarter(
    quarter_cn: str,
    object_type: int = 1,  # 1=ЗУ, 5=ОКС
    limit: int = 100,
    offset: int = 0
) -> dict:
    """
    Получает список объектов в кадастровом квартале через ПКК API

    Args:
        quarter_cn: Кадастровый номер квартала (например: 12:05:0201001)
        object_type: Тип объектов (1=земельные участки, 5=ОКС)
        limit: Максимальное количество объектов
        offset: Смещение для пагинации

    Returns:
        Словарь с результатами поиска
    """

    # Проверяем формат
    if not is_cadastral_quarter(quarter_cn):
        return {
            "success": False,
            "error": f"Неверный формат кадастрового квартала: {quarter_cn}. Ожидается формат XX:XX:XXXXXXX"
        }

    try:
        # Запрос к ПКК API
        url = f"{PKK_API_URL}/{object_type}"
        params = {
            "text": quarter_cn,
            "limit": min(limit, 100),  # ПКК ограничивает 100
            "skip": offset,
            "tolerance": 4
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://pkk.rosreestr.ru/"
        }

        # verify=False для российских SSL-сертификатов
        response = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
        response.raise_for_status()
        data = response.json()

        features = data.get("features", [])
        total = data.get("total", len(features))

        # Обрабатываем результаты
        objects = []
        settlements = set()

        for feature in features:
            attrs = feature.get("attrs", {})
            cn = attrs.get("cn", "")
            address = attrs.get("address", "")

            # Проверяем, что объект принадлежит запрашиваемому кварталу
            if not cn.startswith(quarter_cn):
                continue

            obj = {
                "cn": cn,
                "address": address,
                "area": attrs.get("area_value"),
                "cost": attrs.get("cad_cost"),
                "category": attrs.get("category_type"),
                "permitted_use": attrs.get("util_by_doc"),
                "status": attrs.get("statecd"),
                "has_geometry": feature.get("center") is not None
            }

            # Извлекаем НП из адреса
            settlement = extract_settlement_from_address(address)
            if settlement:
                obj["settlement"] = settlement
                settlements.add(settlement)

            objects.append(obj)

        return {
            "success": True,
            "quarter": quarter_cn,
            "object_type": object_type,
            "object_type_name": "Земельные участки" if object_type == 1 else "Объекты капитального строительства",
            "total": total,
            "returned": len(objects),
            "offset": offset,
            "limit": limit,
            "settlements": list(settlements),
            "objects": objects
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "quarter": quarter_cn,
            "error": "Timeout: ПКК API не ответил вовремя"
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "quarter": quarter_cn,
            "error": f"Ошибка запроса к ПКК: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "quarter": quarter_cn,
            "error": str(e)
        }


# === ОСНОВНЫЕ ФУНКЦИИ ===
def get_area_data(
    cadastral_number: str,
    area_type: int = 1,
    full_info: bool = True,
    center_only: bool = False,
    use_cache: bool = True
) -> dict:
    """Получить данные об участке"""

    # Проверяем кэш
    if use_cache:
        cache_key = get_cache_key(cadastral_number, area_type, center_only)
        cached = get_from_cache(cache_key)
        if cached:
            cached["from_cache"] = True
            return cached

    try:
        # Вызываем Area без center_only - библиотека его не поддерживает
        # center_only используется только для фильтрации возвращаемых данных
        area = Area(
            code=cadastral_number,
            area_type=area_type,
            with_log=False,
            timeout=30
        )

        if area.feature:
            geometry = area.feature.get("geometry", {})
            properties = area.feature.get("properties", {})
            options = properties.get("options", {})

            info = extract_full_info(options, area_type) if full_info else {
                "cadastral_number": options.get("cn") or options.get("cad_num", cadastral_number),
                "address": options.get("address") or options.get("readable_address", "")
            }

            # Центр - извлекаем из геометрии или из properties
            center = extract_center(geometry) or properties.get("center", {})

            result = {
                "success": True,
                "cadastral_number": cadastral_number,
                "area_type": area_type,
                "area_type_name": AREA_TYPES.get(area_type, f"Тип {area_type}"),
                "data": info,
                "center": center,
                "from_cache": False
            }

            # Добавляем геометрию только если не center_only
            if not center_only:
                result["geometry"] = {
                    "type": geometry.get("type", ""),
                    "coordinates": geometry.get("coordinates", [])
                }
                result["geojson"] = {
                    "type": "Feature",
                    "properties": info,
                    "geometry": geometry
                }

            # Кэшируем результат
            if use_cache:
                set_cache(cache_key, result)

            return result
        else:
            return {
                "success": False,
                "cadastral_number": cadastral_number,
                "area_type": area_type,
                "error": "No data found"
            }

    except Exception as e:
        return {
            "success": False,
            "cadastral_number": cadastral_number,
            "area_type": area_type,
            "error": str(e)
        }


def search_by_text(query: str, area_type: int = 1) -> dict:
    """Поиск объектов по тексту"""
    try:
        area = Area(code=query, area_type=area_type, with_log=False, timeout=30)
        if area.feature:
            cn = area.feature.get("properties", {}).get("options", {}).get("cn")
            cn = cn or area.feature.get("properties", {}).get("options", {}).get("cad_num", query)
            return get_area_data(cn, area_type)
        else:
            return {"success": False, "query": query, "error": "No results found"}
    except Exception as e:
        return {"success": False, "query": query, "error": str(e)}


# === ENDPOINTS ===
@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "rosreestr2coord-api",
        "version": "3.1.1",
        "features": [
            "Полная информация об объектах недвижимости",
            "Все типы объектов НСПД (1, 2, 4, 5, 7, 15)",
            "center_only - только центры участков",
            "KML экспорт",
            "Кэширование (TTL 1 час)",
            "GeoJSON координаты",
            "Список объектов в кадастровом квартале"
        ],
        "cache_size": len(CACHE)
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "cache_entries": len(CACHE)}


@app.get("/api/types")
async def get_types():
    """Получить список типов объектов"""
    return {
        "types": AREA_TYPES,
        "categories": LAND_CATEGORIES,
        "coord_systems": COORD_SYSTEMS
    }


@app.delete("/api/cache")
async def clear_cache(authorization: Optional[str] = Header(None)):
    """Очистить кэш"""
    verify_token(authorization)
    count = len(CACHE)
    CACHE.clear()
    return {"cleared": count}


@app.get("/api/cadastral/{cadastral_number}")
async def get_cadastral(
    cadastral_number: str,
    area_type: int = Query(1, description="Тип объекта (1, 2, 4, 5, 7, 15)"),
    full: bool = Query(True, description="Полная информация"),
    center_only: bool = Query(False, description="Только центр (без геометрии)"),
    no_cache: bool = Query(False, description="Не использовать кэш"),
    authorization: Optional[str] = Header(None)
):
    """Получить информацию по кадастровому номеру"""
    verify_token(authorization)
    return get_area_data(cadastral_number, area_type, full, center_only, use_cache=not no_cache)


@app.get("/api/cadastral/{cadastral_number}/center")
async def get_cadastral_center(
    cadastral_number: str,
    area_type: int = Query(1),
    authorization: Optional[str] = Header(None)
):
    """Получить только центр участка"""
    verify_token(authorization)
    return get_area_data(cadastral_number, area_type, full_info=False, center_only=True)


@app.get("/api/cadastral/{cadastral_number}/kml", response_class=PlainTextResponse)
async def get_cadastral_kml(
    cadastral_number: str,
    area_type: int = Query(1),
    authorization: Optional[str] = Header(None)
):
    """Экспорт в KML формат"""
    verify_token(authorization)
    result = get_area_data(cadastral_number, area_type, center_only=False)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))

    geometry = result.get("geometry", {})
    name = result.get("data", {}).get("cadastral_number", cadastral_number)
    address = result.get("data", {}).get("address", "")

    kml = geometry_to_kml(geometry, name, address)
    return Response(content=kml, media_type="application/vnd.google-earth.kml+xml")


@app.get("/api/cadastral/{cadastral_number}/oks")
async def get_cadastral_oks(
    cadastral_number: str,
    authorization: Optional[str] = Header(None)
):
    """Получить информацию об ОКС на участке"""
    verify_token(authorization)
    land_result = get_area_data(cadastral_number, area_type=1)
    oks_result = get_area_data(cadastral_number, area_type=5)
    return {
        "cadastral_number": cadastral_number,
        "land_plot": land_result if land_result.get("success") else None,
        "oks": oks_result if oks_result.get("success") else None
    }


# === ЭНДПОИНТ: Список объектов в квартале ===
@app.get("/api/quarter/{quarter_cn}/objects")
async def get_quarter_objects(
    quarter_cn: str,
    object_type: int = Query(1, description="Тип объектов: 1=ЗУ, 5=ОКС"),
    limit: int = Query(100, description="Максимальное количество (до 100)"),
    offset: int = Query(0, description="Смещение для пагинации"),
    authorization: Optional[str] = Header(None)
):
    """
    Получить список объектов в кадастровом квартале

    - **quarter_cn**: Кадастровый номер квартала (например: 12:05:0201001)
    - **object_type**: 1 = земельные участки, 5 = объекты капитального строительства
    - **limit**: Максимальное количество объектов (до 100)
    - **offset**: Смещение для пагинации

    Возвращает список объектов с кадастровыми номерами, адресами и характеристиками.
    Также извлекает названия населённых пунктов из адресов.
    """
    verify_token(authorization)
    return get_objects_in_quarter(quarter_cn, object_type, limit, offset)


@app.get("/api/quarter/{quarter_cn}/settlements")
async def get_quarter_settlements(
    quarter_cn: str,
    authorization: Optional[str] = Header(None)
):
    """
    Получить список населённых пунктов в кадастровом квартале

    Анализирует адреса объектов и извлекает уникальные названия НП.
    """
    verify_token(authorization)

    # Получаем объекты (земельные участки)
    result = get_objects_in_quarter(quarter_cn, object_type=1, limit=100)

    if not result.get("success"):
        return result

    settlements = result.get("settlements", [])

    return {
        "success": True,
        "quarter": quarter_cn,
        "settlements": settlements,
        "count": len(settlements),
        "analyzed_objects": result.get("returned", 0)
    }


@app.post("/api/cadastral")
async def post_cadastral(request: CadastralRequest, authorization: Optional[str] = Header(None)):
    """Получить информацию по кадастровому номеру (POST)"""
    verify_token(authorization)
    return get_area_data(request.cadastral_number, request.area_type, center_only=request.center_only)


@app.post("/api/batch")
async def batch_cadastral(request: BatchRequest, authorization: Optional[str] = Header(None)):
    """Пакетное получение информации"""
    verify_token(authorization)
    results = [
        get_area_data(cn, request.area_type, center_only=request.center_only)
        for cn in request.cadastral_numbers
    ]

    if request.center_only:
        # Для center_only возвращаем упрощённый формат
        centers = [
            {"cadastral_number": r["cadastral_number"], "center": r.get("center")}
            for r in results if r.get("success")
        ]
        return {
            "total": len(request.cadastral_numbers),
            "success_count": len(centers),
            "centers": centers
        }
    else:
        features = [r["geojson"] for r in results if r.get("success") and r.get("geojson")]
        return {
            "total": len(request.cadastral_numbers),
            "success_count": sum(1 for r in results if r.get("success")),
            "results": results,
            "geojson": {"type": "FeatureCollection", "features": features}
        }


@app.post("/api/batch/kml", response_class=PlainTextResponse)
async def batch_kml(request: BatchRequest, authorization: Optional[str] = Header(None)):
    """Пакетный экспорт в KML"""
    verify_token(authorization)
    results = [get_area_data(cn, request.area_type) for cn in request.cadastral_numbers]

    kml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '<name>Кадастровые объекты</name>'
    ]

    for r in results:
        if r.get("success") and r.get("geometry"):
            name = r.get("data", {}).get("cadastral_number", "")
            geo = r.get("geometry", {})
            geo_type = geo.get("type", "")
            coords = geo.get("coordinates", [])

            kml_parts.append(f'<Placemark><name>{name}</name>')

            if geo_type == "Polygon" and coords:
                kml_parts.append('<Polygon><outerBoundaryIs><LinearRing><coordinates>')
                coord_str = ' '.join([f'{p[0]},{p[1]},0' for p in coords[0]])
                kml_parts.append(coord_str)
                kml_parts.append('</coordinates></LinearRing></outerBoundaryIs></Polygon>')

            kml_parts.append('</Placemark>')

    kml_parts.extend(['</Document>', '</kml>'])
    return Response(content='\n'.join(kml_parts), media_type="application/vnd.google-earth.kml+xml")


@app.post("/api/search")
async def search(request: SearchRequest, authorization: Optional[str] = Header(None)):
    """Поиск объектов по адресу или номеру"""
    verify_token(authorization)
    return search_by_text(request.query, request.area_type)


@app.get("/api/search")
async def search_get(
    q: str = Query(..., description="Поисковый запрос"),
    area_type: int = Query(1, description="Тип объекта"),
    authorization: Optional[str] = Header(None)
):
    """Поиск объектов по адресу или номеру (GET)"""
    verify_token(authorization)
    return search_by_text(q, area_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
