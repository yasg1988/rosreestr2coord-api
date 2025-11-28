"""
Rosreestr2Coord API Server v3.3.0
Полная информация о земельных участках и ОКС по кадастровому номеру
+ Глубокое сканирование объектов в кадастровом квартале
+ Сканирование кварталов в кадастровом районе
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

# Отключаем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(
    title="Rosreestr2Coord API",
    description="API для получения полной информации об объектах недвижимости по кадастровому номеру",
    version="3.3.0"
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
    info = {
        "cadastral_number": options.get("cn") or options.get("cad_num", ""),
        "object_type": options.get("obj_kind_value") or AREA_TYPES.get(area_type, f"Тип {area_type}"),
        "address": options.get("address") or options.get("readable_address", ""),
        "status": options.get("statecd") or options.get("status", ""),
    }

    area_value = parse_area_value(options)
    if area_value:
        info["area"] = {"value": area_value, "unit": options.get("area_unit", "кв.м")}

    cad_cost = format_cadastral_cost(options)
    if cad_cost:
        info["cadastral_cost"] = cad_cost

    if area_type == 1:
        category = options.get("category_type", "")
        if category:
            info["category"] = {"code": category, "name": get_category_name(category)}
        permitted = options.get("util_by_doc") or options.get("permitted_use_name", "")
        if permitted:
            info["permitted_use"] = {"by_document": permitted}

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

    if options.get("cnt_land"):
        info["statistics"] = {
            "land_plots": options.get("cnt_land", 0),
            "land_plots_with_geometry": options.get("cnt_land_geom", 0),
            "oks_count": options.get("cnt_oks", 0),
            "oks_with_geometry": options.get("cnt_oks_geom", 0),
            "total_land_area": options.get("sum_land_area", 0),
            "total_cost": options.get("cost_value_total_geom", 0)
        }

    dates = {}
    if options.get("cad_record_date") or options.get("build_record_registration_date"):
        dates["registration"] = options.get("cad_record_date") or options.get("build_record_registration_date", "")
    if options.get("adate") or options.get("date_ch"):
        dates["data_update"] = options.get("adate") or options.get("date_ch", "")
    if dates:
        info["dates"] = dates

    if options.get("rights_reg") or options.get("right_type"):
        info["rights"] = {
            "registered": bool(options.get("rights_reg")),
            "type": options.get("right_type") or options.get("ownership_type", "")
        }

    quarter = options.get("kvartal_cn") or options.get("quarter_cad_number", "")
    if quarter:
        info["cadastral_quarter"] = quarter

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
    """Проверяет, является ли номер кадастровым кварталом"""
    pattern = r'^\d{2}:\d{2}:\d{6,7}$'
    return bool(re.match(pattern, cn))


def extract_settlement_from_address(address: str) -> Optional[str]:
    """Извлекает название населённого пункта из адреса"""
    if not address:
        return None

    patterns = [
        r'(?:г\.|город)\s*([^,]+)',
        r'(?:д\.|деревня)\s*([^,]+)',
        r'(?:с\.|село)\s*([^,]+)',
        r'(?:п\.|посёлок|пос\.)\s*([^,]+)',
        r'(?:пгт|пгт\.)\s*([^,]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, address, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


# === ФУНКЦИЯ СКАНИРОВАНИЯ КВАРТАЛОВ В РАЙОНЕ ===
def scan_quarters_in_district(
    district_cn: str,
    max_quarters: int = 100,
    zones: List[str] = None
) -> dict:
    """
    Сканирует кварталы в районе путём перебора возможных номеров.
    
    Формат кадастрового квартала: XX:XX:XXYYZZZZ
    где XX:XX - район, XXYY - зона, ZZZZ - номер в зоне (обычно 001)
    
    Стандартные зоны: 01-15 для первой части, 01-10 для второй
    """
    
    # Валидация формата района
    if not re.match(r'^\d{2}:\d{2}$', district_cn):
        return {
            "success": False,
            "error": f"Неверный формат кадастрового района: {district_cn}. Ожидается формат XX:XX"
        }
    
    # Если зоны не указаны, сканируем стандартные
    if zones is None:
        zones = []
        # Генерируем возможные зоны: 0101, 0102, ..., 0201, 0202, ... до 1510
        for z1 in range(1, 16):  # 01-15
            for z2 in range(1, 11):  # 01-10
                zones.append(f"{z1:02d}{z2:02d}")
    
    found_quarters = []
    scanned = 0
    empty_zones = 0
    max_empty_zones = 30  # Останавливаемся после 30 пустых зон подряд
    
    for zone in zones:
        if len(found_quarters) >= max_quarters:
            break
        if empty_zones >= max_empty_zones:
            break
            
        # Пробуем номер 001 в этой зоне
        quarter_cn = f"{district_cn}:{zone}001"
        scanned += 1
        
        try:
            area = Area(
                code=quarter_cn,
                area_type=2,
                with_log=False,
                timeout=10
            )
            
            if area.feature:
                props = area.feature.get("properties", {})
                options = props.get("options", {})
                geometry = area.feature.get("geometry", {})
                center = extract_center(geometry)
                
                quarter_info = {
                    "cn": quarter_cn,
                    "name": options.get("obj_label", quarter_cn),
                    "area_sqm": options.get("sum_land_area"),
                    "land_plots": options.get("cnt_land", 0),
                    "center": center,
                    "has_geometry": geometry is not None
                }
                
                found_quarters.append(quarter_info)
                empty_zones = 0  # Сбрасываем счётчик пустых зон
                
                # Проверяем 002, 003 и т.д. в этой зоне
                for sub_num in range(2, 10):
                    sub_quarter_cn = f"{district_cn}:{zone}{sub_num:03d}"
                    scanned += 1
                    
                    try:
                        sub_area = Area(
                            code=sub_quarter_cn,
                            area_type=2,
                            with_log=False,
                            timeout=10
                        )
                        
                        if sub_area.feature:
                            sub_props = sub_area.feature.get("properties", {})
                            sub_options = sub_props.get("options", {})
                            sub_geometry = sub_area.feature.get("geometry", {})
                            sub_center = extract_center(sub_geometry)
                            
                            found_quarters.append({
                                "cn": sub_quarter_cn,
                                "name": sub_options.get("obj_label", sub_quarter_cn),
                                "area_sqm": sub_options.get("sum_land_area"),
                                "land_plots": sub_options.get("cnt_land", 0),
                                "center": sub_center,
                                "has_geometry": sub_geometry is not None
                            })
                        else:
                            break  # Нет больше кварталов в этой зоне
                    except:
                        break
                    
                    time.sleep(0.1)
            else:
                empty_zones += 1
        except:
            empty_zones += 1
        
        time.sleep(0.15)  # Пауза между запросами
    
    return {
        "success": True,
        "district": district_cn,
        "quarters": found_quarters,
        "count": len(found_quarters),
        "scanned": scanned,
        "note": f"Просканировано {scanned} номеров, найдено {len(found_quarters)} кварталов"
    }


# === ФУНКЦИЯ ПОЛУЧЕНИЯ ОБЪЕКТОВ В КВАРТАЛЕ (стандартная) ===
def get_objects_in_quarter(
    quarter_cn: str,
    object_type: int = 1,
    limit: int = 50,
    offset: int = 0,
    scan_depth: int = 100,
    max_number: int = 2000
) -> dict:
    """
    Получает список объектов в кадастровом квартале через последовательный перебор
    
    Args:
        quarter_cn: Кадастровый номер квартала
        object_type: Тип объекта (1=ЗУ, 5=ОКС)
        limit: Максимальное количество объектов для возврата
        offset: Начальный номер для сканирования
        scan_depth: Сколько пустых номеров подряд допускать (default 100)
        max_number: Максимальный номер для сканирования (default 2000)
    """

    if not is_cadastral_quarter(quarter_cn):
        return {
            "success": False,
            "error": f"Неверный формат кадастрового квартала: {quarter_cn}. Ожидается формат XX:XX:XXXXXXX"
        }

    try:
        objects = []
        settlements = set()
        not_found_count = 0
        current_num = offset + 1
        last_found_num = 0

        while len(objects) < limit and current_num <= max_number:
            # Прекращаем если слишком много пустых подряд
            if not_found_count >= scan_depth:
                break
                
            cn = f"{quarter_cn}:{current_num}"

            try:
                area = Area(
                    code=cn,
                    area_type=object_type,
                    with_log=False,
                    timeout=10
                )

                if area.feature:
                    props = area.feature.get("properties", {})
                    options = props.get("options", {})
                    geometry = area.feature.get("geometry", {})
                    address = options.get("address") or options.get("readable_address", "")
                    center = extract_center(geometry) if geometry else None

                    obj = {
                        "cn": cn,
                        "address": address,
                        "area": options.get("area_value"),
                        "cost": options.get("cad_cost"),
                        "category": options.get("category_type"),
                        "permitted_use": options.get("util_by_doc"),
                        "status": options.get("statecd"),
                        "has_geometry": geometry is not None,
                        "center": center
                    }

                    settlement = extract_settlement_from_address(address)
                    if settlement:
                        obj["settlement"] = settlement
                        settlements.add(settlement)

                    objects.append(obj)
                    not_found_count = 0
                    last_found_num = current_num
                else:
                    not_found_count += 1

            except Exception:
                not_found_count += 1

            current_num += 1
            time.sleep(0.05)  # Ускорили с 0.1 до 0.05

        return {
            "success": True,
            "quarter": quarter_cn,
            "object_type": object_type,
            "object_type_name": "Земельные участки" if object_type == 1 else "Объекты капитального строительства",
            "total_scanned": current_num - offset - 1,
            "last_scanned_number": current_num - 1,
            "last_found_number": last_found_num,
            "returned": len(objects),
            "offset": offset,
            "limit": limit,
            "scan_depth": scan_depth,
            "max_number": max_number,
            "settlements": list(settlements),
            "objects": objects,
            "note": f"Просканировано: 1-{current_num-1}, найдено: {len(objects)}, последний найденный: {last_found_num}"
        }

    except Exception as e:
        return {
            "success": False,
            "quarter": quarter_cn,
            "error": str(e)
        }


# === ФУНКЦИЯ ГЛУБОКОГО СКАНИРОВАНИЯ (для 100% покрытия) ===
def deep_scan_quarter(
    quarter_cn: str,
    object_type: int = 1,
    max_number: int = 3000,
    batch_size: int = 500
) -> dict:
    """
    Глубокое сканирование квартала для получения 100% объектов.
    Сканирует весь диапазон номеров без остановки на пустых.
    
    Args:
        quarter_cn: Кадастровый номер квартала
        object_type: Тип объекта (1=ЗУ, 5=ОКС)  
        max_number: Максимальный номер для сканирования (до 5000)
        batch_size: Размер батча для отчёта о прогрессе
    """

    if not is_cadastral_quarter(quarter_cn):
        return {
            "success": False,
            "error": f"Неверный формат кадастрового квартала: {quarter_cn}"
        }

    # Ограничиваем max_number
    max_number = min(max_number, 5000)

    try:
        objects = []
        settlements = set()
        found_numbers = []
        
        start_time = time.time()
        
        for current_num in range(1, max_number + 1):
            cn = f"{quarter_cn}:{current_num}"

            try:
                area = Area(
                    code=cn,
                    area_type=object_type,
                    with_log=False,
                    timeout=8
                )

                if area.feature:
                    props = area.feature.get("properties", {})
                    options = props.get("options", {})
                    geometry = area.feature.get("geometry", {})
                    address = options.get("address") or options.get("readable_address", "")
                    center = extract_center(geometry) if geometry else None

                    obj = {
                        "cn": cn,
                        "address": address,
                        "area": options.get("area_value"),
                        "cost": options.get("cad_cost"),
                        "category": options.get("category_type"),
                        "permitted_use": options.get("util_by_doc"),
                        "status": options.get("statecd"),
                        "has_geometry": geometry is not None,
                        "center": center
                    }

                    settlement = extract_settlement_from_address(address)
                    if settlement:
                        obj["settlement"] = settlement
                        settlements.add(settlement)

                    objects.append(obj)
                    found_numbers.append(current_num)

            except Exception:
                pass

            # Небольшая пауза каждые 10 запросов
            if current_num % 10 == 0:
                time.sleep(0.02)

        elapsed = time.time() - start_time
        
        # Анализ распределения номеров
        gaps = []
        if len(found_numbers) > 1:
            for i in range(1, len(found_numbers)):
                gap = found_numbers[i] - found_numbers[i-1]
                if gap > 10:
                    gaps.append({"after": found_numbers[i-1], "gap": gap})

        return {
            "success": True,
            "quarter": quarter_cn,
            "object_type": object_type,
            "object_type_name": "Земельные участки" if object_type == 1 else "ОКС",
            "scanned_range": f"1-{max_number}",
            "total_scanned": max_number,
            "found_count": len(objects),
            "coverage_percent": round(len(objects) / max_number * 100, 2) if max_number > 0 else 0,
            "elapsed_seconds": round(elapsed, 1),
            "settlements": list(settlements),
            "number_distribution": {
                "first": found_numbers[0] if found_numbers else None,
                "last": found_numbers[-1] if found_numbers else None,
                "large_gaps": gaps[:10]  # Показываем до 10 больших разрывов
            },
            "objects": objects,
            "note": f"Полное сканирование 1-{max_number}, найдено {len(objects)} объектов за {round(elapsed, 1)} сек"
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

    if use_cache:
        cache_key = get_cache_key(cadastral_number, area_type, center_only)
        cached = get_from_cache(cache_key)
        if cached:
            cached["from_cache"] = True
            return cached

    try:
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
        "version": "3.3.0",
        "features": [
            "Полная информация об объектах недвижимости",
            "Все типы объектов НСПД (1, 2, 4, 5, 7, 15)",
            "center_only - только центры участков",
            "KML экспорт",
            "Кэширование (TTL 1 час)",
            "GeoJSON координаты",
            "Глубокое сканирование объектов в квартале (до 5000 номеров)",
            "Сканирование кварталов в кадастровом районе"
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


# === ЭНДПОИНТЫ: Сканирование кварталов в районе ===
@app.get("/api/district/{district_cn}/quarters/scan")
async def scan_district_quarters(
    district_cn: str,
    max_quarters: int = Query(100, description="Максимальное количество кварталов"),
    authorization: Optional[str] = Header(None)
):
    """
    Сканировать кварталы в кадастровом районе
    
    - **district_cn**: Кадастровый номер района (например: 12:05)
    - **max_quarters**: Максимальное количество кварталов (по умолчанию 100)
    
    Сканирует кварталы путём перебора возможных номеров.
    Медленно, но надёжно работает без PKK API.
    """
    verify_token(authorization)
    return scan_quarters_in_district(district_cn, max_quarters)


@app.get("/api/district/{district_cn}/quarters/boundaries")
async def get_district_quarters_boundaries(
    district_cn: str,
    max_quarters: int = Query(50, description="Максимальное количество кварталов"),
    authorization: Optional[str] = Header(None)
):
    """
    Получить список кварталов района с их границами (GeoJSON)
    
    Сначала сканирует кварталы, затем получает границы каждого.
    Медленно, но возвращает полную геометрию.
    """
    verify_token(authorization)
    
    # Сначала сканируем кварталы
    scan_result = scan_quarters_in_district(district_cn, max_quarters)
    
    if not scan_result.get("success"):
        return scan_result
    
    quarters = scan_result.get("quarters", [])
    
    if not quarters:
        return {
            "success": True,
            "district": district_cn,
            "features": [],
            "count": 0
        }
    
    # Получаем границы для каждого квартала
    features = []
    for q in quarters:
        cn = q.get("cn")
        if cn:
            result = get_area_data(cn, area_type=2, center_only=False, use_cache=True)
            if result.get("success") and result.get("geojson"):
                features.append(result["geojson"])
            time.sleep(0.1)
    
    return {
        "success": True,
        "district": district_cn,
        "type": "FeatureCollection",
        "features": features,
        "count": len(features)
    }


# === ЭНДПОИНТ: Список объектов в квартале (стандартный) ===
@app.get("/api/quarter/{quarter_cn}/objects")
async def get_quarter_objects(
    quarter_cn: str,
    object_type: int = Query(1, description="Тип объектов: 1=ЗУ, 5=ОКС"),
    limit: int = Query(100, description="Максимальное количество объектов"),
    offset: int = Query(0, description="Начальный номер объекта"),
    scan_depth: int = Query(100, description="Глубина сканирования (пустых номеров подряд)"),
    max_number: int = Query(2000, description="Максимальный номер для сканирования"),
    authorization: Optional[str] = Header(None)
):
    """
    Получить список объектов в кадастровом квартале
    
    - **scan_depth**: сколько пустых номеров подряд проверять (default 100)
    - **max_number**: верхняя граница сканирования (до 5000)
    """
    verify_token(authorization)
    # Ограничения
    scan_depth = min(scan_depth, 500)
    max_number = min(max_number, 5000)
    limit = min(limit, 2000)
    
    return get_objects_in_quarter(quarter_cn, object_type, limit, offset, scan_depth, max_number)


# === ЭНДПОИНТ: Глубокое сканирование квартала (100% покрытие) ===
@app.get("/api/quarter/{quarter_cn}/objects/deep")
async def get_quarter_objects_deep(
    quarter_cn: str,
    object_type: int = Query(1, description="Тип объектов: 1=ЗУ, 5=ОКС"),
    max_number: int = Query(2000, description="Максимальный номер (до 5000)"),
    authorization: Optional[str] = Header(None)
):
    """
    Глубокое сканирование квартала для 100% покрытия объектов
    
    Сканирует ВСЕ номера от 1 до max_number без пропусков.
    Медленно, но гарантирует полноту данных.
    
    **Внимание**: При max_number=3000 сканирование занимает ~5-10 минут!
    """
    verify_token(authorization)
    max_number = min(max_number, 5000)
    return deep_scan_quarter(quarter_cn, object_type, max_number)


@app.get("/api/quarter/{quarter_cn}/settlements")
async def get_quarter_settlements(
    quarter_cn: str,
    authorization: Optional[str] = Header(None)
):
    """
    Получить список населённых пунктов в кадастровом квартале
    """
    verify_token(authorization)
    result = get_objects_in_quarter(quarter_cn, object_type=1, limit=50, scan_depth=100)
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
