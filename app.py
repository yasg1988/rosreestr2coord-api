"""
Rosreestr2Coord API Server v2.0
Получение полной информации о земельных участках и ОКС по кадастровому номеру
"""

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os

from rosreestr2coord.parser import Area

app = FastAPI(
    title="Rosreestr2Coord API",
    description="API для получения полной информации об объектах недвижимости по кадастровому номеру",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Простая авторизация
API_TOKEN = os.environ.get("API_TOKEN", "rr2c_api_7f8a9b3c4d5e6f")

# Типы объектов в ПКК
AREA_TYPES = {
    1: "Земельные участки",
    2: "Кадастровые кварталы",
    3: "Кадастровые районы",
    4: "Кадастровые округа",
    5: "ОКС (здания, сооружения)",
    6: "Границы",
    7: "ЗОУИТ",
    10: "Единые недвижимые комплексы"
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


class CadastralRequest(BaseModel):
    cadastral_number: str
    area_type: int = 1


class BatchRequest(BaseModel):
    cadastral_numbers: List[str]
    area_type: int = 1


class SearchRequest(BaseModel):
    query: str
    area_type: int = 1
    limit: int = 10


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
    area_value = options.get("area_value")
    if area_value:
        try:
            return float(area_value)
        except (ValueError, TypeError):
            pass
    area_str = options.get("area", "")
    if area_str:
        try:
            num = "".join(c for c in str(area_str) if c.isdigit() or c == ".")
            if num:
                return float(num)
        except (ValueError, TypeError):
            pass
    return None


def format_cadastral_cost(options: dict) -> Optional[Dict[str, Any]]:
    """Форматирует кадастровую стоимость"""
    cad_cost = options.get("cad_cost")
    if cad_cost:
        return {
            "value": float(cad_cost) if cad_cost else None,
            "unit": options.get("cad_unit", "руб."),
            "date": options.get("date_cost", "")
        }
    return None


def get_category_name(category_code: str) -> str:
    """Получает название категории земель по коду"""
    return LAND_CATEGORIES.get(category_code, category_code or "Не указана")


def extract_full_info(options: dict, area_type: int = 1) -> Dict[str, Any]:
    """Извлекает полную информацию из options"""
    info = {
        "cadastral_number": options.get("cn", ""),
        "object_type": AREA_TYPES.get(area_type, f"Тип {area_type}"),
        "address": options.get("address", ""),
        "status": options.get("statecd", ""),
    }

    area_value = parse_area_value(options)
    if area_value:
        info["area"] = {
            "value": area_value,
            "unit": options.get("area_unit", "кв.м"),
            "type": options.get("area_type", "")
        }

    cad_cost = format_cadastral_cost(options)
    if cad_cost:
        info["cadastral_cost"] = cad_cost

    if area_type == 1:
        category = options.get("category_type", "")
        info["category"] = {"code": category, "name": get_category_name(category)}
        info["permitted_use"] = {
            "by_document": options.get("util_by_doc", ""),
            "code": options.get("util_code", ""),
            "name": options.get("fp", "")
        }

    if area_type == 5:
        info["building"] = {
            "name": options.get("name", ""),
            "purpose": options.get("purpose", ""),
            "floors": options.get("floors", ""),
            "underground_floors": options.get("underground_floors", ""),
            "year_built": options.get("year_built", ""),
            "year_commissioning": options.get("year_commisioning", ""),
            "walls_material": options.get("walls", "")
        }

    info["dates"] = {
        "registration": options.get("cad_record_date", ""),
        "data_update": options.get("adate", ""),
        "cost_determination": options.get("date_cost", "")
    }

    if options.get("rights_reg"):
        info["rights_registered"] = True

    info["cadastral_quarter"] = options.get("kvartal_cn", "") or options.get("kvartal", "")

    if options.get("oks_list"):
        info["related_oks"] = options.get("oks_list")

    if options.get("special_note"):
        info["special_notes"] = options.get("special_note")

    info["raw_options"] = options
    return info


def get_area_data(cadastral_number: str, area_type: int = 1, full_info: bool = True) -> dict:
    """Получить данные об участке"""
    try:
        area = Area(code=cadastral_number, area_type=area_type, with_log=False, timeout=30)

        if area.feature:
            geometry = area.feature.get("geometry", {})
            properties = area.feature.get("properties", {})
            options = properties.get("options", {})

            info = extract_full_info(options, area_type) if full_info else {
                "cadastral_number": options.get("cn", cadastral_number),
                "address": options.get("address", ""),
                "area": options.get("area", "")
            }

            return {
                "success": True,
                "cadastral_number": cadastral_number,
                "area_type": area_type,
                "area_type_name": AREA_TYPES.get(area_type, f"Тип {area_type}"),
                "data": info,
                "geometry": {
                    "type": geometry.get("type", ""),
                    "coordinates": geometry.get("coordinates", []),
                    "center": properties.get("center", {})
                },
                "geojson": {"type": "Feature", "properties": info, "geometry": geometry}
            }
        else:
            return {"success": False, "cadastral_number": cadastral_number, "area_type": area_type, "error": "No data found"}

    except Exception as e:
        return {"success": False, "cadastral_number": cadastral_number, "area_type": area_type, "error": str(e)}


def search_by_text(query: str, area_type: int = 1) -> dict:
    """Поиск объектов по тексту"""
    try:
        area = Area(code=query, area_type=area_type, with_log=False, timeout=30)
        if area.feature:
            return get_area_data(area.feature.get("properties", {}).get("options", {}).get("cn", query), area_type)
        else:
            return {"success": False, "query": query, "error": "No results found"}
    except Exception as e:
        return {"success": False, "query": query, "error": str(e)}


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "rosreestr2coord-api",
        "version": "2.0.0",
        "features": [
            "Полная информация об объектах недвижимости",
            "Земельные участки (area_type=1)",
            "ОКС - здания, сооружения (area_type=5)",
            "Кадастровая стоимость",
            "Категория земель",
            "Вид разрешённого использования",
            "GeoJSON координаты"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/types")
async def get_types():
    """Получить список типов объектов"""
    return {"types": AREA_TYPES, "categories": LAND_CATEGORIES}


@app.get("/api/cadastral/{cadastral_number}")
async def get_cadastral(
    cadastral_number: str,
    area_type: int = Query(1, description="Тип объекта: 1=Участки, 5=ОКС"),
    full: bool = Query(True, description="Возвращать полную информацию"),
    authorization: Optional[str] = Header(None)
):
    """Получить информацию по кадастровому номеру"""
    verify_token(authorization)
    return get_area_data(cadastral_number, area_type, full)


@app.get("/api/cadastral/{cadastral_number}/oks")
async def get_cadastral_oks(cadastral_number: str, authorization: Optional[str] = Header(None)):
    """Получить информацию об ОКС (здании) на участке"""
    verify_token(authorization)
    land_result = get_area_data(cadastral_number, area_type=1)
    oks_result = get_area_data(cadastral_number, area_type=5)
    return {
        "cadastral_number": cadastral_number,
        "land_plot": land_result if land_result.get("success") else None,
        "oks": oks_result if oks_result.get("success") else None
    }


@app.post("/api/cadastral")
async def post_cadastral(request: CadastralRequest, authorization: Optional[str] = Header(None)):
    """Получить информацию по кадастровому номеру (POST)"""
    verify_token(authorization)
    return get_area_data(request.cadastral_number, request.area_type)


@app.post("/api/batch")
async def batch_cadastral(request: BatchRequest, authorization: Optional[str] = Header(None)):
    """Пакетное получение информации"""
    verify_token(authorization)
    results = [get_area_data(cn, request.area_type) for cn in request.cadastral_numbers]
    features = [r["geojson"] for r in results if r.get("success") and r.get("geojson")]
    return {
        "total": len(request.cadastral_numbers),
        "success_count": sum(1 for r in results if r.get("success")),
        "results": results,
        "geojson": {"type": "FeatureCollection", "features": features}
    }


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
