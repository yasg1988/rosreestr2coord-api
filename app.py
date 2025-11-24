"""
Rosreestr2Coord API Server
Получение координат земельных участков по кадастровому номеру
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import os

from rosreestr2coord.parser import Area

app = FastAPI(
    title="Rosreestr2Coord API",
    description="API для получения координат участков по кадастровому номеру",
    version="1.0.0"
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


class CadastralRequest(BaseModel):
    cadastral_number: str
    area_type: int = 1


class BatchRequest(BaseModel):
    cadastral_numbers: List[str]
    area_type: int = 1


def verify_token(authorization: Optional[str] = Header(None)):
    """Проверка токена авторизации"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    token = authorization.replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


def get_area_data(cadastral_number: str, area_type: int = 1) -> dict:
    """Получить данные об участке"""
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

            return {
                "success": True,
                "cadastral_number": cadastral_number,
                "data": {
                    "cn": options.get("cn", cadastral_number),
                    "address": options.get("address", ""),
                    "area": options.get("area", ""),
                    "geometry_type": geometry.get("type", ""),
                    "coordinates": geometry.get("coordinates", []),
                    "center": properties.get("center", {})
                },
                "geojson": {
                    "type": "Feature",
                    "properties": {
                        "cadastral_number": options.get("cn", cadastral_number),
                        "address": options.get("address", ""),
                        "area": options.get("area", "")
                    },
                    "geometry": geometry
                }
            }
        else:
            return {
                "success": False,
                "cadastral_number": cadastral_number,
                "error": "No data found"
            }

    except Exception as e:
        return {
            "success": False,
            "cadastral_number": cadastral_number,
            "error": str(e)
        }


@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "service": "rosreestr2coord-api", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/cadastral/{cadastral_number}")
async def get_cadastral(
    cadastral_number: str,
    area_type: int = 1,
    authorization: Optional[str] = Header(None)
):
    """Получить координаты по кадастровому номеру"""
    verify_token(authorization)
    return get_area_data(cadastral_number, area_type)


@app.post("/api/cadastral")
async def post_cadastral(
    request: CadastralRequest,
    authorization: Optional[str] = Header(None)
):
    """Получить координаты по кадастровому номеру (POST)"""
    verify_token(authorization)
    return get_area_data(request.cadastral_number, request.area_type)


@app.post("/api/batch")
async def batch_cadastral(
    request: BatchRequest,
    authorization: Optional[str] = Header(None)
):
    """Пакетное получение координат"""
    verify_token(authorization)

    results = []
    for cn in request.cadastral_numbers:
        result = get_area_data(cn, request.area_type)
        results.append(result)

    # Собираем GeoJSON FeatureCollection
    features = []
    for r in results:
        if r.get("success") and r.get("geojson"):
            features.append(r["geojson"])

    return {
        "total": len(request.cadastral_numbers),
        "success_count": sum(1 for r in results if r.get("success")),
        "results": results,
        "geojson": {
            "type": "FeatureCollection",
            "features": features
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
