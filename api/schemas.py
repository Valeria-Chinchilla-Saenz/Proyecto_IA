from pydantic import BaseModel, Field


class SolicitudDemanda(BaseModel):
    season: int = Field(..., ge=1, le=4, description="1=invierno, 2=primavera, 3=verano, 4=otoño")
    weathersit: int = Field(..., ge=1, le=3, description="1=despejado, 2=nublado, 3=lluvia/nieve")
    hr: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    weekday: int = Field(..., ge=0, le=6, description="Día de la semana (0=domingo ... 6=sábado)")
    holiday: int = Field(..., ge=0, le=1)
    workingday: int = Field(..., ge=0, le=1)
    temp: float = Field(..., description="Temperatura normalizada (0 a 1)")
    atemp: float = Field(..., description="Sensación térmica normalizada (0 a 1)")
    hum: float = Field(..., description="Humedad normalizada (0 a 1)")
    windspeed: float = Field(..., description="Velocidad del viento normalizada (0 a 1)")


class RespuestaDemanda(BaseModel):
    demanda_estimada: float

class RespuestaNivelDemanda(BaseModel):
    nivel_demanda: str