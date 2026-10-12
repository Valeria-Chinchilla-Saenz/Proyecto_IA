from fastapi import FastAPI
from api.schemas import SolicitudDemanda, RespuestaDemanda, RespuestaNivelDemanda
from api.predict import PredictorDemanda, PredictorNivelDemanda

app = FastAPI(title="API Predicción de Demanda de Bicicletas")


predictor_demanda = PredictorDemanda()
#predictor_nivel = PredictorNivelDemanda()


@app.post("/predict/demand", response_model=RespuestaDemanda)
def predecir_demanda(solicitud: SolicitudDemanda):
    demanda = predictor_demanda.predecir(solicitud)
    return RespuestaDemanda(demanda_estimada=demanda)


'''@app.post("/predict/demand_level", response_model=RespuestaNivelDemanda)
def predecir_nivel_demanda(solicitud: SolicitudDemanda):
    nivel = predictor_nivel.predecir(solicitud)
    return RespuestaNivelDemanda(nivel_demanda=nivel)'''