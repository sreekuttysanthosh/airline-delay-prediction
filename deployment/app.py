
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "airline_delay_classification_pipeline"
)


# --------------------------------------------------
# INITIALIZE FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="Airline Delay Prediction API",
    description=(
        "Predicts whether a flight will be delayed "
        "by more than 15 minutes."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# INITIALIZE SPARK AND LOAD MODEL
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("AirlineDelayPredictionAPI")
    .getOrCreate()
)

pipeline_model = PipelineModel.load(str(MODEL_PATH))


# --------------------------------------------------
# INPUT DATA VALIDATION
# --------------------------------------------------

class FlightInput(BaseModel):

    OP_CARRIER: str = Field(min_length=1)
    ORIGIN: str = Field(min_length=1)
    DEST: str = Field(min_length=1)

    DEP_HOUR: int = Field(ge=0, le=23)
    DEP_MINUTE: int = Field(ge=0, le=59)

    ARR_HOUR: int = Field(ge=0, le=23)
    ARR_MINUTE: int = Field(ge=0, le=59)

    CRS_ELAPSED_TIME: float = Field(gt=0)
    DISTANCE: float = Field(gt=0)

    MONTH: int = Field(ge=1, le=12)
    DAY_OF_WEEK: int = Field(ge=1, le=7)


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Airline Delay Prediction API",
        "status": "running",
        "model": "Weighted Logistic Regression"
    }


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "api_status": "healthy",
        "model_status": "loaded",
        "pipeline_stages": len(pipeline_model.stages)
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_flight(flight: FlightInput):

    try:

        input_data = flight.model_dump()

        input_df = spark.createDataFrame(
            [input_data]
        )

        prediction_df = (
            pipeline_model
            .transform(input_df)
        )

        result = (
            prediction_df
            .select("prediction", "probability")
            .first()
        )

        predicted_class = int(result["prediction"])

        probabilities = result["probability"]

        delay_probability = float(probabilities[1])

        prediction_label = (
            "Delayed"
            if predicted_class == 1
            else "Not Delayed"
        )

        return {
            "prediction": predicted_class,
            "prediction_label": prediction_label,
            "delay_probability": round(
                delay_probability, 4
            ),
            "not_delayed_probability": round(
                float(probabilities[0]), 4
            ),
            "delayed_probability": round(
                delay_probability, 4
            ),
            "delay_threshold_minutes": 15
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
