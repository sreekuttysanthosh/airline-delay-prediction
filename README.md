✈️ Airline FlightPulse
Airline Delay Prediction Using Apache Spark and Machine Learning
Live application: https://airline-flightpulse.streamlit.app/\
GitHub repository:
https://github.com/sreekuttysanthosh/airline-delay-prediction

Airline FlightPulse is an academic Big Data and Machine Learning project
that analyses historical U.S. airline flight records and predicts
whether a flight is likely to arrive more than 15 minutes late. Apache
Spark and Spark MLlib are used for data preparation, feature
engineering, model training and evaluation. The selected classification
model is exported to JSON and served through a lightweight Streamlit
application using NumPy for inference.
Table of contents
- Objectives
- Dataset
- Methodology
- Results
- Application and deployment
- Repository structure
- Run locally
- Technology stack
- Limitations
- Future improvements
- References
Objectives
- Process historical airline records with PySpark.
- Explore delay patterns across time, carriers and routes.
- Train classification models for delayed-flight detection.
- Train regression models for arrival-delay duration.
- Investigate class imbalance.
- Compare selected Spark and Pandas aggregation workloads.
- Export and validate model inference outside Spark.
- Deploy an interactive prediction interface.
Dataset
The project uses U.S. airline flight records for 2016, 2017 and 2018.

  Detail                              Value
  Raw records                    18,505,725
Raw columns                            27
Eligible modelling records     18,193,961
Training records (2016)         5,538,145
Validation records (2017)       5,579,410
Test records (2018)             7,076,406

Cancelled and diverted flights and records without the required
arrival-delay target were excluded.
Target definition
DELAY_STATUS = 1 when ARR_DELAY > 15 minutes; otherwise
DELAY_STATUS = 0.

  Class              Records    Share
  Not delayed     14,967,741   82.27%
Delayed          3,226,220   17.73%

The data are split chronologically: 2016 for training, 2017 for
validation and 2018 for final testing.
Methodology
Feature set
The model uses information intended to be available from the scheduled
flight record:
- Categorical: OP_CARRIER, ORIGIN, DEST
- Scheduled time: DEP_HOUR, DEP_MINUTE, ARR_HOUR, ARR_MINUTE
- Flight characteristics: CRS_ELAPSED_TIME, DISTANCE
- Calendar: MONTH, DAY_OF_WEEK
YEAR defines the chronological split and is not a model feature.
Post-flight information such as actual arrival delay is excluded to
avoid direct target leakage.
Feature engineering
The Spark ML pipeline uses StringIndexer, OneHotEncoder,
VectorAssembler and StandardScaler. The final assembled feature
vector contains 641 dimensions: 12 carrier dimensions, 311 origin
dimensions, 310 destination dimensions and 8 scaled numerical values.
Models
Classification models compared: - Logistic Regression - Random Forest -
Gradient-Boosted Trees

Weighted Logistic Regression was selected based on validation results.
Linear Regression and Random Forest Regression were evaluated for
predicting arrival-delay duration.
Evaluation metrics
Classification: accuracy, precision, recall, F1-score and ROC-AUC.
Regression: MAE, RMSE and R².
Results
Classification validation
  Model                            Accuracy   Weighted F1   ROC-AUC
  Weighted Logistic Regression       0.6024        0.6490    0.6397
Random Forest                      0.5854        0.6342    0.6339
Gradient-Boosted Trees             0.5773        0.6269    0.6292
Final classification test results (2018)
  Metric                 Result
  Accuracy               0.6043
Weighted precision     0.7523
Weighted recall        0.6043
Weighted F1-score      0.6482
ROC-AUC                0.6279

The delayed class is the minority class. In the validation
class-weighting experiment, delayed-class recall increased from
approximately 0.00004 with unweighted Logistic Regression to 0.6085 with
weighted Logistic Regression, while accuracy decreased. This shows why
accuracy alone is insufficient for imbalanced classification.
Regression results
  Model                        Validation MAE (min)   Validation RMSE (min)   Validation R²
  Linear Regression                         22.1225                 45.4534          0.0179
Random Forest Regression                  22.1539                 45.4953          0.0160

Final 2018 Linear Regression test results: MAE 22.5177 minutes, RMSE
46.5863 minutes, R² 0.0145. The low R² indicates that the selected
features explain little of the variation in actual delay duration.
Spark--Pandas benchmark
For the full aggregation workload, observed median times were 3.127
seconds for Spark and 2.955 seconds for Pandas. Outputs matched for the
tested aggregation. Pandas was slightly faster in this specific
single-machine experiment; this result should not be generalised to
other workloads or distributed Spark deployments.
Application and deployment
The Streamlit application accepts airline code, origin and destination,
scheduled departure and arrival times, scheduled duration, distance,
month and day of week. It displays the predicted class and the estimated
delayed/not-delayed probabilities.

The trained Spark MLlib Logistic Regression parameters were exported to
streamlit_deploy/deployment_artifacts/airline_delay_model.json. The
app reconstructs the 641-dimensional feature vector and performs
inference with NumPy, so Spark is not required for each user prediction.

The NumPy implementation was validated against Spark on 100 test
records: all 100 predicted classes matched, with a maximum probability
difference of approximately 2.22 × 10⁻¹⁶.

Deployment distinction: the public Streamlit app is a lightweight
inference interface. It is not a live Spark service and does not host
the Dockerized FastAPI service.
Repository structure
Key deployment files:
airline-delay-prediction/
├── streamlit_deploy/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── deployment_artifacts/
│       └── airline_delay_model.json
├── Dockerfile
├── requirements.txt
└── README.md
The repository also contains project implementation files. Exact
notebook and data paths may depend on the local setup.
Run locally
Requirements: Python 3.10 or later, pip and Git.
1. Clone
git clone https://github.com/sreekuttysanthosh/airline-delay-prediction.git
cd airline-delay-prediction
2. Create and activate a virtual environment
Windows:
python -m venv .venv
.venv\Scripts\activate
macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install app dependencies
pip install -r streamlit_deploy/requirements.txt
4. Launch
streamlit run streamlit_deploy/streamlit_app.py
Open the local URL printed by Streamlit, usually
http://localhost:8501.
Technology stack
  Area                      Technology
  Programming               Python
Big-data processing       Apache Spark, PySpark
Machine learning          Spark MLlib
Data processing           Pandas, NumPy
Inference                 NumPy
Web application           Streamlit
Cloud hosting             Streamlit Community Cloud
Version control           Git, GitHub
Development environment   Google Colab
Limitations
- Spark ran in local mode on a two-core Colab runtime; a multi-node
  cluster was not tested.
- Weather, air-traffic, aircraft rotation and other operational
  variables were not included.
- Classification performance is modest and should not be treated as
  operationally reliable flight guidance.
- Regression performance is weak, with test R² of 0.0145.
- Cancelled and diverted flights are outside the modelling scope.
- The app uses historical model parameters and user-entered
  information; it does not retrieve live flight data.
- Benchmark timings apply only to the tested machine, data and
  aggregation workload.
Future improvements
- Incorporate weather and airport congestion data.
- Add aircraft rotation and network-level delay propagation features.
- Evaluate threshold tuning, probability calibration and
  precision--recall metrics.
- Test on additional chronological periods.
- Run a multi-node Spark cluster experiment.
- Explore richer regression methods and uncertainty estimation.
- Integrate live flight data if a reliable source is available.
References
1. U.S. Department of Transportation, Bureau of Transportation
   Statistics, Airline On-Time
   Tables.
2. "Predicting the Unpredictable: A Big Data Approach to Airline Delay
   Analysis," 2025 IEEE International Conference on Signal Processing,
   Information, Communication and Systems (SPICSCON). DOI:
   10.1109/SPICSCON69221.2025.11504423. IEEE
   Xplore.
3. Apache Spark, Classification and Regression --- MLlib
   Documentation.
4. Streamlit, Deploy your app on Community
   Cloud.
Academic project note
This repository accompanies an academic capstone project on Big Data
Technologies and Cloud Computing. Metrics describe the experiments
performed in this project and should be interpreted within the stated
dataset, features and hardware limitations.

Live application: https://airline-flightpulse.streamlit.app/
https://airline-flightpulse.streamlit.app/ 
