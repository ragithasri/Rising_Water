# Flood Prediction — Structured Project

## Epic 2: Visualizing and Analysing the Data
`rising-water.py` imports packages, reads `flood dataset.xlsx`, prints descriptive analysis, and creates univariate and multivariate charts.

## Epic 3: Data Pre-processing
The script handles missing values, clips numeric outliers, supports categorical encoding, splits the dataset, and scales numeric values.

## Epic 4: Model Building
Decision Tree, Random Forest, KNN, and XGBoost are trained, compared, evaluated, and the best pipeline is saved in `outputs/`.

## Epic 5: Application Building
The Flask app provides an HTML prediction form and a results page.

### Run

```powershell
python -m pip install -r requirements.txt
python rising-water.py
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.
