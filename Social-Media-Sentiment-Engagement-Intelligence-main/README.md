# Social Media Sentiment Analysis & Engagement Prediction

An end-to-end machine learning project that analyzes social media posts, explores engagement and sentiment patterns, trains machine learning models, and provides an interactive Flask web application for real-time predictions.

## Project Overview

This project is organized as a complete data analytics and machine learning workflow:

1. **Data Acquisition & SQL** – loads the social media dataset and stores it in SQLite.
2. **Exploratory Data Analysis (EDA)** – checks data quality, handles missing values/outliers, and generates analytical visualizations.
3. **Machine Learning** – trains:
   - a Random Forest classifier for sentiment prediction
   - a Random Forest regressor for engagement-score prediction
4. **Model Persistence** – saves trained models and categorical encoders as `.pkl` files.
5. **Web Deployment** – exposes the models through a Flask application with a responsive web interface.

## Key Features

- Social media dataset with 10,000 posts
- Sentiment classification: **Negative, Neutral, Positive**
- Engagement score prediction
- Platform and language encoding
- SQLite database integration
- EDA visualizations
- Pre-trained model artifacts
- Interactive Flask prediction interface
- Production-friendly Flask startup configuration using the `PORT` environment variable

## Dataset

The project uses `data/social_media.csv`.

Main fields include:

| Category | Columns |
|---|---|
| Identification | `Post_ID` |
| Platform/content | `Platform`, `Post_Content` |
| Engagement | `Likes`, `Shares`, `Comments_Count`, `Engagement_Score` |
| Time | `Post_Timestamp` |
| Audience | `Follower_Count`, `Account_Age_Days`, `Verified_User` |
| Content metadata | `Hashtags_Used`, `Language` |
| ML target | `Sentiment_Label` |
| Moderation | `Flagged_Content` |

The dataset contains **10,000 records and 15 columns**.

## Machine Learning Pipeline

### Sentiment Classification

**Features**

- `Platform`
- `Likes`
- `Shares`
- `Comments_Count`
- `Verified_User`
- `Language`

**Target**

- `Sentiment_Label`

**Algorithm**

- `RandomForestClassifier`
- 100 estimators
- `random_state=42`
- 80/20 train-test split

### Engagement Prediction

**Features**

- `Follower_Count`
- `Hashtags_Used`
- `Platform`

**Target**

- `Engagement_Score`

**Algorithm**

- `RandomForestRegressor`
- 100 estimators
- `random_state=42`
- 80/20 train-test split

> Note: The supplied notebook trains and saves the models but does not currently report evaluation metrics such as accuracy, precision/recall/F1 or MAE/RMSE. For a stronger ML portfolio, add those metrics before claiming model performance.

## Project Structure

```text
Sentiment Analysis/
├── app.py
├── Project.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── social_media.csv
├── graphs/
│   ├── Correlation_Heatmap.jpg
│   ├── Day of Week Activity.jpg
│   ├── Engagement vs Sentiment Analysis.jpg
│   ├── Hour of Highest User Activity.jpg
│   └── Language Dominance by Platform.jpg
├── models/
│   ├── sentiment_model.pkl
│   ├── engagement_model.pkl
│   ├── platform_encoder.pkl
│   └── language_encoder.pkl
├── templates/
│   └── index.html
└── social_media.db
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Using the Application

### Sentiment Prediction

Enter:

- Platform
- Likes
- Shares
- Comments
- Verified-user status
- Language

The application returns one of:

- Negative Sentiment
- Neutral Sentiment
- Positive Sentiment

### Engagement Prediction

Enter:

- Follower count
- Number of hashtags
- Platform

The application returns a predicted engagement score.

## Notebook Workflow

`Project.ipynb` contains the main analysis workflow:

- data loading and inspection
- duplicate/null checks
- SQLite database creation
- data cleaning
- outlier handling
- exploratory analysis
- visualization
- feature encoding
- model training
- model serialization

## Visualizations

The `graphs/` directory contains the project's generated analytical charts, including:

- Correlation heatmap
- Day-of-week activity
- Engagement vs sentiment
- Hourly user activity
- Language dominance by platform

## SQL

The SQLite database is stored as `social_media.db`.

The notebook demonstrates importing the CSV into the `Social_Media_Data` table and performing SQL-based data analysis.

## Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **SQLite**
- **Jupyter Notebook**
- **Flask**
- **HTML/CSS**
- **Chart.js**
- **Font Awesome**

## Important Notes

- The `.pkl` model files are already included so the Flask application can run without retraining.
- The dataset and SQLite database are included for reproducibility.
- Do not commit API keys, passwords, `.env` files, virtual environments, or other secrets.
- Pickled models should only be loaded from trusted sources.
- Scikit-learn model files can be version-sensitive. If a model loading warning/error occurs after dependency upgrades, retrain the models using the environment specified for the project.

## Future Improvements

- Add model evaluation metrics and a model-comparison section.
- Add confusion matrix and classification report.
- Add MAE/RMSE/R² for engagement prediction.
- Add API endpoints returning JSON predictions.
- Add prediction history and downloadable reports.
- Add interactive dashboard filters.
- Containerize the application with Docker.
- Deploy the Flask application to a cloud platform.

## Author

**Mathavan**

This project was developed as an end-to-end data analytics and machine learning project, covering data preparation, SQL, EDA, machine learning, model persistence, and web deployment.
