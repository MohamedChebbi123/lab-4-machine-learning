FROM python:3.8-slim
WORKDIR /app
COPY MusicRecommender.py music.csv music_recommender.joblib ./  
RUN pip install pandas fastapi uvicorn joblib scikit-learn
EXPOSE 8000
CMD ["uvicorn", "MusicRecommender:app", "--host", "0.0.0.0", "--port", "8000"]
