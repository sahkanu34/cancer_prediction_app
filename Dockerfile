FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install pandas plotly seaborn numpy scikit-learn streamlit matplotlib

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py"]