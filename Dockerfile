# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files into the image
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose default Streamlit port
EXPOSE 8501

# Set Streamlit to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
