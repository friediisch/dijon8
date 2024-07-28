# Use vanilla python image
FROM python:3.12.3

# Set working directory
WORKDIR /app

# Copy cotnent from current dir
COPY . /app

# Install packages
RUN pip install --no-cache-dir -r requirements-freeze.txt

# Define HF_TOKEN
ENV HF_TOKEN='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'

# download Llama models
RUN ["python", "src/download_models.py"]

# Run app.py when the container launches
CMD ["python", "src/main.py"]