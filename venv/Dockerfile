FROM python
WORKDIR /app
RUN 
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD [ "python", "app.py","Parkinson.ipynb" ]