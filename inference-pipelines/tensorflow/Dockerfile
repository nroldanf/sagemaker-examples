FROM tensorflow/tensorflow

RUN mkdir /app
WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

# docker run -v $(pwd):/app -it --rm tensorflow-embed bash