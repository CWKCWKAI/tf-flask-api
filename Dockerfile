# base image
FROM python:3.7-slim

# create app directory 
WORKDIR /usr/src/app

# set env variables
ENV ABC="ABC"

# Install app dependencies 
EXPOSE 8080
COPY . . 
RUN pip install -r requirements.txt

# run app
CMD ["python", "./main.py"] # execute the above, but from python script 


# to build, run from same dir: 
#docker image build --tag <mynamehere> .
#docker run -d -p 8080:8080 <mynamehere>