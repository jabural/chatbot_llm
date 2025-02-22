# Specify the parent image from which we build
FROM python:3.12

# Set the working directory
WORKDIR /LLM_application_chatbot

# This copies the requirements.txt file from the local directory to the current directory (.) in the container
COPY src/requirements.txt .

# Install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# Copy every content from the local file to the image
COPY src/ src/


# This informs Docker that the container will listen on port 8000 at runtime.
EXPOSE 8000

# configure the container to run in an executed manner
CMD ["uvicorn", "src.main:app", "--host=0.0.0.0","--reload"]
