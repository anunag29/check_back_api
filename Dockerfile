# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the source code into the container
COPY src /app/src

# Copy the requirements file into the container
COPY requirements.txt /app

COPY application.yaml /app

COPY ./models/config.json /app/models/
COPY ./models/generation_config.json /app/models/
COPY ./models/model.safetensors /app/models/
COPY ./models/special_tokens_map.json /app/models/
COPY ./models/preprocessor_config.json /app/models/
COPY ./models/tokenizer_config.json /app/models/
COPY ./models/merges.txt /app/models/
COPY ./models/vocab.json /app/models/
COPY ./models/tokenizer.json /app/models/

# Install any needed dependencies specified in requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user
# RUN useradd -r -u 1001 serviceuser

# Change ownership of application directory to the non-root user
# RUN chown -R serviceuser /app
# RUN chmod -R 755 /app

# Include /bin/sh for debugging
# RUN ln -sf /bin/bash /bin/sh

SHELL ["/bin/dash", "-c"]
# Rename /bin/sh to /bin/sh_tmp
RUN mv /bin/sh /bin/sh_tmp && \
    ln -sf /bin/false /bin/sh_tmp && \
    rm /bin/sh_tmp

# Rename /bin/bash to /bin/bash_tmp
RUN mv /bin/bash /bin/bash_tmp && \
    ln -sf /bin/false /bin/bash_tmp && \
    rm /bin/bash_tmp

# Rename /bin/rbash to /bin/rbash_tmp
RUN mv /bin/rbash /bin/rbash_tmp && \
    ln -sf /bin/false /bin/rbash_tmp && \
    rm /bin/rbash_tmp

# Rename /bin/dash to /bin/dash_tmp
RUN mv /bin/dash /bin/dash_tmp && \
    ln -sf /bin/false /bin/dash_tmp && \
    rm /bin/dash_tmp

# Switch to the non-root user
# USER serviceuser

# Command to run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
