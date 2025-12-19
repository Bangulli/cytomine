# Whole Slide Image Content Based Image Retrieval
Based on the [WSI-CBIR](https://github.com/Bangulli/WSI-CBIR) repository.
This microservice currently implements the [CHIEF](https://github.com/hms-dbmi/CHIEF) architecture under the AGPL-3.0 License.

## Compose
The image is included as a service in the superior compose application as such:

```Dockerfile
  wsi-cbir:
    stop_grace_period: 2m
    image: cytomine/wsi-cbir:latest
    restart: unless-stopped
    volumes:
      - ${DATA_PATH:-./data}/wsi-cbir/embeddings:/embeddings
      - ${DATA_PATH:-./data}/images/:/images
    environment:
      API_BASE_PATH: /wsi-cbir
      DATA_PATH: /data
    networks:
      host_network:
        ipv4_address: 172.16.238.16
    shm_size: "20g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Volume Mounts
To run the service 3 mounted volumes are required:
/embeddings: This directory is used to store and load the Index, accompanying files and the individual embeddings for each image.
/images: This is the directory where the microservice sources the uploaded images from to use for embeddings.

## Variables
The container needs to have GPU support available to be feasible for large scale image embedding computation.
The shm is set fairly high because the I/O of the images is quite memory intensive.
The container is available on 0.0.0.0 port 6001
The configuration can be adjusted in the [config file](./wsi-cbir/src/config.py)

## Registry
The image is available from docker hub: lokuhn/wsi-cbir:latest