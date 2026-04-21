# Cytomine + WSI-CBIR

<div align="center">
  <img alt="Cytomine" src="https://raw.githubusercontent.com/cytomine/cytomine/main/docs/src/.vuepress/public/images/cytomine-uliege-logo.png">
</div>

Cytomine is an open-source platform for collaborative analysis of large-scale imaging data.

This repository provides the necessary files and instructions to build and launch the Cytomine product using Docker Compose.

This version of cytomine ships with the [WSI-CBIR](https://github.com/imi-bigpicture/WSI-CBIR) microservice to search vast data repositories efficiently.
Due to the microservice relying on foundation models a GPU is required in the environment to run, the service was tested on an Nvidia H100/100GB GPU, at least 32GB of GPU memory are required to avoid cuda out of memory errors when running at 20X magnification.

## Installation

This repo is a couple of commits behind the main repo so containers cant be fetched from the up-to-date registry, follow these instructions instead:
```bash
git clone https://github.com/Bangulli/cytomine
cd cytomine
git checkout wsi-cbir-1.1.0
docker compose config --services | grep -v '^wsi-cbir$' | xargs docker compose build
docker compose up -d
```

Per default this will launche the application with the heavy prov-gigapath model as an encoder, this will not run in weak GPU environments.
Therefore for test purposes you can fetch a much lighter version from the registry and plug it into the compose app:
```bash
git clone https://github.com/Bangulli/cytomine
cd cytomine
git checkout wsi-cbir-1.1.0
docker compose config --services | grep -v '^wsi-cbir$' | xargs docker compose build
docker pull lokuhn/wsi-cbir:1.1.1.chief
docker tag lokuhn/wsi-cbir:1.1.1.chief lokuhn/wsi-cbir:1.1.1
docker compose up -d
```

## Requirements

This microservice makes use of the [FAISS](https://github.com/facebookresearch/faiss) similarity serach library which keeps an index in memory at all times.
When 3 Million WSI are available on the repository this index will be ~10GB so the system needs at least 32 GB RAM, ideally much more to not run into any throtteling.

## License

[Apache 2.0](https://github.com/cytomine/cytomine/blob/main/LICENSE).

