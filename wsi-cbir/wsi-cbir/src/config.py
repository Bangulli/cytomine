CYTOMINE_CONFIG={
    "encoder":'CHIEF', # the encoder architecture to use, currently only supports CHIEF
    "embeddings":'/embeddings', # where to store the embeddings at DATA_PATH
    "remove_bg":'dilated-otsu', # which background removal strategy to use
    "level":1, # on which level to compute the embeddings
    "full_precision":False, # use full precision
    "index_saving_interval":600, # at which interval to save the index object at runtime
}