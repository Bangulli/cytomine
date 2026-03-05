CYTOMINE_CONFIG={
    "encoder":'PGP', # the encoder architecture to use, currently only supports CHIEF
    "embeddings":'/embeddings', # where to store the embeddings at DATA_PATH
    "remove_bg":'dilated-otsu', # which background removal strategy to use
    'target_mpp': 0.5, # ~20X Magnification
    "full_precision":False, # use full precision
    "index_saving_interval":1800, # at which interval to save the index object at runtime
}