import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import faiss
from src.networks.encoder_mgmt import DIMS
from src.utils.metadata_filtration import load_filter_deps, get_meta_with_codes
import numpy as np
import json
import pathlib as pl
import torch


class Index:
    def __init__(self, path, dims=768):
        """Constructor

        Args:
            path (Path): Path to the home embedding directory in which the indexer lives
            dims (int, optional): Embedding dimension size. Defaults to 768.
        """
        self.path = pl.Path(path)
        self.index = faiss.IndexFlatL2(dims)
        self.index = faiss.IndexIDMap(self.index)
        self.idx2id_mapping = {}
        self.id2idx_mapping = {}
        self.metadata = {}
        
        
    def search(self, query, k):
        """Searches the index with a query and returns k best results, uses Euclidean distance

        Args:
            query (np.array): The numpy array containing the query embedding
            k (int): The amount of best matches to find

        Returns:
            tuple: lists of best match distances and id strings
        """
        dists, idxs = self.index.search(query, k)
        dists, idxs = dists.squeeze().tolist(), idxs.squeeze().tolist()
        ids = map(self.idx2id_mapping.__getitem__, [str(i) for i in idxs])
        return ids, dists
    
    def search_subset(self, query, k, ids):
        """Searches a subset of the index, defined by the ids, with a query and returns k best results, uses Euclidean distance

        Args:
            query (np.array): The numpy array containing the query embedding
            k (int): The amount of best matches to find
            ids (list): list of ID strings to search

        Returns:
            tuple: lists of best match distances and id strings
        """
        params = faiss.SearchParametersIVF()
        ids_to_search = self._get_ids_to_search(ids)
        params.sel = faiss.IDSelectorArray(ids_to_search.size, faiss.swig_ptr(ids_to_search))
        # mask = np.zeros(self.index.ntotal, dtype="int64")
        # mask[ids_to_search] = 1
        # params.sel = faiss.IDSelectorBatch(ids_to_search.size, faiss.swig_ptr(mask))
        dists, idxs = self.index.search(query, k, params=params)
        dists, idxs = dists.squeeze().tolist(), idxs.squeeze().tolist()
        ids = map(self.idx2id_mapping.__getitem__, [str(i) for i in idxs])
        return ids, dists
        
    def _get_ids_to_search(self, ids):
        """Get a list of indexes for the list of ID strings

        Args:
            ids (list): List of ID strings

        Returns:
            np.array: Numpy array of the correspondings indexes of the list of ids.
        """
        ids_to_search = list(map(self.id2idx_mapping.__getitem__, [id.split('.')[0] for id in ids]))
        return np.asarray(ids_to_search, dtype='int64')
    
    def filter_metadata(self, conditions, ids=None):
        """A bad implementation of a metadata filter that supports some and/or logic to customize queries

        Args:
            conditions (dict): Dictionary where the keys are the variables and the values are the conditions
            ids (list, optional): List of preselected ids, if none use all present in the index. Defaults to None.

        Returns:
            list: The filtered ids
        """
        subset = []
        if ids is None: ids=list(self.metadata.keys())
        for id in ids:
            id_meta = self.metadata[id]
            is_include = []
            for k, requested in conditions.items():
                ### check ors
                OR = requested.split('|')
                or_is_true = False
                for or_condition in OR:                   
                    ### check ands
                    AND = or_condition.strip().split('&')
                    and_is_true = []
                    for and_condition in AND:
                        stripped = and_condition.strip()
                        if stripped.startswith('!'):
                            ### check nots
                            if f'{k}_code' in list(id_meta.keys()):
                                if (stripped[1:] not in id_meta[k]) and (stripped[1:] not in id_meta[f'{k}_code']):
                                    and_is_true.append(True)
                                else: and_is_true.append(False)
                            else:
                                if (stripped[1:] not in id_meta[k]):
                                    and_is_true.append(True)
                                else: and_is_true.append(False)
                            ### check nots
                        else:
                            if f'{k}_code' in list(id_meta.keys()):
                                if (stripped in id_meta[k]) or (stripped in id_meta[f'{k}_code']):
                                    and_is_true.append(True)
                                else: and_is_true.append(False)
                            else:
                                if (stripped in id_meta[k]):
                                    and_is_true.append(True)
                                else: and_is_true.append(False)
                    ### check ands
                    if all(and_is_true): 
                        or_is_true = True
                if or_is_true: is_include.append(True)
                else: is_include.append(False)
                ### check ors     
            if all(is_include): subset.append(id)
        print(f'== {len(subset)} samples fulfill the filter conditions')
        return subset
        
    def add(self, samples, ids, meta): # samples are the embeddings arrays, ids are the corresponding image paths
        """Add embedding data into the index

        Args:
            samples (np.array): the embeddings in an (n, d) np.array
            ids (list): A list of the ID strings of the corresponding images

        Returns:
            tuple: The index range of the subset
        """
        idxs = np.arange(self.ntotal, self.ntotal+samples.shape[0])
        self.index.add_with_ids(samples, idxs)
        for idx, id in zip(idxs, ids):
            self.idx2id_mapping[str(idx)]=id
            self.id2idx_mapping[id]=str(idx)
            self.metadata[id]=get_meta_with_codes(id.split('/')[-1], meta[0][id.split('/')[-1]], meta[1])
        self.save()
        return idxs[0], idxs[-1]
        
    def add_dir(self, dir):
        """Add a directory of embeddings into the index DB. Is meant to be run on a dataset subdirectory of an embedding directory in src.cli_methods.indexing

        Args:
            dir (str, Path): The path to the directory

        Returns:
            tuple: The index range of the subset
        """
        dir = pl.Path(dir)
        ids = []
        embs = []
        for file in [f for f in os.listdir(dir) if f.endswith('.pth')]:
            emb = torch.load(dir/file, map_location='cpu', weights_only=False).squeeze().numpy()
            embs.append(emb)
            ids.append(f"{dir.name}/{file.replace('_embedding.pth', '')}")
        xmls = load_filter_deps(dir.name)
        return self.add(np.stack(embs), ids, meta=xmls)
        
    @property
    def ntotal(self):
        """Returns the total number of samples in the index

        Returns:
            int: Total number of samples in the index
        """
        return self.index.ntotal

    def load(self):
        """Loads the object from disk

        Returns:
            Index: the loaded instance of the object
        """
        self.index=faiss.read_index(str(self.path/'index.faiss'))
        with open(self.path/'mapping.json', 'r') as file:
            self.idx2id_mapping=json.load(file)
        self.id2idx_mapping = {v: int(k) for k, v in self.idx2id_mapping.items()}
        with open(self.path/'metadata.json', 'r') as file:
            self.metadata=json.load(file)
        return self
    
    def save(self):
        """Saves the object to disk as an index.faiss and a mapping.json
        """
        faiss.write_index(self.index, str(self.path/'index.faiss'))
        with open(self.path/'mapping.json', 'w') as file:
            file.seek(0)
            json.dump(self.idx2id_mapping, file, indent=4)
        with open(self.path/'metadata.json', 'w') as file:
            file.seek(0)
            json.dump(self.metadata, file, indent=4)
            
