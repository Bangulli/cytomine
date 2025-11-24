import argparse, sys
import os
from src.cli_methods import indexing
from src.cli_methods import retrieval
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from src.utils.switch_case import SwitchCase
from src.datasets.dataset_mgmt import get_dataset_factory
from src.networks.encoder_mgmt import get_encoder
from src.retrieval.retrieval import get_distance_type
from src.datasets.bg_removal import get_bg_rm_tool
      
#------------------------------------------------ MAIN PARSER ------------------------------------------------#
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True, description="Entrypoint to the Content Based Image Retrieval (CBIR) framework developed for IMI-BIGPICTURE at HES-SO.\n This container has functions for indexing databases as well as performing CBIR for query images on indexed databases")

#------------------------------------------------ INDEXING PARSER ------------------------------------------------#
####### FINALIZED ARGUMENTS ########
indexer = sub.add_parser("indexing", description="Computes and stores embeddings for all WSI files in a given BigPicture Dataset")
indexer.set_defaults(func=indexing.calculate_embeddings)
#------REQUIRED
indexer.add_argument("-n", "--name", required=True, help="Name of to the input dataset directory")
indexer.add_argument("-e", "--embeddings", required=True, help="Name of the embeddings directory") # NOTE renamed "output" to "embeddings" for consistency with retrieval
#------OPTIONS
# DEPRECATED: indexer.add_argument("--datareader", default="wsidicom", help=f"""Which library to use for I/O, wsidicom is recommended for DICOM files, supports {SwitchCase.get_options(get_dataset_factory)}""") # change default to wsidicom for compat with BigPic 
# DEPRECATED: indexer.add_argument("--patch-size", default=518, type=int, help="Size of the patches for the patch encoder")
# DEPRECATED: indexer.add_argument("--patch-stride", default=518, type=int, help="Size of the patch strides")
indexer.add_argument("--encoder", default="ProvGigaPath", help=f"""Which slide encoder to use, supports {SwitchCase.get_options(get_encoder)}""") 
indexer.add_argument("--level", default=1, type=int, help="Which scale level to use for inference")
indexer.add_argument("--remove-bg", default="dilated-otsu", help=f"Enable on-the-fly mask generation by using the specified method, supports {SwitchCase.get_options(get_bg_rm_tool)}")
#------FLAGS
indexer.add_argument("--full-precision", action="store_true", help="Use full precision, significantly slows down embedding claculation, NOT RECOMMENDED")

#------------------------------------------------ RETRIEVAL PARSER ------------------------------------------------#
retriever = sub.add_parser("retrieval", description="Finds matching images from an indexed database for a query image")
retriever.set_defaults(func=retrieval.find_k_similar)
#------REQUIRED
retriever.add_argument("-e", "--embeddings", required=True, help="path to an indexed WSI database")
#------OPTIONS
retriever.add_argument("-s", "--save", default=None, help="path/name of the json result file. If None, it is not saved.")
retriever.add_argument("-q", "--query", default=None, help="path to the query WSI file or query embedding")
retriever.add_argument("-k", "--k-best", default=3, type=int, help="the amount of closest matches to return")
retriever.add_argument("--datasets", default=None, nargs='+', help="Which specific dataset(s) to search in. If None will search in the entire embedded database")
# DEPRECATED: retriever.add_argument("--distance", default="euclidean", help=f"The distance metric used, supports {SwitchCase.get_options(get_distance_type)}")
retriever.add_argument("--metadata", default=None, help="Path to an xml file containing a filter configuration")
#------FLAGS
retriever.add_argument("--cpu", action="store_true", help="Run on cpu for use in ressource constrained environments, not recommended when gpu is available. This flag only has an effect if the query is an image, if its a precomputed embedding all computations happen on cpu directly. NOTE: PGP and TITAN are not supported!")

if __name__ == "__main__":
    args = p.parse_args()
    #environment_router.run_in_env(args.encoder)
    sys.exit(args.func(args))