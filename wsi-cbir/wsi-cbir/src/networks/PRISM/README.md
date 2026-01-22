---
license: cc-by-nc-nd-4.0
license_link: https://huggingface.co/paige-ai/Prism/resolve/main/LICENSE

tags:
- nlp
- code
- vision
- PyTorch

extra_gated_prompt: |

  This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the Prism Model and its derivatives, which include models trained on outputs from the Prism Model or datasets created from the Prism Model, is prohibited and requires prior approval. Please note that the primary email used to sign up for your Hugging Face account must match your institutional email to receive approval. By downloading the Prism Model, you attest that all information (affiliation, research use) is correct and up-to-date. Downloading the Prism Model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading the Prism model, you agree not to distribute, publish or reproduce a copy of the Prism Model. If another user within your organization wishes to use the Prism Model, they must register as an individual user and agree to comply with the terms of use. Requests from commercial entities will be rejected by default if no statement of use received separately. If you are a commercial entity, please contact the corresponding author of this paper.

  Further, by downloading the Prism model, you agree you will only use the Prism model for academic research purposes and will not use, or allow others to use, the Prism model to:

    1. Diagnose, cure, mitigate, treat, or prevent disease or any other conditions, including for Investigational Use Only (“IUO”), Research Use Only (“RUO”), commercial, clinical or other similar use, and including as a substitute for professional medical advice, a healthcare opinion, a diagnosis, treatment, or the clinical judgment of a healthcare professional, as no license or right is granted for any such purposes.

    2. Re-identify the deidentified data used to develop the Prism Model;

    3. Violate the law or others’ rights, including to:

        a. Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content;

        b. Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals;

        c. Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services;

        d. Engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices;

        e. Collect, process, disclose, generate, or infer the identity of individuals or the health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws;

        f. Engage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any products or services using the Prism Model or any related materials; and

        g. Create, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system.

    4. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including the use of the Prism Model as a medical device, clinical support, diagnostic tool, or other technology intended to be used in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions, including for Investigational Use Only (“IUO”), Research Use Only (“RUO”), commercial, clinical or similar use; and

    5. Intentionally deceive or mislead others, including representing that the use of the Prism Model or its outputs is human-generated.

  Further, you agree that you will appropriately disclose to end users any known dangers of your AI system.

extra_gated_fields:
  First and Last Name: text
  Institutional Email (must match your primary HuggingFace email): text
  I agree to the license and terms of use described above: checkbox
---

# Model card for PRISM

PRISM is a multi-modal generative foundation model for slide-level analysis of H&E-stained histopathology images.
Utilizing Virchow tile embeddings and clinical report texts for pre-training, PRISM combines these embeddings into a single slide embedding and generates a text-based diagnostic report.
These can be used for tasks such as cancer detection, sub-typing, and biomarker identification.
The model's slide encoder can be fine-tuned for specific classification tasks, leveraging both image and text data to enhance diagnostic performance and robustness.

> [!WARNING]
> This model is available solely for non-commercial research and evaluation purposes.
> It is not designed for clinical use and should not be employed to diagnose any diseases.
> The diagnostic reports generated by PRISM are intended for assessing the model's quality and may contain errors.
> Therefore, using the generated reports in clinical settings is strictly prohibited.
> The generated reports do not reflect the model's true performance for zero-shot or finetuning benchmark diagnostic tasks.

PRISM supports several modes of use:
- text report generation to describe tissue in H&E whole slide images
- zero-shot cancer detection and sub-typing using text prompts
- adaptation to new tasks via PRISM finetuning, or linear classifier on the slide embedding


## Model Details
- **Developed by:** Paige, NYC, USA and Microsoft Research, Cambridge, MA USA
- **Model Type:** Vision-Language Encoder-Decoder
- **Model Stats:**
  - Params (M): 558
- **Model Architecture:**
  - Encoder: Perceiver (https://doi.org/10.48550/arXiv.2103.03206)
  - Decoder: BioGPT (https://huggingface.co/microsoft/biogpt)
  - Model inputs: tile image embeddings and text captions
  - Tile image encoder: Virchow V1 (https://huggingface.co/paige-ai/Virchow)
- **Training Details:**:
  - Objective: CoCa (https://doi.org/10.48550/arXiv.2205.01917)
  - Precision: Mixed precision (`fp16`)
- **Paper:**
  - PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology: https://arxiv.org/abs/2405.10254
- **Pretraining Dataset:**: Internal dataset of 587 thousand whole slide images and 195 thousand clinical reports from Memorial Sloan Kettering Cancer Center.
- **License:** CC-BY-NC-ND-4.0


## Model Usage

### Direct use

PRISM is a vision-language model that can analyze whole slide images using the following methods:
- CLIP-style zero-shot classification via `zero_shot` method, or
- generate a tissue description in the image via `generate` method.

The model takes whole slide images in the form of tile embeddings from our Virchow model.
Please see https://huggingface.co/paige-ai/Virchow for instructions on how to use it
to generate embeddings for your whole slide image.

### Downstream use

You can use PRISM to compute slide embedding for downstream tasks such as slide-level classification.
The slide embedding can be further adapted to new tasks by finetuning the slide encoder of PRISM
on slide-level labels, e.g. biomarkers.

Slide embeddings are accessible via `slide_representations` method.

### Requirements

Using Python 3.10 or newer:

```
transformers~=4.53.0
torch>=2.3,<2.8
einops==0.8.0
environs==11.0.0
sacremoses==0.1.1

# # install xformers to use memory-efficient attention
# # set env `PERCEIVER_MEM_EFF_ATTN=true` to enable
# xformers==0.0.31
```

Also see `requirements.txt` file.

### Get the model

After gaining access to the model here, you will need to login to HuggingFace in the environment you wish to use the model.
This can be done from the command line:

```bash
python -m pip install huggingface_hub
huggingface-cli login
```

or in your Python code:

```python
from huggingface_hub import login

login()
```

Please refer to official HuggingFace [documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) for more details.

Test model access:

```python
from transformers import AutoModel
AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
```

### Sample inference code

This repo includes a file with Virchow tile embeddings for a WSI from the TCGA cancer database.
Download the embeddings to your local environment to run the inference example code below.

- TCGA case id: TCGA-B6-A0WZ (https://portal.gdc.cancer.gov/cases/08740d7f-5a5e-4dfa-bd48-7fbf228a7a28)
- Slide name: TCGA-B6-A0WZ-01Z-00-DX1.6CFB236E-36F5-43D6-8DE3-C4ECBD3C14C6.svs

The code is also available in `example_inference.py`.


```python
import torch
from transformers import AutoModel

# Load PRISM model.
model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
model = model.to('cuda')


# Load Virchow tile embeddings.
# See https://huggingface.co/paige-ai/Virchow on how to generate them
# given a whole slide image.
embedding_data = torch.load('tcga/TCGA-B6-A0WZ-01Z-00-DX1.6CFB236E-36F5-43D6-8DE3-C4ECBD3C14C6.pth')
tile_embeddings = embedding_data['embeddings'].unsqueeze(0).to('cuda')
print(tile_embeddings.shape)  # (batch_size, tile_seq_len, tile_embed_dim)
# > torch.Size([1, 12137, 2560])


# Compute slide embedding and latents. Only Perceiver is evaluated.
# We highly recommend running the model on a GPU in mixed precision (`fp16`)
# using `torch.autocast`.
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    reprs = model.slide_representations(tile_embeddings)
print(reprs['image_embedding'].shape)
# > torch.Size([1, 1280])
print(reprs['image_latents'].shape)
# > torch.Size([1, 512, 1280])
print(reprs['image_embedding'][0, :8])
# > tensor([ 0.0620, -0.2983,  0.5849, -0.7383,  0.1242, -0.7440, -0.4719, -0.2920],
#          device='cuda:0')


# Do zero-shot prediction using the slide embedding.
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    scores = model.zero_shot(
        reprs['image_embedding'],
        neg_prompts=['lobular carcinoma, invasive'],
        pos_prompts=['ductal carcinoma, invasive'],
    )
print(scores)
# > tensor([[0.0013, 0.9987]], device='cuda:0')


# Generate report using latent features.
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    genned_ids = model.generate(
        key_value_states=reprs['image_latents'],
        do_sample=False,
        num_beams=5,
        num_beam_groups=1,
    )
    genned_caption = model.untokenize(genned_ids)
print(genned_caption)
# > ['</s>Diagnosis: Moderately differentiated invasive ductal carcinoma '
# >  'with micropapillary features in breast tissue. </s>']


# Basic forward pass used in training.
# Computes slide embedding, text embedding, image latents (see Perceiver),
# next token logits, and similarity between slide and text embeddings
# used in contrastive alignment.
caption = model.tokenize(['some caption']).to('cuda')
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    output = model(input_ids=caption, tile_embeddings=tile_embeddings)
print(output.keys())
# > dict_keys(['logits', 'text_embedding', 'image_embedding',
# >            'image_latents', 'sim'])
```


## Terms of use

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the PRISM Model and its derivatives, which include models trained on outputs from the PRISM Model or datasets created from the PRISM Model, is prohibited and requires prior approval. Please note that the primary email used to sign up for your Hugging Face account must match your institutional email to receive approval. By downloading the PRISM Model, you attest that all information (affiliation, research use) is correct and up-to-date. Downloading the PRISM Model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading the PRISM model, you agree not to distribute, publish or reproduce a copy of the PRISM Model. If another user within your organization wishes to use the PRISM Model, they must register as an individual user and agree to comply with the terms of use. If you are a commercial entity, please contact the corresponding author.

Further, by downloading the PRISM model, you agree you will only use the PRISM model for academic research purposes and will not use, or allow others to use, the PRISM model to:

  1. Diagnose, cure, mitigate, treat, or prevent disease or any other conditions, including for Investigational Use Only (“IUO”), Research Use Only (“RUO”), commercial, clinical or other similar use, and including as a substitute for professional medical advice, a healthcare opinion, a diagnosis, treatment, or the clinical judgment of a healthcare professional, as no license or right is granted for any such purposes.

  2. Re-identify the deidentified data used to develop the PRISM Model;

  3. Violate the law or others’ rights, including to:

      a. Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content;

      b. Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals;

      c. Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services;

      d. Engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices;

      e. Collect, process, disclose, generate, or infer the identity of individuals or the health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws;

      f. Engage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any products or services using the PRISM Model or any related materials; and

      g. Create, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system.

  4. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including the use of the PRISM Model as a medical device, clinical support, diagnostic tool, or other technology intended to be used in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions, including for Investigational Use Only (“IUO”), Research Use Only (“RUO”), commercial, clinical or similar use; and

  5. Intentionally deceive or mislead others, including representing that the use of the PRISM Model or its outputs is human-generated.

Further, you agree that you will appropriately disclose to end users any known dangers of your AI system.


## Citation

Please cite the following work if you use this model in your research.

Shaikovski, George, Adam Casson, Kristen Severson, Eric Zimmermann et al. "PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology." arXiv preprint arXiv:2405.10254 (2024). https://doi.org/10.48550/arXiv.2405.10254

```
@article{shaikovski2024prism,
  title={PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology},
  author={Shaikovski, George and Casson, Adam and Severson, Kristen and Zimmermann, Eric and Wang, Yi Kan and Kunz, Jeremy D and Retamero, Juan A and Oakley, Gerard and Klimstra, David and Kanan, Christopher and others},
  journal={arXiv preprint arXiv:2405.10254},
  year={2024}
}
```


## Disclaimer

PRISM has been developed for research purposes and is not intended for diagnosis of real patients or projection/prediction of future disease possibilities.

Fairness evaluation cannot be completed due to limitations in the metadata.
Underlying biases of the training datasets may not be well characterized and may not be representative of all demographics.


## Acknowledgements

The results shown here (specifically, in the section "Sample inference code") are in whole or part based upon data generated by the TCGA Research Network: http://cancergenome.nih.gov/.
