from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


class BLIP2EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")
        
        self.embedding_config.model_init_params["device_map"] = 0
        print(self.embedding_config.model_init_params)

        self.embedding_model = Blip2ForConditionalGeneration.from_pretrained(**self.embedding_config.model_init_params)
        self.processor = Blip2Processor.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_model.to("cuda")
        #self.embedding_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        #self.embedding_dim = self.embedding_model.config.hidden_size

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        
        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                "torch_dtype": self.global_config.embedding_model_dtype,
                # **kwargs
            },
            "encode_params": {
                "truncation": True,
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                #"instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                #"num_workers": 32,
                "text": "",
                "return_tensors": "pt",
                "padding": True,
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    # def _add_eos(self, texts: List[str]) -> List[str]:
    #     # Adds EOS token to each text
    #     return [text + self.embedding_model.tokenizer.eos_token for text in texts]
    
    
    def clean_text(self, text):
        return ''.join(c for c in text if c.isprintable()).strip()


    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        try:
            del params["norm"]
        except:
            pass
            
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["text"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            del params["instruction"]

        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            #params["text"] = texts  # self._add_eos(texts=texts)
            params["text"] = [f"{params['text']}{text}" for text in texts]
            params["images"] = [Image.new('RGB', (224, 224), color=(255, 255, 255)) for i in range(len(params["text"]))]  # Pure white dummy img
            
            img_paths = [path[path.find("Image path for RAG: ")+20:] if path.find("Image path for RAG: ") != -1 else '' for path in params["text"]]
            
            for i in range(len(params["text"])):
                if params["text"][i].find("Image path for RAG: ") != -1:
                    params["text"][i] = params["text"][i][:params["text"][i].find("Image path for RAG: ")]
                
            for i in range(len(img_paths)):
                if img_paths[i] == '':
                    pass
                else:
                    params["images"][i] = Image.open(img_paths[i])
                    params["text"][i] = params["text"][i] + "\n The above is the context surrounding the image, now describe the image in as much detail as you can."
            
            #print(params)
            try:
                del params["norm"]
            except:
                pass
            inputs = self.processor(**params).to("cuda")
            #results = self.embedding_model.get_text_features(**inputs)
            for i in range(len(img_paths)):
                if img_paths[i] == '':
                    pass
                else:
                    input1 = self.processor(images=params["images"][i], text="Describe the image in as much detail as you can.", return_tensors="pt").to("cuda")
                    output1 = self.embedding_model.generate(**input1, max_new_tokens=150, do_sample=True, temperature=0.8, top_p=0.95, num_beams=5,)
                    output_text = self.processor.tokenizer.decode(output1[0], skip_special_tokens=True)
                    print(output_text)
                    params["text"][i] = params["text"][i] + "\n The contents of the image are as follows: " + output_text
                    print(params["text"][i])
                
            inputs = self.processor(**params).to("cuda")
            
            
            with torch.no_grad():
                vision_outputs = self.embedding_model.vision_model(inputs.pixel_values)[0]
                outputs = self.embedding_model.qformer(
                    query_embeds=self.embedding_model.query_tokens.expand(inputs.pixel_values.shape[0], -1, -1),
                    encoder_hidden_states=vision_outputs,
                    encoder_attention_mask=None,
                    return_dict=True,
                )
                qformer_embeds = outputs.last_hidden_state  # (batch, queries, hidden_dim)
            #print("if: ", qformer_embeds.shape)
            results = qformer_embeds.mean(dim=1)  # (batch, hidden_dim)
            #print("if: ", results.shape)

        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                params["text"] = [self.clean_text(f"{item}") for item in texts[i:i + batch_size]]
                #params["text"] = params["text"] + [params["text"][0]] * (batch_size - len(params["text"]))
                #params["text"] = texts[i:i + batch_size]
                params["images"] = [Image.new('RGB', (224, 224), color=(255, 255, 255)) for i in range(len(params["text"]))]  # Pure white dummy img
                img_paths = [path[path.find("Image path for RAG: ")+20:] if path.find("Image path for RAG: ") != -1 else '' for path in texts[i:i + batch_size]]
                
                for i in range(len(img_paths)):
                    if img_paths[i] == '':
                        pass
                    else:
                        params["images"][i] = Image.open(img_paths[i])
                        params["text"][i] = params["text"][i] + "\n The above is the context surrounding the image, now describe the image in as much detail as you can."
                        
                
                for i, t in enumerate(params["text"]):
                    try:
                        self.processor(params["images"][i], text=t, return_tensors="pt")
                    except Exception as e:
                        print(f"Text {i} caused tokenizer failure: {t[:100]}...")
                        raise

                
                #inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
                #inputs = move_inputs_to_device(inputs, model.hf_device_map)


                # Remove unsupported keys
                try:
                    params.pop("norm", None)
                except:
                    pass
                #inputs = self.processor(**params).to("cuda")
                
                

                
                
                for i in range(len(img_paths)):
                    if img_paths[i] == '':
                        pass
                    else:
                        input1 = self.processor(images=params["images"][i], text="Describe the image in as much detail as you can.", return_tensors="pt").to("cuda")
                        output1 = self.embedding_model.generate(**input1, max_new_tokens=150, do_sample=True, temperature=0.8, top_p=0.95, num_beams=5,)
                        output_text = self.processor.tokenizer.decode(output1[0], skip_special_tokens=True)
                        print(output_text)
                        params["text"][i] = params["text"][i] + "\n The contents of the image are as follows: " + output_text
                        print(params["text"][i])
                
                inputs = self.processor(**params).to("cuda")
                
                with torch.no_grad():
                    #vision_outputs = self.embedding_model.vision_model(inputs.pixel_values)[0]
                    vision_outputs = self.embedding_model.vision_model(inputs["pixel_values"])[0]
                    outputs = self.embedding_model.qformer(
                        query_embeds=self.embedding_model.query_tokens.expand(inputs["pixel_values"].shape[0], -1, -1),
                        #query_embeds=self.embedding_model.query_tokens.expand(inputs.pixel_values.shape[0], -1, -1),
                        encoder_hidden_states=vision_outputs,
                        encoder_attention_mask=None,
                        return_dict=True,
                    )
                    
                    #outputs = self.embedding_model(
                    #    pixel_values=inputs["pixel_values"],
                    #    input_ids=inputs.get("input_ids", None),
                    #    attention_mask=inputs.get("attention_mask", None),
                    #    return_dict=True,
                    #)
                    
                    qformer_embeds = outputs.last_hidden_state  # (batch, queries, hidden_dim) 8, 32, 768
                    
                    #qformer_embeds = outputs.qformer_hidden_states[-1]  # (batch, num_query_tokens, hidden_dim)
                #qformer_embeds = qformer_embeds.transpose(1, 2)  # (batch, hidden_dim, num_query_tokens)
                #pooled = self.pooling(qformer_embeds).squeeze(-1)  # (batch, hidden_dim)


                #print("else: ", qformer_embeds.shape)
                # Optionally, you could pool or flatten here:
                pooled_embeds = qformer_embeds.mean(dim=1)  # (batch, hidden_dim)
                #print("else: ", pooled_embeds.shape)
                #embeddings.append(pooled_embeds.cpu())

                results.append(pooled_embeds)
                
                ##############
                pbar.update(batch_size)
                
                #print("results shape: ", len(results))
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.detach().numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        #print("final shape: ", len(results))
        return results
        
        
        
# batch-ify this and use this as batch_encode()
# from transformers import CLIPProcessor, CLIPModel
# import torch
# from PIL import Image

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# inputs = processor(
#     text=["a red apple"],
#     images=image,
#     return_tensors="pt",
#     padding=True
# )
# with torch.no_grad():
#     outputs = model(**inputs)
#     text_embeds = outputs.text_embeds  # [1, 512]
#     image_embeds = outputs.image_embeds  # [1, 512]

