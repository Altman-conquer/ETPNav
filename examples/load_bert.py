from transformers import AutoTokenizer

# 从本地缓存加载模型（必须写绝对路径），避免hugging face connect error的报错
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='/home/zhandijia/.cache/huggingface/transformers/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/')
