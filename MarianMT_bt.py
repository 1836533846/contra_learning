from transformers import MarianMTModel, MarianTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# MarianMT是一个基于Transformer的神经机器翻译模型，专门用于翻译任务。Hugging Face的Transformers库已经包含了MarianMT的预训练模型
# 使用了Helsinki-NLP/opus-mt-{src}-{tgt}模型来进行回译

def round_trip_translation(text, source_lang="en", target_lang="fr", model_name="Helsinki-NLP/opus-mt-{src}-{tgt}"):

    # 初始化源语言和目标语言的模型和分词器
    src_model_name = model_name.format(src=source_lang, tgt=target_lang)
    tgt_model_name = model_name.format(src=target_lang, tgt=source_lang)

    src_tokenizer = MarianTokenizer.from_pretrained(src_model_name)
    tgt_tokenizer = MarianTokenizer.from_pretrained(tgt_model_name)

    src_model = MarianMTModel.from_pretrained(src_model_name)
    tgt_model = MarianMTModel.from_pretrained(tgt_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_model.to(device)
    tgt_model.to(device)

    # 将原始文本翻译成目标语言
    translated_text = translate(text, src_tokenizer, src_model, device)
    # 将翻译后的文本翻译回原始语言
    back_translated_text = translate(translated_text, tgt_tokenizer, tgt_model, device)

    return back_translated_text

def translate(text, tokenizer, model, device):
    # 准备输入数据
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    # 生成翻译结果
    with torch.no_grad():
        output_ids = model.generate(input_ids, num_beams=4, max_length=512, early_stopping=True)

    # 将输出ID解码成文本
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return translated_text

# 示例
original_text = "This is an example of round-trip translation for data augmentation."
back_translated_text = round_trip_translation(original_text)
print("Original text:", original_text)
print("Back-translated text:", back_translated_text)
