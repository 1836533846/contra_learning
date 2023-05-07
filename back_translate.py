import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def round_trip_translation(text, source_lang="en", target_lang="fr", model_name="t5-base"):
    # 初始化模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 将原始文本翻译成目标语言
    translated_text = translate(text, tokenizer, model, device, source_lang, target_lang)
    # 将翻译后的文本翻译回原始语言
    back_translated_text = translate(translated_text, tokenizer, model, device, target_lang, source_lang)

    return back_translated_text

def translate(text, tokenizer, model, device, source_lang, target_lang):
    # 准备输入数据
    input_text = f"translate {source_lang} to {target_lang}: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

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
