from  transformers  import  AutoTokenizer, AutoModelWithLMHead, pipeline

model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
generator_tokenizer = AutoTokenizer.from_pretrained(model_name)
generator_model = AutoModelWithLMHead.from_pretrained(model_name)

# Generate an answer given the retrieved context
def generate_answer(query, context):
    input_text = f"question: {query} context: {context}"
    inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = generator_model.generate(inputs["input_ids"])
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)