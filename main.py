from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model and tokenizer from Hugging Face Hub
model_checkpoint = "gvaishnavi/bilingual-translation"
tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

# Set the source language (English) and target language (Telugu)
text = "What are you doing?"

# Tokenize the input and generate translation
inputs = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(
    **inputs
)

# Decode and print the output
translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(translation)  # Output: మీరు ఏమి చేస్తున్నారు.
