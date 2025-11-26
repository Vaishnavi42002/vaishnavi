from flask import Flask, render_template, request
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

app = Flask(__name__)

# Load model and tokenizer
model_checkpoint = "gvaishnavi/bilingual-translation"
tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    input_text = ""

    if request.method == "POST":
        input_text = request.form["input_text"]

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate translation
        output_tokens = model.generate(**inputs)

        # Decode output
        translation = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return render_template("index.html", translation=translation, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
