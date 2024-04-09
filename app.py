import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the GRU model
model = tf.keras.models.load_model(r"GRU_model.h5")

# Load the GRU tokenizer and max_sequence_len
with open(r"GRU_tokenizer.pickle", 'rb') as file:
    tokenizer = pickle.load(file)

with open(r"GRU_max_sequence_len.pickle", 'rb') as file:
    max_sequence_len = pickle.load(file)

# Load the GAN model
gan_model = tf.keras.models.load_model(r"GANmodel.h5")

# Load the GAN tokenizer and max_sequence_len
with open(r"GAN_tokenizer.pickle", 'rb') as file:
    gan_tokenizer = pickle.load(file)

with open(r"GAN_max_sequence_len.pickle", 'rb') as file:
    gan_max_sequence_len = pickle.load(file)

def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    generated_text = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = tf.argmax(predicted_probs, axis=1).numpy()[0]
        output_word = ''

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word
        generated_text += " " + output_word

    return generated_text

def main():
    st.title("Text Generation")
    st.write("Enter the seed text and the number of words to generate.")

    seed_text = st.text_input("Enter the seed text:")
    next_words = st.number_input("Enter the number of words to generate:", min_value=1, step=1)

    if st.button("Generate"):
        if seed_text and next_words:
            generated_gru_text = generate_text(seed_text, next_words, model, tokenizer, max_sequence_len)
            generated_gan_text = generate_text(seed_text, next_words, gan_model, gan_tokenizer, gan_max_sequence_len)
            st.write("Generated Text (GRU Model):")
            st.write(generated_gru_text)
            st.write("Generated Text (GAN Model):")
            st.write(generated_gan_text)

if __name__ == '__main__':
    main()
