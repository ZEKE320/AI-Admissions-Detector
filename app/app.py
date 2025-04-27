import re
import string
from enum import Enum

import eli5
import joblib
import numpy as np
import scipy
import shap
import spacy
import streamlit as st
import torch
from app.custom_models import (
    HF_BertBasedModelAppDocs,
    HF_DistilBertBasedModelAppDocs,
)
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
from transformers import AutoTokenizer


class BaselineModel(Enum):
    LOGISTIC_REGRESSION = "Logistic Regression"
    NAIVE_BAYES = "Naive Bayes"
    LOGISTIC_REGRESSION_FOR_THESIS = "Logistic Regression (for thesis)"
    NAIVE_BAYES_FOR_THESIS = "Naive Bayes (for thesis)"


class TransformerModel(Enum):
    DISTILBERT = "DistilBERT-based model (BERT light)"
    DISTILBERT_FOR_THESIS = "DistilBERT-based model (BERT light, for thesis)"
    BERT = "BERT-based model"
    BERT_FOR_THESIS = "BERT-based model"


models_available = {
    BaselineModel.LOGISTIC_REGRESSION.value: "models/baseline_model_lr_longer.joblib",
    BaselineModel.NAIVE_BAYES.value: "models/baseline_model_nb_longer.joblib",
    TransformerModel.DISTILBERT.value: "ferdmartin/HF_DistilBertBasedModelAppDocs2",
    TransformerModel.BERT.value: "ferdmartin/HF_BertBasedModelAppDocs2",
    BaselineModel.LOGISTIC_REGRESSION_FOR_THESIS.value: "models/baseline_thesis_model_lr.joblib",
    BaselineModel.NAIVE_BAYES_FOR_THESIS.value: "models/baseline_thesis_model_nb.joblib",
    TransformerModel.DISTILBERT_FOR_THESIS.value: "models/distilbert-for-thesis",
}


def main() -> None:
    """
    Creates a Streamlit web app that classifies a given body of text as either human-made or AI-generated,
    using a pre-trained model.
    """
    global nlp, device

    nlp = spacy.load("en_core_web_sm")

    # device to run DL model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Streamlit app:
    st.set_page_config(page_title="AI/Human GradAppDocs", page_icon="🤖", layout="wide")
    st.title("Academic Application Document Classifier")
    st.header("Is it human-made 📝 or Generated with AI 🤖 ?  ")

    st.markdown(
        "AI-generated content has reached an unprecedented level of realism. The models on this website focus on identifying AI-generated application materials, such as Statements of Intent (SOI) and Letters of Recommendation (LOR). These models were trained using real-world SOIs and LORs, alongside their AI counterparts generated through prompts crafted using information from actual applications. An example of such prompts is as follows:"
    )
    st.markdown(
        '''"Write a statement of intent for a master's in Data Science at Fordham University. My undergraduate degree is in Mathematics, and my GPA is 3.45. I possess proficiency in python, java, matlab, software, calculus, and linear algebra."'''
    )
    st.markdown(
        "It's worth noting that these models are not trained to detect AI-enhanced text originating from human-authored drafts. Using tools like ChatGPT for document revision may be acceptable for some admissions committees, as they combine human creativity with AI language proficiency. However, our upcoming research will focus on identifying AI-modified texts, including paraphrasing, to ensure transparency and trustworthiness across various applications."
    )

    option = st.selectbox(
        "Select a model to use:",
        models_available,
        on_change=restore_prediction_state,
    )
    assert option in models_available, "Invalid model option selected"

    # Load the selected trained model
    if option in (
        TransformerModel.BERT.value,
        TransformerModel.DISTILBERT.value,
        TransformerModel.DISTILBERT_FOR_THESIS.value,
    ):
        tokenizer = load_tokenizer(option)
        model = load_model(option)
    else:
        model = joblib.load(models_available[option])

    text = st.text_area(
        "Enter either a statement of intent or a letter of recommendation:"
    )

    # Hide footer "made with streamlit"
    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Use model
    if st.button("Let's check this text!"):
        if text.strip() == "":
            st.error("Please enter some text")
        else:
            with st.spinner("Wait for the magic 🪄🔮"):
                # Use model
                if option in (
                    BaselineModel.NAIVE_BAYES.value,
                    BaselineModel.LOGISTIC_REGRESSION.value,
                    BaselineModel.NAIVE_BAYES_FOR_THESIS.value,
                    BaselineModel.LOGISTIC_REGRESSION_FOR_THESIS.value,
                ):
                    prediction, predict_proba = nb_lr(model, text)
                    st.session_state["sklearn"] = True
                else:
                    prediction, predict_proba = torch_pred(
                        tokenizer, model, format_text(text)
                    )
                    st.session_state["torch"] = True

            # Store the result in session state
            st.session_state["color_pred"] = "blue" if prediction == 0 else "red"
            prediction = pred_str(prediction)
            st.session_state["prediction"] = prediction
            st.session_state["predict_proba"] = predict_proba
            st.session_state["text"] = text

            # Print result
            st.markdown(
                f"I think this text is: **:{st.session_state['color_pred']}[{st.session_state['prediction']}]** (Confidence: {st.session_state['predict_proba'] * 100}%)"
            )

    elif "prediction" in st.session_state:
        # Display the stored result if available
        st.markdown(
            f"I think this text is: **:{st.session_state['color_pred']}[{st.session_state['prediction']}]** (Confidence: {st.session_state['predict_proba'] * 100}%)"
        )

    if st.button("Model Explanation"):
        # Check if there's text in the session state
        if "text" in st.session_state and "prediction" in st.session_state:
            if option in (
                BaselineModel.NAIVE_BAYES.value,
                BaselineModel.LOGISTIC_REGRESSION.value,
                BaselineModel.NAIVE_BAYES_FOR_THESIS.value,
                BaselineModel.LOGISTIC_REGRESSION_FOR_THESIS.value,
            ):
                with st.spinner("Wait for it 💭..."):
                    explainer = TextExplainer(sampler=MaskingTextSampler())
                    explainer.fit(st.session_state["text"], model.predict_proba)
                    html = eli5.format_as_html(
                        explainer.explain_prediction(target_names=["Human", "AI"])
                    )
            else:
                with st.spinner(
                    "Wait for it 💭... BERT-based model explanations take around 4-10 minutes. In case you want to abort, refresh the page."
                ):
                    # TORCH EXPLAINER PRED FUNC (USES logits)
                    def f(x):
                        tv = torch.tensor(
                            [
                                tokenizer.encode(
                                    v,
                                    padding="max_length",
                                    max_length=512,
                                    truncation=True,
                                )
                                for v in x
                            ]
                        ).to(device)
                        outputs = model(tv).detach().cpu().numpy()
                        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
                        val = scipy.special.logit(
                            scores[:, 1]
                        )  # use one vs rest logit units
                        return val

                    # build an explainer using a token masker
                    explainer = shap.Explainer(f, tokenizer)
                    shap_values = explainer([st.session_state["text"]], fixed_context=1)
                    html = shap.plots.text(shap_values, display=False)
            # Render HTML
            st.components.v1.html(html, height=500, scrolling=True)
        else:
            st.error(
                "Please enter some text and click 'Let's check!' before requesting an explanation."
            )


def format_text(text: str) -> str:
    """
    This function takes a string as input and returns a formatted version of the string.
    The function replaces specific substrings in the input string with empty strings,
    converts the string to lowercase, removes any leading or trailing whitespace,
    and removes any punctuation from the string.
    """

    doc = nlp(text)
    text = " ".join(
        [token.text for token in doc if token.ent_type_ not in ["PERSON", "DATE"]]
    )

    pattern1 = r"f\d+"
    pattern2 = r"\b[A-Za-z]+\d+\b"
    pattern3 = r"\[(.*?)\]"

    text = re.sub(pattern1, "", text)
    text = re.sub(pattern2, "", text)
    text = re.sub(pattern3, "", text)

    return (
        text.replace("REDACTED", "")
        .lower()
        .replace("  ", " ")
        .replace("[Name]", "")
        .replace("[your name]", "")
        .replace("\n your name", "")
        .replace("dear admissions committee,", "")
        .replace("sincerely,", "")
        .replace("[university's name]", "fordham")
        .replace("dear sir/madam,", "")
        .replace("– statement of intent  ", "")
        .replace(
            "program: master of science in data analytics  name of applicant:    ",
            "",
        )
        .replace("data analytics", "data science")
        .replace("| \u200b", "")
        .replace("m.s. in data science at lincoln center  ", "")
        .translate(str.maketrans("", "", string.punctuation))
        .strip()
        .lstrip()
    )


# Define the function to classify text
def nb_lr(model, text):
    # Clean and format the input text
    text = format_text(text)
    # Predict using either LR or NB and get prediction probability
    prediction = model.predict([text]).item()
    predict_proba = round(model.predict_proba([text]).squeeze()[prediction].item(), 4)
    return prediction, predict_proba


def torch_pred(tokenizer, model, text):
    # DL models (BERT/DistilBERT based models)
    cleaned_text_tokens = tokenizer(
        [text], padding="max_length", max_length=512, truncation=True
    )
    with torch.inference_mode():
        input_ids, att = (
            cleaned_text_tokens["input_ids"],
            cleaned_text_tokens["attention_mask"],
        )
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(att).to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(logits, 1)
        prediction = prediction.item()
        predict_proba = round(
            torch.softmax(logits, 1).cpu().squeeze().tolist()[prediction], 4
        )
        return prediction, predict_proba


def pred_str(prediction):
    # Map the predicted class to string output
    if prediction == 0:
        return "Human-made 🤷‍♂️🤷‍♀️"
    else:
        return "Generated with AI 🦾"


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_tokenizer(option):
    if option == TransformerModel.BERT.value:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
    elif (
        option == TransformerModel.DISTILBERT.value
        or option == TransformerModel.DISTILBERT_FOR_THESIS.value
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
    return tokenizer


@st.cache(allow_output_mutation=False, suppress_st_warning=True)
def load_model(option):
    # Use configured model paths for each transformer model
    model_path = models_available[option]
    if option == TransformerModel.BERT.value:
        model = HF_BertBasedModelAppDocs.from_pretrained(model_path).to(device)
    elif option == TransformerModel.DISTILBERT.value:
        model = HF_DistilBertBasedModelAppDocs.from_pretrained(model_path).to(device)
    elif option == TransformerModel.DISTILBERT_FOR_THESIS.value:
        # Load local thesis model weights from safetensors file
        model = HF_DistilBertBasedModelAppDocs.from_pretrained(
            model_path, local_files_only=True
        ).to(device)
    return model


# Check the model to use
def restore_prediction_state() -> None:
    if "prediction" in st.session_state:
        del st.session_state.prediction


if __name__ == "__main__":
    main()
