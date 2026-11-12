<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=8A2A78&height=120&section=header" alt="header"/> 
<h1 align="center">🤖 Human and AI-generated Movie Reviews Classifier</h1>
<h4 align="center">Status: <i>Completed</i></h4>
<h4 align = "center"> Authors: <a href = 'https://github.com/husnazainurin'>Husna Zainurin</a> and Norhanim Fatihah Ahmad</h4>
<hr>

<p align = "center">A dashboard that classify between human-written and AI-generated favorite movie reviews, just to understand AI language framework a little bit better.</p>

## 🏆 Key Results
* Achieved 97.7% mean accuracy across 5 cross-validation evaluation.
* Deployment in interactive dashboard. Users can enter their own written review, or prompt any AI tools to challenge the model. Check it out [here](#link)!

*Limitation*: We know that AI prompts are usually more complex to fit each users use-case. The dataset collection only uses 1 prompt repeatedly, using multiple accounts to replicate variability. The  generated texts are taken as-is, without any changes/re-prompt. Hence, this model is suitable for this specific prompt, and it's performance on any other use-cases are not explored.

---
## 📚 Table of Contents
* [About the Project](#about-the-project)
* [Dataset Used](#dataset-used)
* [Methodology Overview](#methodology-overview)
* [Performance Overview](#performance-overview)
* [Future Interests](#future-interests)

## About the Project
Generative artificial intelligence (AI) has been advancing at a very rapid pace due to its rising popularity in recent years. Due to that, generative AI has been exploited in various industries involving human writing, especially in education, journalism, and online platforms. 

This project is conducted to understand just a little bit more about the difference between AI and human writing. Hence, we tried to develop a model that aims to classify human and AI generated movie review text.

## Dataset Used
We asked both humans and generative AIs to answer this question:
<h4 align = "center">In about 50 words, tell us about the best TV show you have ever watched, and why? Explain a little, like your favorite character or scene.</h4>

|Class|Source|Description|
|---|---|---|
|Human Generated Review|Google Form Survey|The human-written essays are obtained by conducting an e-survey that was distributed to 45 respondents with overall excellent English language proficiency. These respondents were asked to write about their favourite TV show in approximately 50 words without any help of AI chatbots.|
|AI-Generated Essays|Gemini, ChatGPT and Microsoft Copilot|For consistency, all three chatbots were prompted with the exact same word-for-word question used in the e-survey for human respondents.|

## Methodology Overview
This is an overview of the methodology. I considered two models for this task, which is Logistic Regression (LR) and Support Vector Machine (SVM), as they are suitable for numerical TF-IDF as the main predictor.
<img width="758" height="339" alt="image" src="https://github.com/user-attachments/assets/a4b14517-693e-4e29-8e29-1583d4d33c7c" />

## Performance Overview

Below is the classification report of the test set:

<img width="494" height="273" alt="image" src="https://github.com/user-attachments/assets/bad1344a-235c-43a9-a0dd-23c4f5b59d8a" />

## Future Interests
We are very keen on exploring more LLM frameworks to study more universal use-cases, instead of a specific prompt. 


<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=8A2A78&height=120&section=footer" alt="footer"/>
