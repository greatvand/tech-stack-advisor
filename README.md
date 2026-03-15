---
title: Tech Stack Advisor
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
app_file: app.py
pinned: false
license: apache-2.0
duplicable: true
---

# 🧠 Tech Stack Advisor – ML App (with Docker & Hugging Face Deployment)

**Tech Stack Advisor** is a hands-on machine learning project designed to teach you how to build, containerize, and deploy an ML-powered web application using Docker and Hugging Face Spaces.

> 🎯 This project is part of the **"Artificial Intelligence and Machine Learning (AI/ML) with Docker"** course from **School of DevOps**.

---

## 🚀 What You'll Learn

- Build and train a simple ML model using `scikit-learn`
- Create a UI using `Gradio`
- Containerize your app using a Dockerfile
- Push your Docker image to Docker Hub
- Deploy the Dockerized app on Hugging Face Spaces (free tier)

---

## 📁 Project Structure

```

tech-stack-advisor/
├── app.py             # Gradio web app
├── train.py           # Script to train and save ML model
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker build file (added during the lab)
├── model.pkl          # Trained ML model (generated after training)
├── encoders.pkl       # Encoders for categorical inputs (generated after training)
├── LICENSE            # Apache 2.0 license
└── README.md          # This guide

````

---

## 🧠 Step 1: Setup and Train Your ML Model

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/tech-stack-advisor.git
cd tech-stack-advisor
````

2. **Install dependencies**

(Optional: Use a virtual environment)

```bash
pip install -r requirements.txt
```

3. **Train the model**

```bash
python train.py
```

This creates:

* `model.pkl`: the trained ML model
* `encoders.pkl`: label encoders for input/output features

---

## 🖥️ Step 2: Run the App Locally (Without Docker)

```bash
python app.py
```

Visit the app in your browser at:

```
http://localhost:7860
```

---

## 🐳 Step 3: Add Docker Support

Create a file named `Dockerfile` in the root of the project:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

---

## 🔧 Step 4: Build and Run the Docker Container

1. **Build the image**

```bash
docker build -t tech-stack-advisor .
```

2. **Run the container**

```bash
docker run -p 7860:7860 tech-stack-advisor
```

Visit: `http://localhost:7860`

---

## ☁️ Step 5: Publish to Docker Hub

1. **Login to Docker Hub**

```bash
docker login
```

2. **Tag the image**

```bash
docker tag tech-stack-advisor <your-dockerhub-username>/tech-stack-advisor:latest
```

3. **Push it**

```bash
docker push <your-dockerhub-username>/tech-stack-advisor:latest
```

---

## 🌐 Step 6: Deploy to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create New Space**
3. Select:

   * **SDK**: Docker
   * **Repository**: Link to your GitHub repo with the Dockerfile
4. Hugging Face will auto-build and deploy your container.

---

## 🧪 Test Your Skills

* Can you swap the model in `train.py` for a `LogisticRegression` model?
* Can you add logging to show which inputs were passed?
* Try changing the Gradio layout or theme!

---

## 🧾 License

This project is licensed under the **Apache License 2.0**.
See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Credits

Created by \[Gourav Shah](https://www.linkedin.com/in/gouravshah) as part of the **AI/ML with Docker** course at **School of DevOps**.

---

> 🛠 Happy shipping, DevOps and MLOps builders!

