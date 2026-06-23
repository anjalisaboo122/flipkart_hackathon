# 🚦 traffiKart — Real-Time Traffic Intelligence

Welcome to **traffiKart**, a dynamic traffic disruption and predictive policing dashboard developed for the **Bengaluru Traffic Police** at the Flipkart Gridlock Hackathon.

This project processes historical parking violation data, intersects it with live MapmyIndia (Mappls) traffic congestion data to quantify economic disruption, and outputs optimal, AI-driven predictive patrol routes.

---

## 🚀 Live Demo

If you are just looking to explore the interface and our simulation, you can view the live demo here (it operates in "Demo Mode" using historical fallback data to bypass IP restrictions):
**[Insert Your Streamlit Cloud Link Here]**

---

## 💻 How to Run Locally (For Reviewers)

To experience the **full, live functionality** of the dashboard (including real-time API calls), you must run the application locally on your machine and provide a valid MapmyIndia API Token. 

### Step 1: Environment Setup
1. Clone this repository.
2. Open your terminal and navigate to the repository directory.
3. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Generate a Mappls (MapmyIndia) Token
Because Mappls enforces strict IP whitelisting on their API keys, you will need to generate your own free token to test the live API functionality on your local network.

1. Go to the **[Mappls API Developer Portal](https://about.mappls.com/api/)** and click **Get Started** / **Sign Up**.
2. Create a free developer account.
3. Once logged into the dashboard, navigate to the **Credentials** or **API Keys** section.
4. Generate a new API Key/Token (ensure it has access to routing and traffic APIs).
5. Copy your **Access Token**.

### Step 3: Configure the Environment Variables
1. In the root directory of this repository, create a new file named exactly `.env`.
2. Open the `.env` file and paste your token inside it like this:
```env
MAPPLS_TOKEN=paste_your_token_here
```
*(Note: Do not put quotes around the token, just paste the raw string).*

### Step 4: Launch the Dashboard
With your `.env` file saved, run the Streamlit application:
```bash
streamlit run app.py
```

The dashboard will automatically open in your web browser at `http://localhost:8501`. 
Because your `.env` file is present, the app will detect your token, disable "Demo Mode", and the **Fetch Live Congestion Data** button in Tab 4 will become fully active!

---

## 🧠 Project Architecture
* **Frontend:** Streamlit, custom CSS glassmorphism, Plotly, PyDeck (Hexagon layers), Folium.
* **Backend Models:** Prophet (time-series forecasting), custom disruption indices, AI spatial clustering.
* **Data Processing:** Highly compressed Parquet formatting, Pandas, Geopy.
* **CCTV Model Integration:** YOLOv8 Nano object detection pipeline.
