# Gridlock Dashboard Handoff

This project contains the backend traffic disruption and predictive patrol models for the Gridlock 2.0 hackathon. It processes historical parking violation data, intersects it with live Mappls traffic congestion data to quantify economic disruption caused by violations, and uses a predictive model alongside a routing engine to output optimal patrol zones and dynamic risk scores. 

## Setup

To set up the environment on your local machine, run the following commands:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You must also set the `MAPPLS_TOKEN` environment variable for the routing and traffic functionality to work. (Note: Please use your own token or the one sent separately; it is not included in this directory for security reasons.)
```bash
export MAPPLS_TOKEN="your_token_here"
```

## How to Run

To run the Streamlit dashboard:
```bash
streamlit run app.py
```

## Next step: build the demo dashboard

[paste the 4-tab dashboard prompt here]
