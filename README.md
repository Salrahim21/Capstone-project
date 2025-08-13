# Airbnb Cape Town Business Analysis

## Overview
This project analyzes Airbnb listings and reviews in Cape Town to uncover key drivers of revenue, guest satisfaction, and booking behavior.  
We combined structured data modeling with NLP techniques to:
- Build a **price prediction model**
- Extract review topics using **BERTopic**
- Perform **sentiment analysis** for actionable insights

## Business Problem
Airbnb hosts in Cape Town face increasing competition and often lack data-driven tools to:
- Optimize pricing
- Interpret reviews
- Improve guest satisfaction

We addressed these questions:
1. What factors most strongly predict **listing price**, **listing revenue**, and **occupancy**?
2. What themes appear in guest reviews?
3. How do these themes correlate with positive or negative sentiment?
4. How can this help hosts improve their listings and Airbnb refine its search ranking?

## Data
Datasets sourced from **Inside Airbnb**:
- `listings.csv`: Basic listing details
- `listings_detailed.csv`: Extended features including revenue & occupancy estimates
- `reviews_detailed.csv`: Over 600k detailed guest reviews
- `neighbourhoods.csv` and `neighbourhoods.geojson`

**Target Variables**:
- Prediction: `price`, `estimated_revenue_l365d`, `estimated_occupancy_l365d`
- NLP: guest `comments`

## Methodology
1. **Data Cleaning & Preparation**
   - Standardized column names
   - Cleaned prices & availability data
2. **Feature Engineering**
   - Aggregated calendar and reviews into listing-level metrics
   - Created features: `avg_price`, `availability_rate`, `review_count`, `latest_review`
3. **Exploratory Data Analysis**
   - Price distributions
   - Occupancy trends by neighborhood
4. **Modeling**
   - Price & revenue prediction using regression models
   - Topic extraction via BERTopic
   - Sentiment classification
5. **Evaluation**
   - RMSE for regression tasks
   - Topic coherence for NLP

## Key Insights
- Certain property types, amenities, and neighborhoods command significantly higher prices.
- Review themes around cleanliness, location, and communication strongly affect sentiment.
- Listings with higher availability rates tend to have lower prices but higher occupancy.

## Results & Recommendations
To help Airbnb hosts maximize revenue and guest satisfaction, our analysis suggests:
- **Offer entire homes** if possible.
- **Increase booking availability** to capture more demand.
- **Maintain high review scores** by improving guest experience.
- **Optimize pricing** using the model’s feedback to stay competitive.