# Recommendation Customer Portal - System Review and Technical README

## 1) Overview
This project is a Flask-based recommendation service for rental assets. It reads data from MongoDB and returns personalized top-N asset recommendations for a user.

The current system is a hybrid recommender:
- Content-based similarity (category, location, condition, price, popularity)
- Behavior-based signal (user booking history)
- Popularity-based signal (paid, non-cancelled bookings)
- Rule-based weighted score fusion

It is not a deep learning model and not an end-to-end neural recommender.

## 2) High-Level Architecture
Core files:
- `app.py`: Flask API, MongoDB integration, feature preparation, model artifact lifecycle
- `recommendation_model.py`: recommendation ranking logic and final score calculation
- `smart_search_model.py`: optional smart-search filtering and ranking utility (currently not exposed as a Flask route)
- `encoder.pkl` and `scaler.pkl`: persisted preprocessing artifacts

Data source (MongoDB database `lmgtech`):
- `assets` collection: item metadata (category, location, condition, price_per_day, ...)
- `bookings` collection: transaction/engagement history used for popularity and user profile
- `users` collection: user existence validation and fallback context

## 3) How the Recommendation Model Works
### Step A: Data ingestion
At request time (`GET /recommend/<user_id>`), all assets are loaded from MongoDB and converted into a pandas DataFrame.

### Step B: Data cleaning
Missing fields are filled with defaults:
- location -> `Unknown`
- category -> `Other`
- condition -> `Good`
- price_per_day -> `0`
- popularity -> `0`

### Step C: Popularity feature
Popularity is computed from bookings using MongoDB aggregation:
- Include bookings where:
  - `status != CANCELLED`
  - `paymentStatus == PAID`
  - `asset` exists
- Group by `asset` and count bookings
- Map booking counts to each asset as `popularity`

### Step D: User profile extraction
From a user's paid, non-cancelled bookings, the service builds profile signals:
- Top preferred categories
- Top preferred locations
- Top preferred conditions
- Average budget (`avg_budget`)
- Previously booked asset IDs
- Most recent booked asset

If user has no booking history, the system falls back to generic preferences from available data.

### Step E: Feature vectorization
The system uses preprocessing artifacts:
- `OneHotEncoder(handle_unknown='ignore')` for categorical features: category, location, condition
- `StandardScaler()` for numerical features: price_per_day, popularity

Final property vector is:
- `[onehot(category, location, condition), scaled(price_per_day, popularity)]`

### Step F: Similarity + weighted ranking
In `recommendation_model.py`:
1. Build a synthetic user vector from profile preferences and budget
2. Compute cosine similarity between user vector and each property vector
3. Add preference bonus weights:
   - category match bonus: `+0.12`
   - location match bonus: `+0.08`
   - condition match bonus: `+0.05`
4. Compute booking similarity:
   - If user has booked assets, compute max cosine similarity to booked asset vectors
5. Compute normalized popularity score
6. Final weighted score:

```
final_score =
    similarity * 0.55
  + booking_similarity * 0.30
  + popularity_score * 0.15
  + same_property_bonus
```

Where:
- `same_property_bonus = 0.4` for exact previously booked asset matches

Then filter by budget (`price_per_day <= user_budget`) and return top N (`top_n=5`).

## 4) Algorithm(s) Used
Primary algorithms and methods:
- One-hot encoding (categorical transformation)
- Standardization (z-score scaling)
- Cosine similarity (vector similarity)
- Weighted linear score fusion (rule-based)
- Aggregation pipeline in MongoDB for popularity counts

Type of recommender:
- Hybrid classical recommender (content + behavior + popularity + heuristics)

What it is not:
- Not collaborative filtering matrix factorization
- Not neural recommendation model
- Not transformer-based model

## 5) Libraries Used
From `requirements.txt` and code usage:
- Flask: API server
- PyMongo: MongoDB client
- pandas: tabular processing
- NumPy: numerical operations
- scikit-learn: `OneHotEncoder`, `StandardScaler`, `cosine_similarity`
- joblib: artifact persistence (`encoder.pkl`, `scaler.pkl`)
- python-dotenv: environment variable loading
- gunicorn: production serving
- flask-cors, uvicorn: listed dependency/runtime options

## 6) Dataset and Data Schema
There is no static CSV dataset in this repository. The dataset is live operational data from MongoDB:
- Assets data from `assets`
- User behavior from `bookings`
- User metadata from `users`

Minimum expected asset fields for best performance:
- `_id`
- `category`
- `location`
- `condition`
- `price_per_day`

Minimum expected booking fields:
- `userId`
- `asset`
- `status`
- `paymentStatus`

## 7) Training Method: Is It "AutoTrain" or Classical Training?
Short answer:
- This is not Hugging Face AutoTrain (or similar AutoML service).
- This is also not supervised model training with labels.

What happens instead:
- The system "fits" preprocessing artifacts (encoder and scaler) on current assets data.
- No target label is learned.
- Ranking is produced by cosine similarity + manually defined weights.
- A background thread retrains (refits artifacts) every 10,800 seconds (3 hours).

So this is best described as:
- Feature artifact fitting + heuristic ranking pipeline (not a predictive supervised learner).

## 8) API Endpoints
Base URL (production):
- https://ai-customer-portal.onrender.com

Endpoints:
- `GET /health` -> service health check
- `GET /recommend/<user_id>` -> top recommendations
- `GET /debug-user-bookings/<user_id>` -> inspect user bookings
- `GET /users-with-bookings` -> booking counts by user
- `GET /debug-assets` -> sample assets
- `GET /evaluate-model` -> offline evaluation metrics (leave-one-out)

Full URL examples:
- https://ai-customer-portal.onrender.com/health
- https://ai-customer-portal.onrender.com/recommend/<user_id>
- https://ai-customer-portal.onrender.com/debug-user-bookings/<user_id>
- https://ai-customer-portal.onrender.com/users-with-bookings
- https://ai-customer-portal.onrender.com/debug-assets
- https://ai-customer-portal.onrender.com/evaluate-model

Note:
- `smart_search_and_rank` exists in `smart_search_model.py` but currently is not exposed through an API route in `app.py`.

### `/evaluate-model` query params
This endpoint evaluates the recommender offline using booking history:
- Protocol: for each evaluated user, hold out their most recent PAID booking and check whether the held-out asset appears in the top-K recommendations.

Query parameters:
- `k` (default `5`): Top-K used for metrics
- `max_users` (default `200`): Max users to evaluate (most active users first)
- `max_bookings` (default `50000`): Max booking rows to load for evaluation

Response metrics:
- `accuracy_at_k` / `hit_rate_at_k`: fraction of users where held-out asset is in top-K
- `precision_at_k`: $hits / (users\_scored * k)$
- `recall_at_k`: same as hit rate (one held-out relevant item per user)
- `mrr_at_k`: mean reciprocal rank
- `ndcg_at_k`: normalized discounted cumulative gain
- `catalog_coverage`: unique recommended assets / total assets

## 9) Model Lifecycle and Retraining
On startup:
1. Load existing `encoder.pkl` and `scaler.pkl` if present
2. If absent or invalid, fit new artifacts from assets data

Background process:
- Retrains artifacts every 3 hours in a daemon thread

Operational implication:
- New categories/locations/conditions become available after next retrain cycle (or service restart if needed).

## 10) Running the Service
### Prerequisites
- Python 3.11 (runtime file uses `python-3.11.9`)
- MongoDB access
- Environment variable `MONGO_URI`

### Install
```bash
pip install -r requirements.txt
```

### Start locally
```bash
python app.py
```

### Production (Procfile)
```bash
gunicorn app:app
```

## 11) Strengths and Limitations
Strengths:
- Fast to run and easy to interpret
- Works without labeled training dataset
- Uses real user behavior and popularity signals
- Robust fallback behavior for missing profile information

Limitations:
- Heuristic weights are hand-tuned, not learned/optimized
- No offline evaluation metrics included (Precision@K, Recall@K, NDCG)
- No explicit exploration/diversity or novelty constraints
- Cold-start for completely new users relies on global defaults
- Smart search module is not integrated into routes

## 12) Suggested Next Improvements
1. Add offline evaluation pipeline with ranking metrics.
2. Move fixed score weights to configuration and tune experimentally.
3. Add route for `smart_search_and_rank` and unify search + recommendation.
4. Add caching for assets/popularity if traffic grows.
5. Add tests for ranking behavior and edge cases.
6. Introduce feature store/versioning for artifact reproducibility.
