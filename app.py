from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import joblib
import os
import numpy as np
import threading
import time
from collections import Counter
from dotenv import load_dotenv
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recommendation_model import (
    recommend_for_user,
    build_common_asset_model,
    recommend_by_common_asset_history,
)
from smart_search_model import smart_search_and_rank

app = Flask(__name__)
load_dotenv()

# ==============================
# LOAD MODEL FILES
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

encoder = None
scaler = None

# ==============================
# CONNECT TO MONGODB
# ==============================

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "lmgtech"

if not MONGO_URI:
    raise ValueError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

properties_collection = db["assets"]
users_collection = db["users"]
bookings_collection = db["bookings"]

# ==============================
# BOOKING-HISTORY MODEL CACHE
# ==============================

COMMON_ASSET_MODEL_TTL_SECONDS = int(os.getenv("COMMON_ASSET_MODEL_TTL_SECONDS", "900"))
COMMON_ASSET_MODEL_MAX_BOOKINGS = int(os.getenv("COMMON_ASSET_MODEL_MAX_BOOKINGS", "50000"))

_common_asset_model_cache = {
    "built_at": 0.0,
    "model": None,
}


def get_common_asset_model_cached():
    now = time.time()
    model = _common_asset_model_cache.get("model")
    built_at = float(_common_asset_model_cache.get("built_at") or 0.0)

    if model is not None and (now - built_at) < COMMON_ASSET_MODEL_TTL_SECONDS:
        return model

    query = {
        "status": {"$ne": "CANCELLED"},
        "paymentStatus": "PAID",
        "asset": {"$exists": True, "$ne": None},
    }
    projection = {"customer": 1, "userId": 1, "asset": 1, "status": 1, "paymentStatus": 1}
    cursor = bookings_collection.find(query, projection=projection).limit(COMMON_ASSET_MODEL_MAX_BOOKINGS)
    booking_rows = list(cursor)

    model = build_common_asset_model(booking_rows)
    _common_asset_model_cache["model"] = model
    _common_asset_model_cache["built_at"] = now
    return model

# ==============================
# HELPERS
# ==============================

def serialize_doc(doc):
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        return {k: serialize_doc(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc


def clean_dataframe(df):
    df = df.fillna({
        "location": "Unknown",
        "category": "Other",
        "condition": "Good",
        "price_per_day": 0,
        "popularity": 0
    })

    if "price_per_day" not in df.columns:
        df["price_per_day"] = 0

    if "popularity" not in df.columns:
        df["popularity"] = 0

    return df


def compute_popularity():
    pipeline = [
        {
            "$match": {
                "status": {"$ne": "CANCELLED"},
                "paymentStatus": "PAID",
                "asset": {"$exists": True, "$ne": None}
            }
        },
        {"$group": {"_id": "$asset", "count": {"$sum": 1}}}
    ]

    results = bookings_collection.aggregate(pipeline)

    pop_map = {}
    for item in results:
        pop_map[str(item["_id"])] = item["count"]

    return pop_map


def build_user_profile(user_id):
    bookings = list(bookings_collection.find({
        "$or": [
            {"userId": ObjectId(user_id)},
            {"customer": ObjectId(user_id)},
        ],
        "status": {"$ne": "CANCELLED"},
        "paymentStatus": "PAID"
    }))

    print("===================================")
    print("User:", user_id)
    print("Bookings count:", len(bookings))
    print("===================================")

    if len(bookings) == 0:
        return None

    asset_ids = []
    for b in bookings:
        if "asset" in b and b["asset"] is not None:
            if isinstance(b["asset"], ObjectId):
                asset_ids.append(b["asset"])
            else:
                try:
                    asset_ids.append(ObjectId(str(b["asset"])))
                except Exception:
                    continue

    if len(asset_ids) == 0:
        return None

    unique_asset_ids = list(dict.fromkeys(asset_ids))
    assets = list(properties_collection.find({"_id": {"$in": unique_asset_ids}}))
    if len(assets) == 0:
        return None

    df = pd.DataFrame(assets)
    df = clean_dataframe(df)

    if df.empty or "category" not in df.columns:
        return None

    category_counts = Counter(df["category"])
    location_counts = Counter(df["location"])
    condition_counts = Counter(df["condition"])

    total_category = sum(category_counts.values()) or 1
    total_location = sum(location_counts.values()) or 1
    total_condition = sum(condition_counts.values()) or 1

    avg_budget = int(df["price_per_day"].mean()) if "price_per_day" in df.columns else 0

    return {
        "preferred_categories": [c for c, _ in category_counts.most_common(3)],
        "preferred_category_weights": {k: v / total_category for k, v in category_counts.items()},
        "preferred_locations": [l for l, _ in location_counts.most_common(3)],
        "preferred_location_weights": {k: v / total_location for k, v in location_counts.items()},
        "preferred_conditions": [cond for cond, _ in condition_counts.most_common(3)],
        "preferred_condition_weights": {k: v / total_condition for k, v in condition_counts.items()},
        "avg_budget": avg_budget,
        "booked_asset_ids": [str(a) for a in unique_asset_ids],
        "recent_booked_asset_id": str(asset_ids[-1]) if asset_ids else None
    }


def exclude_booked_assets(df, booked_ids):
    if not booked_ids:
        return df
    return df[~df["_id"].astype(str).isin(booked_ids)]


def save_model_artifacts(encoder_obj, scaler_obj):
    joblib.dump(encoder_obj, ENCODER_PATH)
    joblib.dump(scaler_obj, SCALER_PATH)


def train_encoder_scaler():
    properties = list(properties_collection.find())
    df = pd.DataFrame(properties)

    if df.empty:
        raise ValueError("No assets available to train models")

    df = clean_dataframe(df)

    categorical_cols = ['category', 'location', 'condition']
    numerical_cols = ['price_per_day', 'popularity']

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0

    encoder_obj = OneHotEncoder(handle_unknown='ignore', sparse=False)
    scaler_obj = StandardScaler()

    encoder_obj.fit(df[categorical_cols])
    scaler_obj.fit(df[numerical_cols])

    save_model_artifacts(encoder_obj, scaler_obj)
    return encoder_obj, scaler_obj


def load_models():
    global encoder, scaler

    if os.path.exists(ENCODER_PATH) and os.path.exists(SCALER_PATH):
        try:
            encoder = joblib.load(ENCODER_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Loaded encoder and scaler from disk.")
            return
        except Exception as exc:
            print("Failed to load persisted models:", exc)

    encoder, scaler = train_encoder_scaler()


def schedule_model_retraining(interval_seconds=10800):
    def _retrain_loop():
        while True:
            try:
                print("Scheduled retraining starting...")
                train_encoder_scaler()
                load_models()
                print("Scheduled retraining completed.")
            except Exception as exc:
                print("Scheduled retraining failed:", exc)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_retrain_loop, daemon=True, name="ModelRetrainScheduler")
    thread.start()


load_models()
schedule_model_retraining()

# ==============================
# ROUTES
# ==============================

@app.route("/health")
def home():
    return "AI Service Running Successfully"


# ==============================
# DEBUG ROUTES
# ==============================

@app.route("/debug-user-bookings/<user_id>")
def debug_user_bookings(user_id):
    bookings = list(bookings_collection.find({
        "$or": [
            {"userId": ObjectId(user_id)},
            {"customer": ObjectId(user_id)},
        ]
    }))
    return jsonify(serialize_doc(bookings))


@app.route("/users-with-bookings")
def users_with_bookings():

    pipeline = [
        {
            "$group": {
                "_id": "$userId",
                "bookingCount": {"$sum": 1}
            }
        },
        {
            "$sort": {"bookingCount": -1}
        }
    ]

    results = list(bookings_collection.aggregate(pipeline))

    output = []
    for r in results:
        output.append({
            "userId": str(r["_id"]),
            "bookingCount": r["bookingCount"]
        })

    return jsonify(output)


@app.route("/debug-assets")
def debug_assets():
    assets = list(properties_collection.find().limit(5))
    return jsonify(serialize_doc(assets))


# ==============================
# RECOMMENDATION ROUTE
# ==============================

@app.route("/recommend/<user_id>", methods=["GET"])
def recommend(user_id):

    if not ObjectId.is_valid(user_id):
        return jsonify({"error": "Invalid user ID"}), 400

    user = users_collection.find_one({"_id": ObjectId(user_id)})

    if user is None:
        return jsonify({"error": "User not found"}), 404

    properties = list(properties_collection.find())
    df = pd.DataFrame(properties)

    if df.empty:
        return jsonify({"error": "No properties found"}), 404

    df = clean_dataframe(df)

    # ==========================
    # POPULARITY FEATURE
    # ==========================

    pop_map = compute_popularity()
    df["popularity"] = df["_id"].astype(str).map(pop_map).fillna(0)

    # ==========================
    # BOOKING-HISTORY (COMMON ASSET) RECOMMENDATIONS
    # ==========================

    common_asset_model = get_common_asset_model_cached()
    results = recommend_by_common_asset_history(
        user_id,
        df,
        common_asset_model,
        top_n=5,
        popularity_col="popularity",
    )

    # If the common-asset model falls back to popularity only (e.g. user has no history),
    # we keep the existing feature-based recommender as a secondary fallback.
    user_profile = build_user_profile(user_id)
    
    # ==========================
    # FEATURES
    # ==========================

    categorical_cols = ['category', 'location', 'condition']
    numerical_cols = ['price_per_day', 'popularity']

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0

    # ==========================
    # VECTORIZE
    # ==========================

    prop_cat = encoder.transform(df[categorical_cols])
    prop_num = scaler.transform(df[numerical_cols])
    
    property_vectors = np.hstack([prop_cat, prop_num])

    # ==========================
    # MODEL
    # ==========================

    if user_profile is not None and (results is None or len(results) < 3):
        results = recommend_for_user(
            user_profile or user,
            df,
            encoder,
            scaler,
            property_vectors,
            categorical_cols,
            numerical_cols,
            top_n=5
        )

    return jsonify(serialize_doc(results.to_dict(orient="records")))



# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 