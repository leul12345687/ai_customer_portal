from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import joblib
import os
import numpy as np
from collections import Counter
from dotenv import load_dotenv

from recommendation_model import recommend_for_user
from smart_search_model import smart_search_and_rank

app = Flask(__name__)
load_dotenv()

# ==============================
# LOAD MODEL FILES
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

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
        {"$group": {"_id": "$assetId", "count": {"$sum": 1}}}
    ]

    results = bookings_collection.aggregate(pipeline)

    pop_map = {}
    for item in results:
        pop_map[str(item["_id"])] = item["count"]

    return pop_map


def build_user_profile(user_id):
    bookings = list(bookings_collection.find({"userId": ObjectId(user_id)}))

    print("===================================")
    print("User:", user_id)
    print("Bookings count:", len(bookings))
    print("===================================")

    if len(bookings) == 0:
        return None

    asset_ids = [b["assetId"] for b in bookings if "assetId" in b]

    if len(asset_ids) == 0:
        return None

    assets = list(properties_collection.find({"_id": {"$in": asset_ids}}))
    if len(assets) == 0:
        return None

    df = pd.DataFrame(assets)
    df = clean_dataframe(df)

    if df.empty or "category" not in df.columns:
        return None

    category_counts = Counter(df["category"])
    location_counts = Counter(df["location"])
    condition_counts = Counter(df["condition"])

    avg_budget = int(df["price_per_day"].mean()) if "price_per_day" in df.columns else 0

    return {
        "preferred_categories": [c for c, _ in category_counts.most_common(3)],
        "preferred_locations": [l for l, _ in location_counts.most_common(3)],
        "preferred_conditions": [cond for cond, _ in condition_counts.most_common(3)],
        "avg_budget": avg_budget,
        "booked_asset_ids": [str(a) for a in asset_ids]
    }


def exclude_booked_assets(df, booked_ids):
    if not booked_ids:
        return df
    return df[~df["_id"].astype(str).isin(booked_ids)]


# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return "AI Service Running Successfully"


# ==============================
# DEBUG ROUTES
# ==============================

@app.route("/debug-user-bookings/<user_id>")
def debug_user_bookings(user_id):
    bookings = list(bookings_collection.find({"userId": ObjectId(user_id)}))
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
    # USER PERSONALIZATION
    # ==========================

    all_properties_df = df.copy()

    user_profile = build_user_profile(user_id)

    if user_profile:
        print("User recommendation profile:", user_profile)

        if user_profile["preferred_categories"]:
            filtered_df = df[df["category"].isin(user_profile["preferred_categories"])]
            if len(filtered_df) >= 5:
                df = filtered_df
            else:
                print("Fallback: not enough personalized category matches")

        df = exclude_booked_assets(df, user_profile["booked_asset_ids"])

        if df.empty:
            print("Personalized filters removed all assets, restoring full property set.")
            df = all_properties_df.copy()
            df = exclude_booked_assets(df, user_profile["booked_asset_ids"])
    else:
        print("No booking history → using default recommendation")

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
# SMART SEARCH ROUTE
# ==============================

@app.route("/smart-search", methods=["POST"])
def smart_search():

    query = request.json or {}

    properties = list(properties_collection.find())
    df = pd.DataFrame(properties)

    if df.empty:
        return jsonify({"error": "No properties found"}), 404

    df = clean_dataframe(df)

    results = smart_search_and_rank(df, query, top_n=5)

    return jsonify(serialize_doc(results.to_dict(orient="records")))


# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 