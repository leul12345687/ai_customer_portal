from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import joblib
import os
import numpy as np
import threading
import time
import math
from collections import defaultdict
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


def compute_popularity_from_bookings(bookings):
    pop_map = {}
    for b in bookings or []:
        if not isinstance(b, dict):
            continue
        asset = b.get("asset")
        if asset is None:
            continue
        asset_id = str(asset)
        pop_map[asset_id] = pop_map.get(asset_id, 0) + 1
    return pop_map


def _parse_int_arg(name, default, *, min_value=None, max_value=None):
    raw = request.args.get(name, None)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except Exception:
            value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _as_str_id(value):
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _extract_booking_customer_and_asset(booking_doc):
    if not isinstance(booking_doc, dict):
        return None, None

    customer_id = (
        booking_doc.get("customer")
        if booking_doc.get("customer") is not None
        else booking_doc.get("userId")
    )
    asset_id = booking_doc.get("asset")
    return _as_str_id(customer_id), _as_str_id(asset_id)


def _booking_sort_key(booking_doc):
    if not isinstance(booking_doc, dict):
        return 0
    # Prefer createdAt if present; otherwise ObjectId timestamp ordering is usually good enough.
    if booking_doc.get("createdAt") is not None:
        return booking_doc.get("createdAt")
    return booking_doc.get("_id")


def build_user_profile_from_asset_ids(asset_ids, assets_df):
    if not asset_ids or assets_df is None or len(assets_df) == 0:
        return None

    subset = assets_df[assets_df["_id"].astype(str).isin({str(x) for x in asset_ids})]
    if subset.empty:
        return None

    category_counts = Counter(subset["category"]) if "category" in subset.columns else Counter()
    location_counts = Counter(subset["location"]) if "location" in subset.columns else Counter()
    condition_counts = Counter(subset["condition"]) if "condition" in subset.columns else Counter()

    total_category = sum(category_counts.values()) or 1
    total_location = sum(location_counts.values()) or 1
    total_condition = sum(condition_counts.values()) or 1

    avg_budget = int(subset["price_per_day"].mean()) if "price_per_day" in subset.columns else 0

    return {
        "preferred_categories": [c for c, _ in category_counts.most_common(3)],
        "preferred_category_weights": {k: v / total_category for k, v in category_counts.items()},
        "preferred_locations": [l for l, _ in location_counts.most_common(3)],
        "preferred_location_weights": {k: v / total_location for k, v in location_counts.items()},
        "preferred_conditions": [cond for cond, _ in condition_counts.most_common(3)],
        "preferred_condition_weights": {k: v / total_condition for k, v in condition_counts.items()},
        "avg_budget": avg_budget,
        "booked_asset_ids": [str(a) for a in dict.fromkeys(asset_ids)],
        "recent_booked_asset_id": str(asset_ids[-1]) if asset_ids else None,
    }


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
# EVALUATION ROUTE
# ==============================

@app.route("/evaluate-model", methods=["GET"])
def evaluate_model():
    """Offline evaluation endpoint for the recommender.

    Protocol: leave-one-out per user (hold out each evaluated user's most recent PAID booking).
    Metrics: HitRate@K ("accuracy"), Precision@K, Recall@K, MRR@K, NDCG@K, coverage.
    """

    started_at = time.time()

    k = _parse_int_arg("k", 5, min_value=1, max_value=50)
    max_users = _parse_int_arg("max_users", 200, min_value=1, max_value=5000)
    max_bookings = _parse_int_arg(
        "max_bookings",
        COMMON_ASSET_MODEL_MAX_BOOKINGS,
        min_value=100,
        max_value=200000,
    )

    query = {
        "status": {"$ne": "CANCELLED"},
        "paymentStatus": "PAID",
        "asset": {"$exists": True, "$ne": None},
    }
    projection = {
        "customer": 1,
        "userId": 1,
        "asset": 1,
        "status": 1,
        "paymentStatus": 1,
        "createdAt": 1,
    }

    booking_rows = list(bookings_collection.find(query, projection=projection).limit(max_bookings))
    if not booking_rows:
        return jsonify({"error": "No bookings available for evaluation."}), 400

    # Group bookings per user.
    per_user = defaultdict(list)
    for b in booking_rows:
        user_id_str, asset_id_str = _extract_booking_customer_and_asset(b)
        if not user_id_str or not asset_id_str:
            continue
        per_user[user_id_str].append(b)

    # Only evaluate users with >=2 bookings.
    candidates = []
    for u, rows in per_user.items():
        rows_sorted = sorted(rows, key=_booking_sort_key)
        if len(rows_sorted) >= 2:
            candidates.append((u, rows_sorted))

    if not candidates:
        return jsonify({"error": "No users with >=2 paid bookings found for evaluation."}), 400

    # Prefer evaluating more active users first (more meaningful signals).
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    candidates = candidates[:max_users]

    holdout_booking_ids = set()
    test_asset_by_user = {}
    train_asset_ids_by_user = {}

    for u, rows in candidates:
        holdout = rows[-1]
        holdout_id = holdout.get("_id")
        _, test_asset_id = _extract_booking_customer_and_asset(holdout)
        if holdout_id is None or not test_asset_id:
            continue

        holdout_booking_ids.add(holdout_id)
        test_asset_by_user[u] = test_asset_id

        train_assets = []
        for tr in rows[:-1]:
            _, a = _extract_booking_customer_and_asset(tr)
            if a:
                train_assets.append(a)
        train_asset_ids_by_user[u] = train_assets

    evaluated_users = [u for u in test_asset_by_user.keys()]
    if not evaluated_users:
        return jsonify({"error": "Could not build evaluation split."}), 400

    train_bookings = [b for b in booking_rows if b.get("_id") not in holdout_booking_ids]

    # Build global booking-history model from training bookings only.
    common_asset_model = build_common_asset_model(train_bookings)

    # Load assets and compute training-only popularity to avoid leakage.
    properties = list(properties_collection.find())
    df = pd.DataFrame(properties)
    if df.empty:
        return jsonify({"error": "No properties found."}), 404
    df = clean_dataframe(df)

    pop_map = compute_popularity_from_bookings(train_bookings)
    df["popularity"] = df["_id"].astype(str).map(pop_map).fillna(0)

    # Vectorize assets once for the feature-based recommender.
    categorical_cols = ["category", "location", "condition"]
    numerical_cols = ["price_per_day", "popularity"]
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0

    prop_cat = encoder.transform(df[categorical_cols])
    prop_num = scaler.transform(df[numerical_cols])
    property_vectors = np.hstack([prop_cat, prop_num])

    # Evaluate.
    hits = 0
    sum_recip_rank = 0.0
    sum_ndcg = 0.0
    recommended_asset_ids = set()
    users_scored = 0

    for u in evaluated_users:
        test_asset_id = test_asset_by_user.get(u)
        train_asset_ids = train_asset_ids_by_user.get(u, [])
        if not test_asset_id or not train_asset_ids:
            continue

        user_profile = build_user_profile_from_asset_ids(train_asset_ids, df)

        results = recommend_by_common_asset_history(
            u,
            df,
            common_asset_model,
            top_n=k,
            popularity_col="popularity",
        )

        if user_profile is not None and (results is None or len(results) < 3):
            results = recommend_for_user(
                user_profile,
                df,
                encoder,
                scaler,
                property_vectors,
                categorical_cols,
                numerical_cols,
                top_n=k,
            )

        if results is None or len(results) == 0 or "_id" not in results.columns:
            continue

        rec_ids = [str(x) for x in results["_id"].tolist()]
        recommended_asset_ids.update(rec_ids)
        users_scored += 1

        rank = None
        for idx, rid in enumerate(rec_ids):
            if rid == str(test_asset_id):
                rank = idx + 1
                break

        if rank is not None and rank <= k:
            hits += 1
            sum_recip_rank += 1.0 / float(rank)
            sum_ndcg += 1.0 / math.log2(float(rank) + 1.0)

    if users_scored == 0:
        return jsonify({"error": "No users could be scored (check assets/bookings schema)."}), 400

    hit_rate = hits / float(users_scored)
    precision_at_k = hits / float(users_scored * k)
    recall_at_k = hit_rate  # single held-out item per user
    mrr_at_k = sum_recip_rank / float(users_scored)
    ndcg_at_k = sum_ndcg / float(users_scored)
    coverage = len(recommended_asset_ids) / float(len(df)) if len(df) else 0.0

    elapsed = time.time() - started_at
    return jsonify(
        {
            "protocol": "leave_one_out_last_booking",
            "k": k,
            "max_users": max_users,
            "max_bookings": max_bookings,
            "users_candidate": len(candidates),
            "users_evaluated": len(evaluated_users),
            "users_scored": users_scored,
            "hits": hits,
            "metrics": {
                "hit_rate_at_k": hit_rate,
                "accuracy_at_k": hit_rate,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "mrr_at_k": mrr_at_k,
                "ndcg_at_k": ndcg_at_k,
                "catalog_coverage": coverage,
            },
            "timing_seconds": elapsed,
        }
    )



# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 