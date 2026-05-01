
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def _as_str_id(value):
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _extract_booking_customer_and_asset(booking_doc):
    """Extract (customer_id, asset_id) from a booking document.

    Supports multiple schema variants:
    - customer vs userId
    - asset as ObjectId or string
    """
    if not isinstance(booking_doc, dict):
        return None, None

    customer_id = (
        booking_doc.get("customer")
        if booking_doc.get("customer") is not None
        else booking_doc.get("userId")
    )
    asset_id = booking_doc.get("asset")

    return _as_str_id(customer_id), _as_str_id(asset_id)


def build_common_asset_model(
    bookings,
    *,
    valid_payment_statuses=("PAID",),
    invalid_statuses=("CANCELLED",),
):
    """Build a simple item-item co-occurrence model from booking history.

    Model idea:
    - Each customer contributes a set of booked assets.
    - Co-occurrence counts how often two assets appear in the same customer's history.
    - Similarity uses cosine on binary customer vectors:
        sim(a, b) = co(a,b) / sqrt(freq(a) * freq(b))
      where freq(x) is number of customers who booked asset x.
    """
    from collections import defaultdict
    from itertools import combinations
    import math

    customers_to_assets = defaultdict(set)

    for booking in bookings or []:
        if not isinstance(booking, dict):
            continue

        status = booking.get("status")
        payment_status = booking.get("paymentStatus")

        if status in invalid_statuses:
            continue
        if valid_payment_statuses is not None and payment_status not in valid_payment_statuses:
            continue

        customer_id, asset_id = _extract_booking_customer_and_asset(booking)
        if not customer_id or not asset_id:
            continue

        customers_to_assets[customer_id].add(asset_id)

    asset_user_freq = defaultdict(int)
    for assets in customers_to_assets.values():
        for asset_id in assets:
            asset_user_freq[asset_id] += 1

    co_counts = defaultdict(int)
    for assets in customers_to_assets.values():
        if len(assets) < 2:
            continue
        for a, b in combinations(sorted(assets), 2):
            co_counts[(a, b)] += 1

    neighbors = defaultdict(list)
    for (a, b), co in co_counts.items():
        neighbors[a].append((b, co))
        neighbors[b].append((a, co))

    return {
        "customers_to_assets": customers_to_assets,
        "asset_user_freq": asset_user_freq,
        "co_counts": co_counts,
        "neighbors": neighbors,
        "similarity": lambda a, b: (
            0.0
            if (a is None or b is None or a == b)
            else (
                (co_counts.get((min(a, b), max(a, b)), 0) / math.sqrt(asset_user_freq[a] * asset_user_freq[b]))
                if asset_user_freq.get(a) and asset_user_freq.get(b)
                else 0.0
            )
        ),
    }


def recommend_by_common_asset_history(
    target_customer_id,
    assets_df,
    common_asset_model,
    *,
    top_n=5,
    popularity_col="popularity",
):
    """Recommend assets based on 'common assets' in booking history.

    Returns a pandas DataFrame subset of assets_df sorted by `final_score`.
    """
    import pandas as pd
    import math
    from collections import defaultdict

    if assets_df is None or len(assets_df) == 0:
        return assets_df

    if not isinstance(common_asset_model, dict) or "customers_to_assets" not in common_asset_model:
        # No model → return popularity-only ranking
        temp_df = assets_df.copy()
        temp_df["final_score"] = temp_df[popularity_col] if popularity_col in temp_df.columns else 0
        return temp_df.sort_values(by="final_score", ascending=False).head(top_n)

    customers_to_assets = common_asset_model["customers_to_assets"]
    asset_user_freq = common_asset_model.get("asset_user_freq", {})
    neighbors = common_asset_model.get("neighbors", {})

    target_customer_id = _as_str_id(target_customer_id)
    user_assets = customers_to_assets.get(target_customer_id, set())

    temp_df = assets_df.copy()
    temp_df["_asset_id"] = temp_df["_id"].astype(str)

    # If the user has no booking history, fall back to popularity.
    if not user_assets:
        temp_df["final_score"] = temp_df[popularity_col] if popularity_col in temp_df.columns else 0
        return temp_df.sort_values(by="final_score", ascending=False).head(top_n)

    # Score candidates by sum of cosine-normalized co-occurrence.
    scores = defaultdict(float)
    for booked_asset in user_assets:
        freq_a = asset_user_freq.get(booked_asset)
        if not freq_a:
            continue

        for other, co in neighbors.get(booked_asset, []):
            if other in user_assets:
                continue
            freq_b = asset_user_freq.get(other)
            if not freq_b:
                continue
            scores[other] += co / math.sqrt(freq_a * freq_b)

    if not scores:
        temp_df["final_score"] = temp_df[popularity_col] if popularity_col in temp_df.columns else 0
        return temp_df.sort_values(by="final_score", ascending=False).head(top_n)

    score_series = pd.Series(scores, name="co_score", dtype=float)
    temp_df = temp_df.merge(score_series, left_on="_asset_id", right_index=True, how="left")
    temp_df["co_score"] = temp_df["co_score"].fillna(0.0)

    if popularity_col in temp_df.columns:
        max_pop = float(temp_df[popularity_col].max()) if len(temp_df) else 0.0
        max_pop = max_pop if max_pop > 0 else 1.0
        temp_df["popularity_norm"] = temp_df[popularity_col] / max_pop
    else:
        temp_df["popularity_norm"] = 0.0

    # Keep scoring simple: booking-history co_score is primary.
    temp_df["final_score"] = temp_df["co_score"] * 0.85 + temp_df["popularity_norm"] * 0.15

    # Exclude already-booked assets
    temp_df = temp_df[~temp_df["_asset_id"].isin({str(x) for x in user_assets})]

    return temp_df.sort_values(by="final_score", ascending=False).head(top_n)

def recommend_for_user(user_context, df, encoder, scaler, property_vectors, categorical_cols, numerical_cols, top_n=5):
    preferred_categories = user_context.get('preferred_categories') if isinstance(user_context, dict) else None
    preferred_locations = user_context.get('preferred_locations') if isinstance(user_context, dict) else None
    preferred_conditions = user_context.get('preferred_conditions') if isinstance(user_context, dict) else None
    booked_asset_ids = user_context.get('booked_asset_ids') if isinstance(user_context, dict) else []

    u_category = None
    u_location = None
    u_condition = None
    u_budget = None

    if preferred_categories:
        u_category = preferred_categories[0]
    elif isinstance(user_context, dict):
        u_category = user_context.get('preferred_category')

    if preferred_locations:
        u_location = preferred_locations[0]
    elif isinstance(user_context, dict):
        u_location = user_context.get('preferred_location')

    if preferred_conditions:
        u_condition = preferred_conditions[0]
    elif isinstance(user_context, dict):
        u_condition = user_context.get('preferred_condition')

    if isinstance(user_context, dict):
        u_budget = user_context.get('avg_budget') or user_context.get('max_budget_per_day')

    if not u_category:
        u_category = df['category'].mode()[0] if not df.empty and 'category' in df.columns else 'Unknown'
    if not u_location:
        u_location = df['location'].mode()[0] if not df.empty and 'location' in df.columns else 'Unknown'
    if not u_condition:
        u_condition = 'Good'
    if not u_budget or u_budget <= 0:
        u_budget = int(df['price_per_day'].max()) if not df.empty and 'price_per_day' in df.columns else 10000

    user_input = {
        'category': u_category,
        'location': u_location,
        'condition': u_condition,
        'price_per_day': u_budget,
        'popularity': df['popularity'].mean() if 'popularity' in df.columns else 0
    }

    import pandas as pd
    user_df = pd.DataFrame([user_input])

    user_cat = encoder.transform(user_df[categorical_cols])
    user_num = scaler.transform(user_df[numerical_cols])
    user_vector = np.hstack([user_cat, user_num])

    similarity_scores = cosine_similarity(user_vector, property_vectors)

    temp_df = df.copy()
    temp_df['similarity'] = similarity_scores[0]

    if preferred_categories:
        temp_df['similarity'] += temp_df['category'].isin(preferred_categories).astype(float) * 0.12
    if preferred_locations:
        temp_df['similarity'] += temp_df['location'].isin(preferred_locations).astype(float) * 0.08
    if preferred_conditions:
        temp_df['similarity'] += temp_df['condition'].isin(preferred_conditions).astype(float) * 0.05

    # Boost based on exact booked asset id and booking-based similarity
    if booked_asset_ids:
        booked_mask = temp_df['_id'].astype(str).isin(booked_asset_ids)
        if booked_mask.any():
            booked_vectors = property_vectors[booked_mask.values]
            booking_similarity = cosine_similarity(property_vectors, booked_vectors).max(axis=1)
        else:
            booking_similarity = np.zeros(len(temp_df))

        temp_df['booking_similarity'] = booking_similarity
        temp_df['same_property_bonus'] = booked_mask.astype(float) * 0.4
    else:
        temp_df['booking_similarity'] = 0
        temp_df['same_property_bonus'] = 0

    if 'popularity' in temp_df.columns:
        max_popularity = temp_df['popularity'].max() or 1
        temp_df['popularity_score'] = temp_df['popularity'] / max_popularity
    else:
        temp_df['popularity_score'] = 0

    temp_df['final_score'] = (
        temp_df['similarity'] * 0.55
        + temp_df['booking_similarity'] * 0.30
        + temp_df['popularity_score'] * 0.15
        + temp_df['same_property_bonus']
    )

    if 'price_per_day' in temp_df.columns and u_budget is not None:
        temp_df = temp_df[temp_df['price_per_day'] <= u_budget]

    return temp_df.sort_values(by='final_score', ascending=False).head(top_n)



