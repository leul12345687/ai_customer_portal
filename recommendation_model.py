
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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



