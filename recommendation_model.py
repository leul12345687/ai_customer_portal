
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_for_user(user_row, df, encoder, scaler, property_vectors, categorical_cols, numerical_cols, top_n=5):
    # Handle missing user preferences with defaults
    u_category = user_row.get('preferred_category', df['category'].mode()[0] if not df.empty else 'Unknown')
    u_location = user_row.get('preferred_location', df['location'].mode()[0] if not df.empty else 'Unknown')
    u_condition = user_row.get('preferred_condition', 'Good')
    u_budget = user_row.get('max_budget_per_day', df['price_per_day'].max() if not df.empty else 10000)

    user_input = {
        'category': u_category,
        'location': u_location,
        'condition': u_condition,
        'price_per_day': u_budget,
        'popularity': df['popularity'].mean()
    }

    import pandas as pd
    user_df = pd.DataFrame([user_input])

    user_cat = encoder.transform(user_df[categorical_cols])
    user_num = scaler.transform(user_df[numerical_cols])

    user_vector = np.hstack([user_cat, user_num])

    similarity_scores = cosine_similarity(user_vector, property_vectors)

    temp_df = df.copy()
    temp_df['similarity'] = similarity_scores[0]

    temp_df = temp_df[temp_df['price_per_day'] <= u_budget]

    return temp_df.sort_values(by='similarity', ascending=False).head(top_n)



