
import pandas as pd

def apply_search_filters(df, query):
    filtered = df.copy()

    if query.get('category'):
        filtered = filtered[filtered['category'] == query['category']]

    if query.get('location'):
        filtered = filtered[filtered['location'] == query['location']]

    if query.get('max_price'):
        filtered = filtered[filtered['price_per_day'] <= query['max_price']]

    return filtered

def compute_ranking_score(row, query, max_popularity):
    score = 0

    # Category match
    if row['category'] == query.get('category'):
        score += 0.35

    # Condition match
    if row['condition'] == query.get('preferred_condition'):
        score += 0.20

    # Price score (normalized)
    max_price = query.get('max_price')
    if max_price:
        price_score = 1 - (row['price_per_day'] / max_price)
        score += 0.30 * max(price_score, 0)

    # Popularity score (scaled)
    if max_popularity > 0:
        popularity_score = row['popularity'] / max_popularity
        score += 0.15 * popularity_score

    return score

def smart_search_and_rank(df_properties, query, top_n=5):
    filtered_df = apply_search_filters(df_properties, query)

    if filtered_df.empty:
        return filtered_df

    # Calculate max_popularity from the original, unfiltered dataframe for accurate scaling
    max_popularity = df_properties['popularity'].max()

    filtered_df['rank_score'] = filtered_df.apply(
        lambda row: compute_ranking_score(row, query, max_popularity), axis=1
    )

    return filtered_df.sort_values(by='rank_score', ascending=False).head(top_n)
