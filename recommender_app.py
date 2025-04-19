# app.py (enhanced with EDA tab, export, similarity exploration, trending, bug-fixed charts)

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    events = pd.read_csv("events.csv")
    props = pd.read_csv("item_properties.csv")
    return events, props

@st.cache_resource
def prepare_model_data():
    events_df, item_props_df = load_data()

    electronics_category_ids = [1338, 1002, 1401, 1661, 1051]
    item_category = item_props_df[item_props_df['property'] == 'categoryid'][['itemid', 'cleaned_value']]
    item_category.columns = ['itemid', 'categoryid']
    electronics_items = item_category[item_category['categoryid'].isin(electronics_category_ids)]['itemid'].unique()

    events_df = events_df[events_df['itemid'].isin(electronics_items)].copy()
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    events_df['user_id'] = user_encoder.fit_transform(events_df['visitorid'])
    events_df['item_id'] = item_encoder.fit_transform(events_df['itemid'])

    user_item_matrix = events_df.groupby(['user_id', 'item_id']).size().unstack(fill_value=0)

    item_user_matrix = user_item_matrix.T
    item_similarity_df = pd.DataFrame(
        cosine_similarity(item_user_matrix),
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )

    return events_df, user_item_matrix, item_similarity_df, user_encoder, item_encoder

def recommend_items_for_user(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []
    user_interactions = user_item_matrix.loc[user_id]
    interacted_items = user_interactions[user_interactions > 0].index.tolist()
    scores = {}
    for item in interacted_items:
        similar_items = item_similarity_df[item].drop(index=interacted_items)
        for sim_item, score in similar_items.items():
            scores[sim_item] = scores.get(sim_item, 0) + score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_items = [item for item, _ in sorted_scores[:top_n]]
    return item_encoder.inverse_transform(recommended_items)

def get_similar_items(item_id, top_n=5):
    if item_id not in item_similarity_df.index:
        return []
    similar_items = item_similarity_df[item_id].sort_values(ascending=False)[1:top_n+1]
    return item_encoder.inverse_transform(similar_items.index)

def get_top_trending_items(n=10):
    return events_df['itemid'].value_counts().head(n)

# Streamlit app UI
st.set_page_config(page_title="Electronics Recommender", layout="centered")
st.title("\U0001F4E6 Electronics Recommender System")

with st.spinner("Preparing model..."):
    events_df, user_item_matrix, item_similarity_df, user_encoder, item_encoder = prepare_model_data()

tab1, tab2, tab3 = st.tabs(["\U0001F4AC Recommend", "\U0001F4CA EDA", "\U0001F50D Explore"])

with tab1:
    st.header("\U0001F464 Personalized Recommendations")
    user_id_input = st.number_input("Enter Encoded User ID:", min_value=0, max_value=user_item_matrix.shape[0]-1, step=1)

    if st.button("Get Recommendations"):
        recs = recommend_items_for_user(user_id_input)
        if len(recs) > 0:
            st.success("Top Recommendations:")
            st.write(pd.DataFrame({'Recommended Item ID': recs}))
            csv = pd.DataFrame({'Recommended Item ID': recs}).to_csv(index=False).encode('utf-8')
            st.download_button("Download as CSV", csv, "recommendations_user_{}.csv".format(user_id_input), "text/csv")
        else:
            st.warning("No recommendations found.")

with tab2:
    st.header("\U0001F4C8 EDA Dashboard")
    st.subheader("Top 10 Most Viewed Items")
    top_items = events_df['itemid'].value_counts().head(10)
    top_items_df = top_items.reset_index()
    top_items_df.columns = ['itemid', 'count']
    st.bar_chart(top_items_df.set_index('itemid'))

    st.subheader("Top 10 Active Users")
    top_users = events_df['visitorid'].value_counts().head(10)
    top_users_df = top_users.reset_index()
    top_users_df.columns = ['visitorid', 'count']
    st.bar_chart(top_users_df.set_index('visitorid'))

    st.subheader("Event Type Distribution")
    event_dist = events_df['event'].value_counts()
    event_dist_df = event_dist.reset_index()
    event_dist_df.columns = ['event', 'count']
    st.bar_chart(event_dist_df.set_index('event'))

with tab3:
    st.header("\U0001F50D Explore Similar Items & Trends")
    item_id_input = st.number_input("Enter Encoded Item ID:", min_value=0, max_value=item_similarity_df.shape[0]-1, step=1)
    if st.button("Show Similar Items"):
        similar = get_similar_items(item_id_input)
        if len(similar) > 0:
            st.success("Items similar to item ID {}:".format(item_id_input))
            st.write(pd.DataFrame({'Similar Item ID': similar}))
        else:
            st.warning("No similar items found.")

    st.subheader("\U0001F525 Top Trending Items")
    trending = get_top_trending_items()
    st.dataframe(trending.reset_index().rename(columns={'index': 'Item ID', 'itemid': 'Interactions'}))
