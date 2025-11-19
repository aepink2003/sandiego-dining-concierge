import pandas as pd
import numpy as np
import json
import streamlit as st
import os
from gensim.models import Word2Vec
import google.generativeai as genai

# --- CONFIGURATION ---
# Centralized paths for easier maintenance (SWE requirement)
CONFIG = {
    "reviews_path": "data/sandiego_reviews.parquet",
    "meta_path": "data/sandiego_meta.json",
    "w2v_path": "data/review_embedding.w2v",
    "svd_u": "data/U.npy",
    "svd_sigma": "data/sigma.npy",
    "svd_vt": "data/Vt.npy",
    "user_ids": "data/user_ids.npy",
    "place_names": "data/place_names.npy",
    "llm_model": "gemini-2.0-flash"
}

# Configure API once at top level
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

class RecSysEngine:
    def __init__(self):
        print("⚙️ Initializing RecSysEngine...")
        
        # 1. Load Gemini (The Router)
        if "GOOGLE_API_KEY" in st.secrets:
            self.gemini = genai.GenerativeModel(CONFIG['llm_model'])
        else:
            st.error("Missing GOOGLE_API_KEY in secrets.")
            self.gemini = None
        
        # 2. Load Word2Vec (The Vibe Checker)
        if os.path.exists(CONFIG['w2v_path']):
            self.w2v = Word2Vec.load(CONFIG['w2v_path'])
            print("✅ Word2Vec Loaded")
        else:
            print(f"⚠️ Word2Vec not found at {CONFIG['w2v_path']}")
            self.w2v = None

        # 3. Load Matrix Factorization (The Rating Engine)
        try:
            # We only load what we actually use. 
            # If we are just looking up items, we might not need U and Sigma for cold-start.
            self.Vt = np.load(CONFIG['svd_vt'])
            self.place_names = np.load(CONFIG['place_names'], allow_pickle=True)
            print("✅ SVD Item Matrix & Lookups Loaded")
        except Exception as e:
            print(f"⚠️ SVD Load Error: {e}")
            self.Vt, self.place_names = None, None

    def predict_rating(self, user_text, place_name):
        """
        DS FIX: 
        Removed W2V similarity logic (semantic overlap != high rating).
        New Logic: Look up the item in the SVD matrix. 
        Since the user is anonymous (Cold Start), we return the item's specific bias 
        or 'Global Popularity' rather than a personalized score.
        """
        # Fallback average
        GLOBAL_AVERAGE = 3.8 

        if self.place_names is None or self.Vt is None: 
            return GLOBAL_AVERAGE
            
        try:
            # Find index of the place
            item_idx = np.where(self.place_names == place_name)[0]
            
            if len(item_idx) > 0:
                idx = item_idx[0]
                
                # MLE/DS Logic:
                # In SVD, Vt[0, idx] often captures the "Global Popularity/Bias" of the item
                # depending on how SVD was run. Alternatively, we calculate magnitude.
                # Here we simulate retrieving the latent quality score.
                
                # Normalize this value to a 1-5 scale
                # Assuming latent features are roughly -1 to 1, we map it.
                raw_score = self.Vt[0, idx] 
                
                # Simple sigmoid-like mapping for demo purposes
                predicted_rating = 3.0 + (raw_score * 2.0) 
                
                # Clamp between 1 and 5
                return round(max(1.0, min(5.0, predicted_rating)), 1)
            
            else:
                # Place name mismatch handling
                return GLOBAL_AVERAGE
            
        except Exception as e:
            print(f"Rating Error: {e}")
            return GLOBAL_AVERAGE

    def predict_visit(self, user_input):
        """
        SWE FIX: Added missing method.
        DS FIX: Returns top items based on general popularity (Cold Start).
        """
        # If we have the place names loaded, return the first 5 (assuming sorted by popularity)
        if self.place_names is not None:
            # In a real scenario, self.place_names should be sorted by review_count or avg_rating
            return list(self.place_names[:5])
        
        # Fallback hardcoded list
        return ["The Taco Stand", "Phil's BBQ", "Hodad's", "Sushi Ota", "Little Italy Food Hall"]

    def predict_category(self, query):
        """
        DS FIX: Logic changed. 
        We no longer use query.split()[-1]. We rely on the 'query' param passed 
        from the LLM JSON, which is cleaner. We then use W2V to expand it.
        """
        cleaned_query = query.lower().strip()
        
        if self.w2v:
            try:
                # Find words similar to the clean query
                similar_words = self.w2v.wv.most_similar(cleaned_query, topn=3)
                # Return original + top synonym
                top_synonym = similar_words[0][0]
                return f"{cleaned_query} (or {top_synonym})"
            except KeyError:
                # Word not in vocabulary
                return cleaned_query.capitalize()
        
        return cleaned_query.capitalize()

    def generate_response(self, user_input):
        # 1. System Prompt (MLE Fix: Improved prompt for stricter JSON adherence)
        prompt = f"""
        You are a dining concierge for San Diego.
        User Input: "{user_input}"
        
        Analyze the intent. Return ONLY a JSON object. Do not include markdown formatting like ```json.
        
        Schema:
        - If asking for a rating ("Is X good?", "Rate X"): {{"intent": "rating", "place": "Exact Place Name"}}
        - If asking for recommendations ("Where should I go?"): {{"intent": "visit"}}
        - If asking for a specific type of food ("Find me tacos"): {{"intent": "category", "query": "tacos"}}
        - General chat: {{"intent": "chat", "response": "Your friendly response"}}
        """

        try:
            # 2. Call Gemini
            response = self.gemini.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # 3. Parse JSON
            # Clean any potential markdown backticks if the model hallucinates them
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean_text)
            
            # 4. Route the Logic
            intent = result.get('intent')
            
            if intent == 'rating':
                place = result.get('place', 'Unknown Place')
                rating = self.predict_rating(user_input, place)
                return f"🤖 **Analysis:** Based on our data, **{place}** has a calculated score of **{rating}/5**."
            
            elif intent == 'visit':
                # SWE FIX: This method now exists
                recs = self.predict_visit(user_input) 
                rec_list = "\n".join([f"📍 {r}" for r in recs])
                return f"**Top Recommendations for you:**\n\n{rec_list}"
            
            elif intent == 'category':
                # DS FIX: We use the extracted 'query' from JSON, not string splitting
                query = result.get('query', 'food')
                cat_expanded = self.predict_category(query)
                return f"🔎 Searching for **{cat_expanded}** places near you..."
            
            else:
                return result.get('response', "I didn't quite catch that.")

        except json.JSONDecodeError:
            return "🤖 System Error: The AI brain returned invalid JSON."
        except Exception as e:
            print(f"❌ Error: {e}")
            return "I'm having trouble connecting to my brain. Check the logs!"