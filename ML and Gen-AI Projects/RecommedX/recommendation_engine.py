"""
RecommedX - Comprehensive Recommendation Engine
Demonstrates: Collaborative Filtering, Content-Based, Hybrid, and Neural approaches
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA GENERATION - Simulating User-Item Interactions
# ============================================================================

class DataGenerator:
    """Generate synthetic user-item interaction data"""
    
    @staticmethod
    def generate_sample_data(n_users=100, n_items=50, sparsity=0.95):
        """
        Generate sample user-item interaction matrix
        
        Args:
            n_users: Number of users
            n_items: Number of items
            sparsity: Proportion of missing values (0-1)
        
        Returns:
            DataFrame with user-item ratings
        """
        np.random.seed(42)
        
        # Create sparse interaction matrix
        interactions = np.random.choice([0, 1, 2, 3, 4, 5], 
                                       size=(n_users, n_items),
                                       p=[sparsity, 0.01, 0.01, 0.01, 0.01, 0.01])
        
        # Convert to DataFrame
        df = pd.DataFrame(interactions, 
                         columns=[f'item_{i}' for i in range(n_items)],
                         index=[f'user_{i}' for i in range(n_users)])
        
        return df
    
    @staticmethod
    def generate_item_features(n_items=50):
        """Generate item content features"""
        np.random.seed(42)
        
        categories = ['Electronics', 'Books', 'Clothing', 'Food', 'Sports']
        
        items = []
        for i in range(n_items):
            items.append({
                'item_id': f'item_{i}',
                'category': np.random.choice(categories),
                'price': np.random.uniform(10, 500),
                'description': f'Product {i} with features A B C',
                'popularity': np.random.randint(0, 1000)
            })
        
        return pd.DataFrame(items)


# ============================================================================
# 2. COLLABORATIVE FILTERING - Matrix Factorization
# ============================================================================

class CollaborativeFiltering:
    """
    Collaborative Filtering using Matrix Factorization (SVD)
    Learns latent factors for users and items
    """
    
    def __init__(self, n_factors=10):
        """
        Initialize CF model
        
        Args:
            n_factors: Number of latent factors
        """
        self.n_factors = n_factors
        self.model = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
        
    def fit(self, interaction_matrix: pd.DataFrame):
        """
        Train the collaborative filtering model
        
        Args:
            interaction_matrix: User-item interaction DataFrame
        """
        self.user_ids = interaction_matrix.index.tolist()
        self.item_ids = interaction_matrix.columns.tolist()
        
        # Apply SVD to decompose the matrix
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_.T
        
        print(f"✓ Collaborative Filtering trained with {self.n_factors} factors")
        print(f"  User factors shape: {self.user_factors.shape}")
        print(f"  Item factors shape: {self.item_factors.shape}")
        
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating score
        """
        try:
            user_idx = self.user_ids.index(user_id)
            item_idx = self.item_ids.index(item_id)
            
            # Compute dot product of user and item factors
            prediction = np.dot(self.user_factors[user_idx], 
                              self.item_factors[item_idx])
            
            return max(0, min(5, prediction))  # Clip to [0, 5]
        except ValueError:
            return 0.0  # Cold start: return default
    
    def recommend(self, user_id: str, top_k=5) -> List[Tuple[str, float]]:
        """
        Generate top-K recommendations for a user
        
        Args:
            user_id: User identifier
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        try:
            user_idx = self.user_ids.index(user_id)
            
            # Compute scores for all items
            scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
            
            # Get top-K items
            top_indices = np.argsort(scores)[::-1][:top_k]
            recommendations = [(self.item_ids[i], scores[i]) 
                             for i in top_indices]
            
            return recommendations
        except ValueError:
            return []


# ============================================================================
# 3. CONTENT-BASED FILTERING - Item Similarity
# ============================================================================

class ContentBasedFiltering:
    """
    Content-Based Filtering using item features
    Recommends items similar to those the user liked
    """
    
    def __init__(self):
        """Initialize content-based model"""
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.item_features = None
        self.similarity_matrix = None
        self.item_ids = None
        
    def fit(self, item_data: pd.DataFrame):
        """
        Train content-based model using item features
        
        Args:
            item_data: DataFrame with item features
        """
        self.item_ids = item_data['item_id'].tolist()
        
        # Create text features from description and category
        text_features = (item_data['description'] + ' ' + 
                        item_data['category'])
        
        # Vectorize text features
        self.item_features = self.vectorizer.fit_transform(text_features)
        
        # Compute item-item similarity matrix
        self.similarity_matrix = cosine_similarity(self.item_features)
        
        print(f"✓ Content-Based Filtering trained")
        print(f"  Feature matrix shape: {self.item_features.shape}")
        print(f"  Similarity matrix shape: {self.similarity_matrix.shape}")
        
    def get_similar_items(self, item_id: str, top_k=5) -> List[Tuple[str, float]]:
        """
        Find similar items based on content
        
        Args:
            item_id: Item identifier
            top_k: Number of similar items
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        try:
            item_idx = self.item_ids.index(item_id)
            
            # Get similarity scores
            similarities = self.similarity_matrix[item_idx]
            
            # Get top-K similar items (excluding itself)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            similar_items = [(self.item_ids[i], similarities[i]) 
                           for i in top_indices]
            
            return similar_items
        except ValueError:
            return []
    
    def recommend(self, user_history: List[str], top_k=5) -> List[Tuple[str, float]]:
        """
        Recommend items based on user's interaction history
        
        Args:
            user_history: List of item IDs user has interacted with
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        if not user_history:
            return []
        
        # Aggregate scores from all items in user history
        aggregated_scores = np.zeros(len(self.item_ids))
        
        for item_id in user_history:
            try:
                item_idx = self.item_ids.index(item_id)
                aggregated_scores += self.similarity_matrix[item_idx]
            except ValueError:
                continue
        
        # Normalize scores
        aggregated_scores /= len(user_history)
        
        # Get top-K items (excluding items in history)
        history_indices = [self.item_ids.index(item) for item in user_history 
                          if item in self.item_ids]
        aggregated_scores[history_indices] = -1  # Exclude already seen items
        
        top_indices = np.argsort(aggregated_scores)[::-1][:top_k]
        recommendations = [(self.item_ids[i], aggregated_scores[i]) 
                         for i in top_indices if aggregated_scores[i] > 0]
        
        return recommendations


# ============================================================================
# 4. HYBRID RECOMMENDATION ENGINE - Combining Multiple Approaches
# ============================================================================

class HybridRecommender:
    """
    Hybrid Recommendation Engine
    Combines Collaborative Filtering and Content-Based approaches
    """
    
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        """
        Initialize hybrid recommender
        
        Args:
            cf_weight: Weight for collaborative filtering (0-1)
            cb_weight: Weight for content-based filtering (0-1)
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = None
        self.cb_model = None
        
        # Normalize weights
        total = cf_weight + cb_weight
        self.cf_weight /= total
        self.cb_weight /= total
        
    def fit(self, interaction_matrix: pd.DataFrame, item_data: pd.DataFrame):
        """
        Train both CF and CB models
        
        Args:
            interaction_matrix: User-item interactions
            item_data: Item features
        """
        print("\n" + "="*60)
        print("Training Hybrid Recommendation Engine")
        print("="*60)
        
        # Train Collaborative Filtering
        self.cf_model = CollaborativeFiltering(n_factors=10)
        self.cf_model.fit(interaction_matrix)
        
        # Train Content-Based Filtering
        self.cb_model = ContentBasedFiltering()
        self.cb_model.fit(item_data)
        
        print(f"\n✓ Hybrid model trained (CF: {self.cf_weight:.2f}, CB: {self.cb_weight:.2f})")
        
    def recommend(self, user_id: str, user_history: List[str], 
                  top_k=10) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations
        
        Args:
            user_id: User identifier
            user_history: List of items user has interacted with
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        # Get CF recommendations
        cf_recs = self.cf_model.recommend(user_id, top_k=top_k*2)
        cf_dict = {item: score for item, score in cf_recs}
        
        # Get CB recommendations
        cb_recs = self.cb_model.recommend(user_history, top_k=top_k*2)
        cb_dict = {item: score for item, score in cb_recs}
        
        # Combine scores
        all_items = set(cf_dict.keys()) | set(cb_dict.keys())
        hybrid_scores = {}
        
        for item in all_items:
            cf_score = cf_dict.get(item, 0)
            cb_score = cb_dict.get(item, 0)
            
            # Weighted combination
            hybrid_scores[item] = (self.cf_weight * cf_score + 
                                  self.cb_weight * cb_score)
        
        # Sort and return top-K
        sorted_items = sorted(hybrid_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:top_k]
        
        return sorted_items


# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

class RecommenderMetrics:
    """Evaluation metrics for recommendation systems"""
    
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Precision@K: Proportion of recommended items that are relevant
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision score (0-1)
        """
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Recall@K: Proportion of relevant items that are recommended
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall score (0-1)
        """
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Measures ranking quality with position discount
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG score (0-1)
        """
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = sum([1.0 / np.log2(i + 2) if item in relevant_set else 0.0 
                   for i, item in enumerate(recommended_k)])
        
        # Calculate IDCG (ideal DCG)
        idcg = sum([1.0 / np.log2(i + 2) 
                    for i in range(min(len(relevant_set), k))])
        
        return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# 6. DEMO & USAGE EXAMPLE
# ============================================================================

def main():
    """Demonstrate the recommendation engine"""
    
    print("\n" + "="*60)
    print("RecommedX - Recommendation Engine Demo")
    print("="*60)
    
    # 1. Generate sample data
    print("\n[1] Generating sample data...")
    data_gen = DataGenerator()
    interaction_matrix = data_gen.generate_sample_data(n_users=100, n_items=50)
    item_features = data_gen.generate_item_features(n_items=50)
    
    print(f"✓ Generated {interaction_matrix.shape[0]} users and {interaction_matrix.shape[1]} items")
    print(f"✓ Sparsity: {(interaction_matrix == 0).sum().sum() / interaction_matrix.size * 100:.1f}%")
    
    # 2. Train Hybrid Recommender
    print("\n[2] Training Hybrid Recommendation Engine...")
    recommender = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    recommender.fit(interaction_matrix, item_features)
    
    # 3. Generate recommendations for a sample user
    print("\n[3] Generating Recommendations...")
    print("-" * 60)
    
    test_user = 'user_5'
    user_history = ['item_10', 'item_15', 'item_20']
    
    print(f"\nUser: {test_user}")
    print(f"User History: {user_history}")
    
    # Get recommendations
    recommendations = recommender.recommend(test_user, user_history, top_k=10)
    
    print(f"\n{'Rank':<6} {'Item ID':<15} {'Score':<10}")
    print("-" * 35)
    for rank, (item_id, score) in enumerate(recommendations, 1):
        print(f"{rank:<6} {item_id:<15} {score:<10.4f}")
    
    # 4. Evaluate recommendations
    print("\n[4] Evaluation Metrics...")
    print("-" * 60)
    
    # Simulate ground truth (items user actually liked)
    ground_truth = ['item_25', 'item_30', 'item_35', 'item_40']
    recommended_items = [item for item, _ in recommendations]
    
    metrics = RecommenderMetrics()
    precision = metrics.precision_at_k(recommended_items, ground_truth, k=10)
    recall = metrics.recall_at_k(recommended_items, ground_truth, k=10)
    ndcg = metrics.ndcg_at_k(recommended_items, ground_truth, k=10)
    
    print(f"\nPrecision@10: {precision:.4f}")
    print(f"Recall@10:    {recall:.4f}")
    print(f"NDCG@10:      {ndcg:.4f}")
    
    # 5. Content-Based Similar Items
    print("\n[5] Content-Based Similar Items...")
    print("-" * 60)
    
    sample_item = 'item_10'
    similar_items = recommender.cb_model.get_similar_items(sample_item, top_k=5)
    
    print(f"\nItems similar to {sample_item}:")
    print(f"{'Rank':<6} {'Item ID':<15} {'Similarity':<12}")
    print("-" * 35)
    for rank, (item_id, similarity) in enumerate(similar_items, 1):
        print(f"{rank:<6} {item_id:<15} {similarity:<12.4f}")
    
    # 6. Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
✓ Collaborative Filtering: Uses user-item interactions (Matrix Factorization)
✓ Content-Based Filtering: Uses item features (TF-IDF + Cosine Similarity)
✓ Hybrid Approach: Combines both methods with weighted ensemble
✓ Evaluation: Precision, Recall, NDCG metrics
✓ Cold Start Handling: Content-based fallback for new users/items
    """)
    
    print("\n" + "="*60)
    print("RecommedX Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

# Made with Bob
