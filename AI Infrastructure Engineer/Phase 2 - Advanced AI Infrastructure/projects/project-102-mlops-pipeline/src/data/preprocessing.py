"""
Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering

Learning Objectives:
- Understand data preprocessing pipelines
- Implement feature engineering techniques
- Handle missing values appropriately
- Scale and normalize features
- Encode categorical variables
- Create reusable preprocessing pipelines

Author: AI Infrastructure Learning
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import joblib

# TODO: Import sklearn preprocessing tools
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Main data preprocessing class

    TODO: Implement comprehensive preprocessing pipeline
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    - Feature engineering
    - Save/load preprocessing artifacts
    """

    def __init__(self, config: Dict):
        """
        Initialize preprocessor

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.preprocessing_pipeline = None
        self.feature_names = None

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        TODO: Handle missing values in dataset

        Steps:
        1. Identify columns with missing values
        2. Apply appropriate strategy per column type
        3. Log imputation details
        4. Return cleaned DataFrame

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'constant', 'drop', 'knn')
            columns: Specific columns to impute (None = all columns with missing values)

        Returns:
            pd.DataFrame: DataFrame with missing values handled

        Strategies:
        - 'mean': Replace with column mean (numerical only)
        - 'median': Replace with column median (numerical only)
        - 'mode': Replace with most frequent value
        - 'constant': Replace with a constant value
        - 'drop': Drop rows with missing values
        - 'knn': Use KNN imputation

        Example:
            df_clean = preprocessor.handle_missing_values(
                df,
                strategy='median',
                columns=['age', 'income']
            )
        """
        # TODO: Implement missing value handling
        """
        from sklearn.impute import SimpleImputer, KNNImputer

        df_copy = df.copy()

        if columns is None:
            columns = df_copy.columns[df_copy.isnull().any()].tolist()

        if not columns:
            self.logger.info("No missing values found")
            return df_copy

        self.logger.info(f"Handling missing values in {len(columns)} columns")

        # Separate numerical and categorical columns
        numerical_cols = df_copy[columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_copy[columns].select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle numerical columns
        if numerical_cols:
            if strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=strategy)
                df_copy[numerical_cols] = imputer.fit_transform(df_copy[numerical_cols])
            elif strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df_copy[numerical_cols] = imputer.fit_transform(df_copy[numerical_cols])

        # Handle categorical columns
        if categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            df_copy[categorical_cols] = imputer.fit_transform(df_copy[categorical_cols])

        # Drop rows strategy
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=columns)

        self.logger.info(f"Missing value handling complete. Rows: {len(df_copy)}")
        return df_copy
        """
        raise NotImplementedError("TODO: Implement missing value handling")

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        encoding_method: str = 'onehot',
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        TODO: Encode categorical variables

        Steps:
        1. Identify categorical columns
        2. Apply encoding strategy
        3. Save encoders for inference time
        4. Return encoded DataFrame and encoder mapping

        Args:
            df: Input DataFrame
            encoding_method: 'onehot', 'label', or 'target'
            columns: Specific columns to encode (None = all object/category columns)

        Returns:
            Tuple[pd.DataFrame, Dict]: (encoded DataFrame, encoder mapping)

        Encoding Methods:
        - 'onehot': One-hot encoding (creates binary columns)
        - 'label': Label encoding (assigns integers)
        - 'target': Target encoding (uses target variable statistics)

        Example:
            df_encoded, encoders = preprocessor.encode_categorical_features(
                df,
                encoding_method='onehot',
                columns=['category', 'region']
            )
        """
        # TODO: Implement categorical encoding
        """
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder

        df_copy = df.copy()
        encoders = {}

        if columns is None:
            columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

        if not columns:
            self.logger.info("No categorical columns found")
            return df_copy, encoders

        self.logger.info(f"Encoding {len(columns)} categorical columns")

        if encoding_method == 'onehot':
            # One-hot encoding
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df_copy[columns])

            # Create column names
            encoded_columns = []
            for i, col in enumerate(columns):
                for category in encoder.categories_[i]:
                    encoded_columns.append(f"{col}_{category}")

            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=encoded_columns,
                index=df_copy.index
            )

            # Drop original columns and add encoded ones
            df_copy = df_copy.drop(columns=columns)
            df_copy = pd.concat([df_copy, encoded_df], axis=1)

            encoders['onehot'] = encoder

        elif encoding_method == 'label':
            # Label encoding
            for col in columns:
                encoder = LabelEncoder()
                df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))
                encoders[col] = encoder

        self.logger.info(f"Categorical encoding complete. Columns: {len(df_copy.columns)}")
        return df_copy, encoders
        """
        raise NotImplementedError("TODO: Implement categorical encoding")

    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        scaling_method: str = 'standard',
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Any]:
        """
        TODO: Scale numerical features

        Steps:
        1. Identify numerical columns
        2. Apply scaling strategy
        3. Save scaler for inference time
        4. Return scaled DataFrame and scaler

        Args:
            df: Input DataFrame
            scaling_method: 'standard', 'minmax', or 'robust'
            columns: Specific columns to scale (None = all numerical columns)

        Returns:
            Tuple[pd.DataFrame, scaler]: (scaled DataFrame, scaler object)

        Scaling Methods:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'minmax': MinMaxScaler (scale to [0, 1])
        - 'robust': RobustScaler (uses median and IQR, robust to outliers)

        Example:
            df_scaled, scaler = preprocessor.scale_numerical_features(
                df,
                scaling_method='standard',
                columns=['age', 'income', 'score']
            )
        """
        # TODO: Implement feature scaling
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            self.logger.info("No numerical columns found")
            return df_copy, None

        self.logger.info(f"Scaling {len(columns)} numerical columns")

        # Select scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        # Fit and transform
        df_copy[columns] = scaler.fit_transform(df_copy[columns])

        self.logger.info(f"Scaling complete using {scaling_method} scaler")
        return df_copy, scaler
        """
        raise NotImplementedError("TODO: Implement feature scaling")

    def engineer_features(self, df: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
        """
        TODO: Create engineered features

        Steps:
        1. Create interaction features (e.g., feature1 * feature2)
        2. Create polynomial features
        3. Create binned features
        4. Create time-based features (if datetime columns exist)
        5. Create domain-specific features

        Args:
            df: Input DataFrame
            feature_config: Feature engineering configuration

        Returns:
            pd.DataFrame: DataFrame with engineered features

        Feature Engineering Types:
        - Interaction: product of two features
        - Polynomial: squared, cubed, etc.
        - Binning: discretize continuous features
        - Date features: day, month, year, day_of_week, etc.
        - Aggregation: rolling statistics, cumulative sums

        Example:
            config = {
                'interactions': [('feature1', 'feature2')],
                'polynomial': {'feature1': 2},
                'bins': {'age': [0, 18, 35, 50, 100]},
                'date_features': ['created_at']
            }
            df_engineered = preprocessor.engineer_features(df, config)
        """
        # TODO: Implement feature engineering
        """
        df_copy = df.copy()

        # Interaction features
        if 'interactions' in feature_config:
            for feat1, feat2 in feature_config['interactions']:
                if feat1 in df_copy.columns and feat2 in df_copy.columns:
                    new_feature = f"{feat1}_x_{feat2}"
                    df_copy[new_feature] = df_copy[feat1] * df_copy[feat2]
                    self.logger.info(f"Created interaction feature: {new_feature}")

        # Polynomial features
        if 'polynomial' in feature_config:
            for feature, degree in feature_config['polynomial'].items():
                if feature in df_copy.columns:
                    for d in range(2, degree + 1):
                        new_feature = f"{feature}_pow{d}"
                        df_copy[new_feature] = df_copy[feature] ** d
                        self.logger.info(f"Created polynomial feature: {new_feature}")

        # Binning
        if 'bins' in feature_config:
            for feature, bins in feature_config['bins'].items():
                if feature in df_copy.columns:
                    new_feature = f"{feature}_binned"
                    df_copy[new_feature] = pd.cut(
                        df_copy[feature],
                        bins=bins,
                        labels=range(len(bins) - 1)
                    )
                    self.logger.info(f"Created binned feature: {new_feature}")

        # Date features
        if 'date_features' in feature_config:
            for feature in feature_config['date_features']:
                if feature in df_copy.columns:
                    df_copy[feature] = pd.to_datetime(df_copy[feature])
                    df_copy[f"{feature}_year"] = df_copy[feature].dt.year
                    df_copy[f"{feature}_month"] = df_copy[feature].dt.month
                    df_copy[f"{feature}_day"] = df_copy[feature].dt.day
                    df_copy[f"{feature}_dayofweek"] = df_copy[feature].dt.dayofweek
                    self.logger.info(f"Created date features for: {feature}")

        self.logger.info(f"Feature engineering complete. Total features: {len(df_copy.columns)}")
        return df_copy
        """
        raise NotImplementedError("TODO: Implement feature engineering")

    def create_pipeline(self, df: pd.DataFrame) -> 'Pipeline':
        """
        TODO: Create sklearn Pipeline for preprocessing

        Steps:
        1. Identify numerical and categorical columns
        2. Create separate transformers for each type
        3. Combine using ColumnTransformer
        4. Return complete pipeline

        This pipeline can be saved and reused for inference!

        Returns:
            Pipeline: sklearn Pipeline object

        Example:
            pipeline = preprocessor.create_pipeline(train_df)
            train_transformed = pipeline.fit_transform(train_df)
            test_transformed = pipeline.transform(test_df)  # Same transformations
        """
        # TODO: Implement sklearn pipeline creation
        """
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer

        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target column if present
        target_col = self.config.get('target_column')
        if target_col:
            numerical_cols = [c for c in numerical_cols if c != target_col]
            categorical_cols = [c for c in categorical_cols if c != target_col]

        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        self.preprocessing_pipeline = preprocessor
        self.logger.info("Preprocessing pipeline created")
        return preprocessor
        """
        raise NotImplementedError("TODO: Implement pipeline creation")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: Fit preprocessing pipeline and transform data

        Args:
            df: Training data

        Returns:
            pd.DataFrame: Transformed data
        """
        # TODO: Implement fit_transform
        raise NotImplementedError("TODO: Implement fit_transform")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: Transform data using fitted pipeline

        Args:
            df: Data to transform (validation/test)

        Returns:
            pd.DataFrame: Transformed data
        """
        # TODO: Implement transform
        raise NotImplementedError("TODO: Implement transform")

    def save_artifacts(self, output_dir: str):
        """
        TODO: Save preprocessing artifacts (pipelines, encoders, scalers)

        Steps:
        1. Create output directory
        2. Save pipeline using joblib
        3. Save feature names
        4. Save preprocessing config

        Args:
            output_dir: Directory to save artifacts

        Example:
            preprocessor.save_artifacts('./artifacts/preprocessing')
        """
        # TODO: Implement artifact saving
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save pipeline
        if self.preprocessing_pipeline:
            pipeline_path = os.path.join(output_dir, 'preprocessing_pipeline.pkl')
            joblib.dump(self.preprocessing_pipeline, pipeline_path)
            self.logger.info(f"Pipeline saved to {pipeline_path}")

        # Save feature names
        if self.feature_names:
            features_path = os.path.join(output_dir, 'feature_names.json')
            import json
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f)

        # Save config
        config_path = os.path.join(output_dir, 'preprocessing_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

        self.logger.info(f"Preprocessing artifacts saved to {output_dir}")
        """
        raise NotImplementedError("TODO: Implement artifact saving")

    @staticmethod
    def load_artifacts(artifacts_dir: str) -> 'DataPreprocessor':
        """
        TODO: Load preprocessing artifacts

        Args:
            artifacts_dir: Directory containing artifacts

        Returns:
            DataPreprocessor: Preprocessor with loaded artifacts
        """
        # TODO: Implement artifact loading
        raise NotImplementedError("TODO: Implement artifact loading")


# TODO: Implement advanced feature engineering techniques
# TODO: Add support for text feature extraction (TF-IDF, embeddings)
# TODO: Implement automatic feature selection
# TODO: Add support for time series features
# TODO: Implement feature importance tracking
# TODO: Add data leakage detection
