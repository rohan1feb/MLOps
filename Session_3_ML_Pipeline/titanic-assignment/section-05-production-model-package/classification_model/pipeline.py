from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder, OneHotEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config
from classification_model.processing import features as pp

# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(imputation_method='missing', variables = config.model_config.categorical_vars, fill_value = 'missing', ignore_format = True)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables = config.model_config.numerical_vars)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(imputation_method='median', variables=config.model_config.numerical_vars)),

    # Extract first letter from cabin
    ('extract_letter', pp.ExtractLetterTransformer(variables=config.model_config.cabin)),

    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(tol=0.05, n_categories=1, variables=config.model_config.categorical_vars, replace_with='Rare', ignore_format = True)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(drop_last=True, variables=config.model_config.categorical_vars, ignore_format = True)),

    # scale using standardization
    ('scaler', StandardScaler()),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C=config.model_config.c, random_state=config.model_config.random_state)),
])