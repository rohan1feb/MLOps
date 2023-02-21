from typing import List, Optional, Tuple

import re
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
    
def get_title(passenger) -> str:
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
def pre_process_df(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.replace("?", np.nan)
    df_copy["cabin"] = df_copy["cabin"].apply(get_first_cabin)
    df_copy["title"] = df_copy["name"].apply(get_title)
    df_copy["fare"] = df_copy["fare"].astype("float")
    df_copy["age"] = df_copy["age"].astype("float")
    df_copy.drop(labels=["name", "ticket", "boat", "body", "home.dest"], axis=1, inplace=True)
    return df_copy


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.categorical_vars_with_na_frequent
        + config.model_config.categorical_vars_with_na_missing
        + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    relevant_data = input_data[config.model_config.features].copy()
    relevant_data = pre_process_df(relevant_data)
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicInputSchema(BaseModel):
    pclass: Optional[int]
    survived: Optional[int]
    sex: Optional[str]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    title: Optional[str]

class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicInputSchema]
