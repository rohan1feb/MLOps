from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_extract_variable_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(
        variables=config.model_config.cabin)
    assert sample_input_data["cabin"].iat[0] == 'C22'

    # When
    subject = transformer.transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[0] == 'C'
