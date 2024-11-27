from enum import Enum


class FeatureType(Enum):
    """
    An enumeration to represent different types of features in a multimodal system.
    Attributes:
        TEXT (int): Represents text features.
        AUDIO (int): Represents audio features.
        VISUAL (int): Represents visual features.
        TEXT_VISUAL (int): Represents a combination of text and visual features.
        TEXT_AUDIO (int): Represents a combination of text and audio features.
        AUDIO_VISUAL (int): Represents a combination of audio and visual features.
        TEXT_AUDIO_VISUAL (int): Represents a combination of text, audio, and visual features.
    Methods:
        from_str(s: str) -> 'FeatureType':
            Converts a string representation of a feature type to its corresponding FeatureType enum value.
            Args:
                s (str): The string representation of the feature type.
            Returns:
                FeatureType: The corresponding FeatureType enum value.
            Raises:
                ValueError: If the string does not match any known feature type.
        get_numof_types(feature_type: 'FeatureType') -> int:
            Returns the number of distinct feature types represented by the given FeatureType.
            Args:
                feature_type (FeatureType): The FeatureType enum value.
            Returns:
                int: The number of distinct feature types.
            Raises:
                ValueError: If the feature type is invalid.
    """

    TEXT = 1
    AUDIO = 2
    VISUAL = 3
    TEXT_VISUAL = 4
    TEXT_AUDIO = 5
    AUDIO_VISUAL = 6
    TEXT_AUDIO_VISUAL = 7

    @staticmethod
    def from_str(s: str) -> 'FeatureType':
        """
        Convert a string representation of a feature type to its corresponding FeatureType enum.
        Args:
            s (str): The string representation of the feature type. 
                     Valid values are "text", "audio", "visual", "text_visual", "text_audio", "audio_visual", "text_audio_visual".
        Returns:
            FeatureType: The corresponding FeatureType enum value.
        Raises:
            ValueError: If the input string does not match any valid feature type.
        """

        if s == 'text':
            return FeatureType.TEXT
        elif s == 'audio':
            return FeatureType.AUDIO
        elif s == 'visual':
            return FeatureType.VISUAL
        elif s == 'text_visual':
            return FeatureType.TEXT_VISUAL
        elif s == 'text_audio':
            return FeatureType.TEXT_AUDIO
        elif s == 'audio_visual':
            return FeatureType.AUDIO_VISUAL
        elif s == 'text_audio_visual':
            return FeatureType.TEXT_AUDIO_VISUAL
        else:
            raise ValueError(f'''Invalid FeatureType string: {
                             s}\nFeature type: ["audio", "text", "visual", "text_audio", "text_visual", "audio_visual", "text_audio_visual"]''')

    @staticmethod
    def get_numof_types(feature_type: 'FeatureType') -> int:
        """
        Returns the number of types associated with the given feature type.
        Parameters:
        feature_type (FeatureType): The feature type for which the number of types is to be determined.
        Returns:
        int: The number of types associated with the given feature type.
        Raises:
        ValueError: If the feature_type is not a valid FeatureType.
        """

        if feature_type == FeatureType.TEXT:
            return 1
        elif feature_type == FeatureType.AUDIO:
            return 1
        elif feature_type == FeatureType.VISUAL:
            return 1
        elif feature_type == FeatureType.TEXT_VISUAL:
            return 2
        elif feature_type == FeatureType.TEXT_AUDIO:
            return 2
        elif feature_type == FeatureType.AUDIO_VISUAL:
            return 2
        elif feature_type == FeatureType.TEXT_AUDIO_VISUAL:
            return 3
        else:
            raise ValueError('Invalid FeatureType')
