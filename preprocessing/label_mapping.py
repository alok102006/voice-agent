def map_category(text: str):
    """
    Maps cleaned legal text to one of the predefined legal categories.
    Used ONLY for Kaggle dataset filtering.
    """

    if not isinstance(text, str):
        return None

    if any(word in text for word in [
        'employment', 'employee', 'employer', 'salary', 'wages', 'workplace'
    ]):
        return 'EMPLOYMENT'

    elif any(word in text for word in [
        'internet', 'online', 'email', 'computer', 'fraud', 'cyber'
    ]):
        return 'CYBER'

    elif any(word in text for word in [
        'marriage', 'divorce', 'husband', 'wife', 'family', 'custody'
    ]):
        return 'FAMILY'

    elif any(word in text for word in [
        'property', 'land', 'tenant', 'lease', 'rent', 'ownership'
    ]):
        return 'PROPERTY'

    elif any(word in text for word in [
        'assault', 'threat', 'crime', 'police', 'violence', 'robbery'
    ]):
        return 'CRIMINAL'

    return None
