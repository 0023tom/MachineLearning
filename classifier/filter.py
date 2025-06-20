import re

class TopicFilter:
    def __init__(self, keywords=None):
        # Default to Sibu-related terms if none provided
        self.keywords = keywords or [
            # General / Tourism
            "sibu", "sarawak", "central region", "tourism", "travel guide",
            "sibu central", "sibu heritage", "sibu map", "destination",
            "attractions", "cultural tourism", "ecotourism", "adventure",

            # Cultural & Historical Sites
            "sibu heritage centre", "lau king howe hospital memorial museum",
            "tua pek kong", "goddess of mercy pagoda", "rejang esplanade",
            "cultural village", "sibu oldest temple", "history of sibu",

            # Nature & Outdoors
            "bukit lima nature reserve", "sibu lake garden", "sibu town square",
            "rajang river", "kampung datu", "kampung hilir", "delta park",
            "borneo eco-tourism", "forest trails", "iban longhouse visit",

            # Shopping & Markets
            "sibu central market", "night market", "pasar tamu",
            "handicrafts", "local products", "sarawak layer cake",
            "kek lapis", "fresh produce",

            # Food & Cuisine
            "sibu food", "kampua mee", "kompia", "foochow cuisine",
            "traditional delicacies", "sarawak cuisine", "local snacks",
            "coffee shops", "hawker food",

            # Education / Institutions
            "education tourism", "university college of technology sarawak",
            "ucts", "kolej laila taib", "klt", "methodist pilley institute",
            "sibu's learning institutions", "technical and vocational training",

            # Accommodation / Hospitality
            "hotels in sibu", "budget hotels", "riverside hotels",
            "homestays", "travel facilities",

            # Events / Activities
            "borneo cultural festival", "lantern festival",
            "chinese new year celebration", "local festivals",
            "cultural performances", "tourism calendar",

            # Transportation
            "sibu airport", "express boats", "local transport",
            "bus terminal", "road access", "transportation hub",
            "central location",

            # Legacy Keywords
            "rejang", "rajang", "longhouse", "iban", "foo chow", "central market", "tua pek kong"
        ]

    def is_relevant(self, text):
        pattern = "|".join([re.escape(k) for k in self.keywords])
        return re.search(pattern, text.lower()) is not None