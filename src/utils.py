import pycountry

CUSTOM_COUNTRY_MAPPINGS = {
    "England": "gb-eng",
    "Wales": "gb-wls",
    "Scotland": "gb-sct",
    "Northern Ireland": "gb-nir",
    "USA": "us",
    "IR Iran": "ir",
    "Korea Republic": "kr",
    "Korea DPR": "kp",
    "China PR": "cn",
    "Cabo Verde": "cv",
    "Congo DR": "cd",
    "Czechia": "cz",
    "Vietnam": "vn",
    "Brunei Darussalam": "bn",
    "Macau": "mo",
    "Hong Kong": "hk",
    "Palestine": "ps",
    "Syria": "sy",
    "Russia": "ru",
    "Turkiye": "tr",
    "Venezuela": "ve",
    "Bolivia": "bo",
}

def get_flag_url(country_name):
    # Check custom mappings first
    if country_name in CUSTOM_COUNTRY_MAPPINGS:
        code = CUSTOM_COUNTRY_MAPPINGS[country_name]
        return f"https://flagcdn.com/w160/{code.lower()}.png"

    # Try normal pycountry lookup
    try:
        # Try exact match
        country = pycountry.countries.get(name=country_name)
        if not country:
            # Try fuzzy search
            country = pycountry.countries.search_fuzzy(country_name)[0]
        
        return f"https://flagcdn.com/w160/{country.alpha_2.lower()}.png"
    except Exception:
        # Fallback empty flag or a generic globe
        return "https://upload.wikimedia.org/wikipedia/commons/e/ef/International_Flag_of_Planet_Earth.svg"
