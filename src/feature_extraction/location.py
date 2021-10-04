import pandas as pd
import pycountry
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.text_utils import clean_text

numeric_to_name = {c.numeric.lower(): c.name.lower() for c in list(pycountry.countries)}
alpha_2_to_name = {c.alpha_2.lower(): c.name.lower() for c in list(pycountry.countries)}
alpha_3_to_name = {c.alpha_3.lower(): c.name.lower() for c in list(pycountry.countries)}

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
us_state_abbrev = {v: k.lower() for k, v in us_state_abbrev.items()}


def add_abrivation_states(str_list):
    l = []
    for s in str_list:
        if s in numeric_to_name:
            l.append(numeric_to_name[s])

        elif s in alpha_2_to_name:
            l.append(alpha_2_to_name[s])

        elif s in alpha_3_to_name:
            l.append(alpha_3_to_name[s])

        elif s in us_state_abbrev:
            l.append(us_state_abbrev[s])
    return str_list + l


def preprocess_location(location):
    location = location.apply(str)
    location = location.apply(clean_text)
    location = location.str.split(' ')
    location = location.apply(add_abrivation_states)
    location = location.apply(lambda ls: ' '.join(ls))
    return location


def process_location(location):
    features = pd.DataFrame(index=location.index)
    location = preprocess_location(location)

    vect = TfidfVectorizer()
    location_vect = vect.fit_transform(location)
    location_vect_df = pd.DataFrame(
        data=location_vect.toarray(),
        columns=list(vect.vocabulary_.keys()),
        index=location.index
    )
    features = features.join(location_vect_df)
    features.columns = ['location_' + c for c in features.columns]

    return features
