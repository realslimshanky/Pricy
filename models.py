from pydantic import BaseModel

class PricePredictionInput(BaseModel):
    neighbourhood: str
    neighbourhood_cleansed: str
    neighbourhood_group_cleansed: str
    property_type: str
    room_type: str

    host_response_rate: int
    host_acceptance_rate: int
    host_is_superhost: int
    host_has_profile_pic: int
    host_identity_verified: int
    latitude: float
    longitude: float
    accommodates: int
    bathrooms: float
    bedrooms: int
    beds: int
    minimum_nights: int
    maximum_nights: int
    is_licensed: int

    amenities_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "neighbourhood": "Mitte",
                "neighbourhood_cleansed": "Mitte",
                "neighbourhood_group_cleansed": "Mitte",
                "property_type": "Apartment",
                "room_type": "Entire home/apt",
                "host_response_rate": 95,
                "host_acceptance_rate": 98,
                "host_is_superhost": 1,
                "host_has_profile_pic": 1,
                "host_identity_verified": 1,
                "latitude": 52.5200,
                "longitude": 13.4050,
                "accommodates": 4,
                "bathrooms": 1.0,
                "bedrooms": 1,
                "beds": 2,
                "minimum_nights": 2,
                "maximum_nights": 30,
                "is_licensed": 1,
                "amenities_text": "wifi kitchen heating washer"
            }
        }


class PricePredictionOutput(BaseModel):
    predicted_price: float
