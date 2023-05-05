import requests, os, json
import geopy
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import plotly_express as px
import tqdm
from tqdm.notebook import tqdm_notebook


def save_image(image_id, sequence_name):
    app_access_token = 'MLY|5472920012744597|0a1926edd41ea7b4c4d7d9f5224e114d'
    headers = { "Authorization" : "OAuth {}".format(app_access_token) }
    url = 'https://graph.mapillary.com/{}?fields=thumb_2048_url&access_token={}'.format(image_id,app_access_token)
    r = requests.get(url, headers)
    data1 = r.json()
    image_url = data1['thumb_2048_url']
    #SAVE IMAGES AS JPGS INTO SEQUENCE FOLDER
    if not os.path.exists(sequence_name):
        os.makedirs(sequence_name)
    with open('{}/{}.jpg'.format(sequence_name, image_id), 'wb') as handler:
        image_data = requests.get(image_url, stream=True).content
        handler.write(image_data)
    

def save_images_from_sequence(sequence_id, sequence_name):
    #TOKEN TO ACCESS AND INTERACT WITH API
    app_access_token = 'MLY|5472920012744597|0a1926edd41ea7b4c4d7d9f5224e114d'
    #ASKING FOR EACH IMAGE ID IN THE SEQUENCE
    url = 'https://graph.mapillary.com/image_ids?sequence_id={}&access_token={}'.format(sequence_id,app_access_token)

    headers = { "Authorization" : "OAuth {}".format(app_access_token) }
    #STORE RESPONSE
    response = requests.get(url, headers)
    data = response.json()
    #CREATE FOLDER TO STORE IMAGES IN COMPUTER
    if not os.path.exists(sequence_name):
        os.makedirs(sequence_name)

    #LOOP THROUGH EACH IMAGE ID TO GET IMAGE URLS
    for value in data.values():
        for i in value:
             for image_id in i.values():
                save_image(image_id, sequence_name)

def get_location_of_image(image_id):
    #TOKEN TO ACCESS AND INTERACT WITH API
    app_access_token = 'MLY|5472920012744597|0a1926edd41ea7b4c4d7d9f5224e114d'
    #ASKING FOR EACH IMAGE ID IN THE SEQUENCE
    url = 'https://graph.mapillary.com/{}?fields=geometry,captured_at&access_token={}'.format(image_id,app_access_token)

    headers = { "Authorization" : "OAuth {}".format(app_access_token) }
    response = requests.get(url, headers)
    location_data = response.json()
    location_list = list(location_data.values())
    location_dict = location_list[0]
    timestamp = int(location_list[1]/1000)
    coords = location_dict["coordinates"]
    coords[1], coords[0] = coords[0], coords[1]
    locator = Nominatim(user_agent='myGeocoder')
    location = locator.reverse(coords)
    address = location.address
    lat, lon = coords[0], coords[1]
    return address, lat, lon, timestamp