import requests

url = "http://192.168.18.29/update"
#on = requests.get(url, params={"state": 1})
#off = requests.get(url, params={"state": 0})


def luz(estado):
    # 1 = ligado
    # 0 = desligado
    if estado:
        r = requests.get(url, params={"state": 1})
        print(r)
    else:
        r = requests.get(url, params={"state": 0})
