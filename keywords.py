import http.client, urllib.request, urllib.parse, urllib.error, base64

def find_keywords(text):

    headers = { 'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': 'subscription key here' }
    
    params = urllib.parse.urlencode({
    })
    
    body = {
      "documents": [
        {
          "language": "en",
          "id": "1",
          "text": text
          }
      ]
    }
    
    try:
        conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
        conn.request("POST", "/text/analytics/v2.0/keyPhrases?%s" % params, str(body), headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return data
    except Exception as e:
        print("Error: {0}, {1}".format(e.errno, e.strerror))
