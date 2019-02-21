import os
import stat
import urllib.request
import playsound

class TTS():
    def __init__(self, resource):
        self.resource = resource
        client_id = "jqhycv20tf"
        client_secret = "EFh6RhwU8DDuLX3O1qaSCqc2jRgu2P6asUl6wmiR"

        url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"
        self.request = urllib.request.Request(url)
        self.request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
        self.request.add_header("X-NCP-APIGW-API-KEY",client_secret)

    def text_2_speech(self):
        count = 0
        while(1):
            if(self.resource.is_new_annotation() == 1):
                text = self.resource.get_annotation()
                encText = urllib.parse.quote(text)
                data = "speaker=jinho&speed=-2&text=" + encText
                response = urllib.request.urlopen(self.request, data=data.encode('utf-8'))
                rescode = response.getcode()

                if(rescode==200):
                    response_body = response.read()
                    path = "./_data/tts/"+str(count)+".mp3"
                    with open(path, 'wb') as f:
                        f.write(response_body)
                    playsound.playsound(path)

                    count = count + 1

                else:
                    print("Error Code:" + rescode)

