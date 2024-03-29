from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import Stream
import json
import movie_review_classification as s

#consumer key, consumer secret, access token, access secret.
ckey="LEyosuVH4UdT84Rmx4aS0zm0f"
csecret="pVP7z4fatVl9DGryYJzE3uy9atqutR006uHVQPOMIutRMSqRki"
atoken="1394351988628885511-IRn63bSpqIR3lWhSrELWpxziOpwCgk"
asecret="rSl7ApnPuVcwh73J9oGYwcnGZ41htpn5ziG4YcpH8N5km"

#from twitterapistuff import *

class listener(Stream):

    def on_data(self, data):

        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

#auth = OAuthHandler(ckey, csecret)
#auth.set_access_token(atoken, asecret)

twitterStream = Stream(ckey, csecret, atoken, asecret)
twitterStream.filter(track=["happy"])
