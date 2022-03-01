from fastapi import FastAPI
from enum import Enum
app = FastAPI()

@app.get('/hello/{name}')
async def hello(name):
    return 'hello! welcome to fastapi tutorials {}'.format(name)
    
food_items = {
    'indian': ['samosa', 'Dosa'],
    'american': ['hot dog', 'apple pie'],
}
class AvailableCuisines(str, Enum):
    indian = 'indian'
    american = 'american'
    

@app.get('/get_items/{cuisine}')
async def get_items(cuisine: AvailableCuisines):
    return food_items[cuisine]