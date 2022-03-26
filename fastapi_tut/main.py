from fastapi import FastAPI, UploadFile, File
from enum import Enum
import uvicorn
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


@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    bytes = await file.read()
    

    return

@app.get('/ping')
async def ping():
    return 'The website is alive.'

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8080)