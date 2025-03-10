from fastapi import FastAPI, UploadFile, File, Response
from PIL import Image
import io
from depther import Depther
import uvicorn

app = FastAPI()

# Initialisation du Depther au démarrage
depther = Depther(
    backbone_size="small",
    head_type="dpt",
    head_dataset="nyu"
)

@app.post("/depth/image")
async def process_image(file: UploadFile = File(...)):
    # Lire l'image depuis le fichier uploadé
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convertir en RGB si nécessaire
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Obtenir la carte de profondeur
    depth_image = depther.get_depth_for_image(image)
    
    # Convertir l'image de profondeur en bytes
    img_byte_arr = io.BytesIO()
    depth_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 