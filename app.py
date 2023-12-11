from ultralytics import YOLO
import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import csv


def get_image_from_url(url):
    response = requests.get(url)
    image = response.content
    image = Image.open(BytesIO(image))
    image = np.asarray(image)
    return image

def get_image_from_file(path_to_file):
    image = Image.open(path_to_file)
    image = np.asarray(image)
    return image

def ecrire_dans_csv(str1, str2, val, nom_fichier):
    entetes = ['images', 'prediction', 'certitude']

    if not os.path.isfile(nom_fichier):
        # Si le fichier n'existe pas, on crée le fichier avec les en-têtes
        with open(nom_fichier, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(entetes)

    # Écrire les données à la fin du fichier
    with open(nom_fichier, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str1, str2, val])
    
def display_result(detailed_predictions, c2, name_image):
    
    n_boar = 0
    n_deer = 0
    print(detailed_predictions)
    for animal in detailed_predictions:
        ecrire_dans_csv(name_image, animal[1], animal[0], "prédictions.csv")
        if animal[1]=="Wild_Boar":
            n_boar += 1
        else:
            n_deer += 1
    print("LA DETECTION")
    print(n_boar," ", n_deer)
    if n_boar*n_deer!=0:
        c2.write(f"Au total {len(detailed_predictions)} animaux sauvages détectés sur cette image.\n\n")
        c2.write(f"Nombre de sanglier(s): {n_boar}")
        c2.write(f"Nombre de cerf(s): {n_deer}")
        
    elif n_boar>0:
        c2.write(f"Au total {len(detailed_predictions)} animaux sauvages détectés sur cette image.\n\n")
        c2.write(f"Nombre de sanglier(s): {n_boar}")
    elif n_deer>0:
        c2.write(f"Au total {len(detailed_predictions)} animaux sauvages détectés sur cette image.\n\n")
        c2.write(f"Nombre de cerf(s): {n_deer}")
    else:
        c2.write("Pas de cerfs ni de sangliers détectés sur cette image")
        
    

def on_upload_bug(file, model_weight_path, conf):
    image = Image.open(BytesIO(file))
    image = image.resize((400, 400))
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    #image = cv2.imread(file)
    model = YOLO(model_weight_path)
    #print("LE TYPE DE LIM")
    #print(type(image))
    #print(image.shape)
    #image = cv2.resize(image, (640, 640))
    results = model(image)[0]
    initial_image = results.orig_img

    detailed_predictions = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        print("score: ", score)
        print("classe_id: ", class_id)
        class_id = int(class_id)
        if class_id == 0:
            class_label = "Deer"
            class_color = (80, 200, 225)
        else:
            class_label = "Wild_Boar"
            class_color = (240, 100, 250)
        
        if score>conf:
            image_res = cv2.rectangle(initial_image,(int(x1), int(y1)), (int(x2), int(y2)), class_color, 3)
            image_res = cv2.putText(image_res, class_label + " "+str(int(score*100)) + "%", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, class_color, 3, cv2.LINE_AA)
        
        detailed_predictions.append((score, class_label))


        return image_res, detailed_predictions

def on_upload(results, conf):

    initial_image = results.orig_img

    detailed_predictions = []
    print(results.boxes.data.tolist())
    print(len(results.boxes.data.tolist()))
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        print("score: ", score)
        print("classe_id: ", class_id)
        class_id = int(class_id)
        if class_id == 0:
            class_label = "Deer"
            class_color = (80, 200, 225)
        else:
            class_label = "Wild_Boar"
            class_color = (240, 100, 250)
        print(class_label)
        if score>conf:
            detailed_predictions.append((score, class_label))
            image_res = cv2.rectangle(initial_image,(int(x1), int(y1)), (int(x2), int(y2)), class_color, 2)
            image_res = cv2.putText(image_res, class_label + " "+str(int(score*100)) + "%", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 1, cv2.LINE_AA)
        else:
            image_res = initial_image
        


    return image_res, detailed_predictions 


def on_upload_bug(file, model_weight_path, conf):
    image = Image.open(BytesIO(file))
    image = image.resize((400, 400))
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    #image = cv2.imread(file)
    model = YOLO(model_weight_path)
    #print("LE TYPE DE LIM")
    #print(type(image))
    #print(image.shape)
    #image = cv2.resize(image, (640, 640))
    results = model(image)[0]
    initial_image = results.orig_img

    detailed_predictions = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        print("score: ", score)
        print("classe_id: ", class_id)
        class_id = int(class_id)
        if class_id == 0:
            class_label = "Deer"
            class_color = (80, 200, 225)
        else:
            class_label = "Wild_Boar"
            class_color = (240, 100, 250)
        
        if score>conf:
            image_res = cv2.rectangle(initial_image,(int(x1), int(y1)), (int(x2), int(y2)), class_color, 3)
            image_res = cv2.putText(image_res, class_label + " "+str(int(score*100)) + "%", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, class_color, 3, cv2.LINE_AA)
        
        detailed_predictions.append((score, class_label))


        return image_res, detailed_predictions  
def app_basic_display():
    st.markdown(
    """
    <div style='text-align:center; border:1px solid black; padding:10px'>
        <h1>Reconnaissance d'animaux sauvages</h1>
    </div>
    """,
    unsafe_allow_html=True) 
    
    st.write("\n\n")
     
     
def main():
    app_basic_display()
    upload = st.file_uploader("Chargez l'image que vous souhaitez détecter",
                           type=['png', 'jpeg', 'jpg'])

    st.write("\n\n")
    c1, c2 = st.columns(2)
    conf = 0.6
    if upload:
        name_image = "images/"+upload.name
        image = cv2.imread(name_image)
        #model = YOLO("yolov8n.pt")
        path_w = 'weights'
        model = YOLO(path_w + "/best.pt")
        results = model.predict(image)
        
        
        result_image, predictions = on_upload(results[0], conf)
        c1.image(result_image)
        display_result(predictions, c2, name_image)
        #c1.image((Image.open(result_image)))
  


if __name__ == "__main__":
    main()



