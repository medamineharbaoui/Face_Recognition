import sys 
import pygame 
import cv2
import numpy as np
import mtcnn
from architecture import InceptionResNetV2
from scipy.spatial.distance import cosine
import pickle
from Button import Button

confidence_t = 0.9
recognition_t = 0.5
required_size = (160, 160)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def l2_normalizer(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def main_program(results_store):
    def get_face(img, box):
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)

    def get_encode(face_encoder, face, size):
        face = normalize(face)
        face = cv2.resize(face, size)
        encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
        return encode
    
    def detect(img, detector, encoder, encoding_dict):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        for res in results:
            if res['confidence'] < confidence_t:
                continue
            face, pt_1, pt_2 = get_face(img_rgb, res['box'])
            encode = get_encode(encoder, face, required_size)
            encode = l2_normalizer(encode.reshape(1, -1))[0]
            name = 'unknown'

            distance = float("inf")
            for db_name, db_encode in encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            results_store.append((name, res['confidence'], distance))
        
            if name == 'unknown':
                cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img


    def mainMenu():
        pygame.init()
        WIDTH, HEIGHT = (1280, 720)
        Screen = pygame.display.set_mode((WIDTH, HEIGHT))
        BG = pygame.image.load("assets/BG_2.png")
        font = pygame.font.Font('assets/abg.ttf', 20)
        pygame.display.set_caption("Face Recognition Project ")

        results_store.clear()  # Clear previous results if any

        while True:
            Screen.blit(BG, (0, 0))

            MENU_MOUSE_POS = pygame.mouse.get_pos()

            START_BUTTON = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(730, 350), 
                                text_input="Start", font=getFont(50), base_color="Black", hovering_color="White")
            
            QUIT_BUTTON = Button(image=pygame.image.load("assets/Quit Rect.png"), pos=(730, 480), 
                                text_input="QUIT", font=getFont(50), base_color="Black", hovering_color="White")

            for button in [START_BUTTON, QUIT_BUTTON]:
                button.changeColor(MENU_MOUSE_POS)
                button.update(Screen)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if START_BUTTON.checkForInput(MENU_MOUSE_POS):
                        start_face_recognition(results_store)
                    
                    if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                        pygame.quit()
                        sys.exit()

            pygame.display.update()

    def start_face_recognition(results_store):
        required_shape = (160, 160)
        face_encoder = InceptionResNetV2()
        path_m = "facenet_keras_weights.h5"
        face_encoder.load_weights(path_m)
        encodings_path = 'encodings/encodings.pkl'
        face_detector = mtcnn.MTCNN()
        encoding_dict = load_pickle(encodings_path)

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("CAM NOT OPENED")
                break

            frame = detect(frame, face_detector, face_encoder, encoding_dict)

            cv2.imshow('camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):       
                break

        cap.release()
        cv2.destroyAllWindows()

        with open('results_store.pkl', 'wb') as f:
            pickle.dump(results_store, f)

    def getFont(size):
        return pygame.font.Font("assets/abg.ttf", size)

    mainMenu()

if __name__ == "__main__":
    main_program([])
