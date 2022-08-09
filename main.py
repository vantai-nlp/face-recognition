import os
import cv2
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
import pickle
from torchvision import transforms
from scipy import spatial


mtcnn = MTCNN()
def detect_face(img, threshold=0.9):
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        boxes_ = []
        for i, prob in enumerate(probs):
            if prob  >= threshold:
                boxes_.append(boxes[i])
        return boxes_
    return None


face_net = InceptionResnetV1(pretrained='vggface2').eval()
def extract_feature_face(face):
    face = transforms.ToTensor()(face)
    feature = face_net(face.unsqueeze(0))
    feature = feature.detach().numpy()[0]
    return feature


def add_face(database_dir):
    
    name_id = input('input name_id: ')
    face_dir = os.path.join(database_dir, name_id)
    if os.path.exists(face_dir):
        yorn = input('do you sure remove available name_id (y/n) ? ')
        if yorn == 'n':
            return False
        else:
            cmd = f'rm -rf {face_dir}'
            os.system(cmd)
    os.makedirs(face_dir)

    if os.path.exists('./features.pkl'):
        with open('./features.pkl', 'rb') as file:
            idName_to_features = pickle.load(file)
    else:
        idName_to_features = {}
    idName_to_features[name_id] = []

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while(time.time() - start_time < 5):
        _, frame = cap.read()
        cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        boxes = detect_face(cvt_frame)
        if boxes != None and len(boxes) == 1:
            xmin, ymin, xmax, ymax = [int(i) for i in boxes[0]]
            face_img = frame[ymin:ymax, xmin:xmax].copy()
            face_img = cv2.resize(face_img, (160, 160))
            feature = extract_feature_face(face_img)
            idName_to_features[name_id].append(feature)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0))

        cv2.imwrite(os.path.join(face_dir, str(time.time())+'.jpg'), face_img)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    with open('./features.pkl', 'wb') as file:
        pickle.dump(idName_to_features, file)
    cap.release()
    cv2.destroyAllWindows()
    return True


def remove_face(database_dir):

    name_id = input('the name_id you want to remove from database: ')
    face_dir = os.path.join(database_dir, name_id)
    if os.path.exists(face_dir):
        yorn = input('do you sure remove available name_id (y/n) ? ')
        if yorn == 'n':
            return False
        else:
            cmd = f'rm -rf {face_dir}'
            os.system(cmd)
            with open('./features.pkl', 'rb') as file:
                idName_2_features = pickle.load(file)
            idName_2_features.pop(name_id)
            with open('./features.pkl', 'wb') as file:
                pickle.dump(idName_2_features, file)
            return True
    else:
        print("the name_id now isn't available in the database")
        return False


def inference():
    
    if os.path.exists('./features.pkl'):
        with open('./features.pkl', 'rb') as file:
            idName_to_features = pickle.load(file)
    else:
        idName_to_features = {}
    
    def recognize_face(face_img, threshold=0.9):
        if len(idName_to_features) == 0:
            return 'None'
        else:
            face_img = cv2.resize(face_img, (160, 160))
            feature = extract_feature_face(face_img)
            best_cosine, name_id = 0, 'None'
            for idname in idName_to_features:
                for f in idName_to_features[idname]:
                    cosine = 1 - spatial.distance.cosine(feature, f)
                    if cosine > best_cosine:
                        best_cosine = cosine
                        name_id = idname
            if best_cosine > 0.8:
                return name_id
            else:
                return 'None'

    cap = cv2.VideoCapture(0)
    while(True):
        _, frame = cap.read()
        start_time = time.time()
        cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        boxes = detect_face(cvt_frame)
        if boxes != None and len(boxes):
            for box in boxes:
                xmin, ymin, xmax, ymax = [int(i) for i in box]
                face_img = frame[ymin:ymax, xmin:xmax].copy()
                text = recognize_face(face_img)
                cv2.putText(frame, text, (xmin+3, ymin-3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0))
        cv2.putText(frame, 'FPS: '+str(round(1/(time.time()-start_time), 2)), (20, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__=='__main__':
    database_dir = './database'
    if os.path.exists(database_dir) == False:
        os.mkdir(database_dir)

    while True:
        cmd = input('command: ')
        if cmd == '1':
            add_face(database_dir)
        elif cmd == '2':
            remove_face(database_dir)
        elif cmd == '3':
            inference()
        else:
            break