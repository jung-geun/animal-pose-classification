import json
import cv2
import matplotlib.pyplot as plt
from IPython.display import Image
# %matplotlib inline

def get_img_coord(ref):
    with open(ref) as f:
        json_object = json.load(f)
    annotations = json_object["annotations"]
    frame = annotations[0]     # 0번 frame
    # print(type(annotations))     # list
    # print(type(frame))     # dict
    # for keypoints, coord in frame["keypoints"].items():
    #     print(f"{keypoints}: x-{coord['x']}, y-{coord['y']}")
    coords = []
    for i in range(15):
        if type(frame["keypoints"].get(str(i+1), "null")) is dict:
            # dic=dict(key1=100,key=200,key3=300)
            coords.append(list(frame["keypoints"].get(str(i+1), "null").values()))
        else :
            coords.append([None, None])
    print(coords)
    # key_list = list(json_object.keys())
    return coords

def annotate_image(img_path, json_path):
    img = cv2.imread(img_path)
    points = get_img_coord(json_path)
    for i, (x, y) in enumerate(points):
        if x is not None or y is not None :
            # draw a small circle at the point
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            # add a label with the point number
            cv2.putText(img, str(i+1), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

image = "./data/BODYLOWER/20201025_dog-bodylower-000086.mp4/frame_0_timestamp_0.jpg"
info = "./label/BODYLOWER/20201025_dog-bodylower-000086.mp4.json"
annotated_img = annotate_image(image, info)

# Save the image to a file
cv2.imwrite('output.jpg', annotated_img)

# Display the image using IPython.display.Image()
display(Image(filename='output.jpg'))

# Display the image using matplotlib
out_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
plt.imshow(out_img)