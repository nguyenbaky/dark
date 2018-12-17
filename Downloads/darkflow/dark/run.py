import cv2
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import numpy as np
# %config InlineBackend.figure_format = 'svg'

img = cv2.imread('sample_img/13.jpeg',1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 4750
# 5125 2.75 2.69
# 5750 2.61 2.57
# 5875 2.4 2.74
# 6125 1.7 2.55
options = {
    'model':'cfg/yolov2-tiny-voc-3c.cfg',
    'load':4750,
    'threshold':0.1,
    'gpu':1.0
}

tfnet = TFNet(options)
results = tfnet.return_predict(img)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
print(results)

for color,result in zip(colors,results):
    tl = (result['topleft']['x'],result['topleft']['y'])
    br = (result['bottomright']['x'],result['bottomright']['y'])
    label = result['label']
    confidence = result['confidence']
    test = '{}: {:.0f}%'.format(label,confidence*100)
    cv2.rectangle(img,tl,br,color,2)
    cv2.putText(img,test, (tl[0],tl[1]+30) ,cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,0,0), 2)

plt.imshow(img)
plt.show()