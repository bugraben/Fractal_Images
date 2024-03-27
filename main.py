import cv2 as cv
import numpy as np
import datetime as dt

IMG_SIZE = 100

now = dt.datetime.now()
img_ = cv.imread('photos/talip.png')
img = cv.resize(img_, (IMG_SIZE, IMG_SIZE))


def inspect_3d_array(array):
    rows, cols, chas = array.shape
    rows, cols = list(map(lambda x: int(x/4), [rows, cols]))
    for row in range(rows):
        print('\n')
        for col in range(cols):
            print(array[row*4, col*4], end='')


def imshow(img, size=(500, 500), window_name='Image'):
    img_rsz = cv.resize(img, size)
    cv.imshow(window_name, img_rsz)
    if cv.waitKey(0):
        cv.destroyAllWindows()


def imwrite(out, name='', folder='outputs', file_extension='.png'):
    '''dont forget to put a dot before extension'''
    filepath = folder + '/' + name + '_' + str(IMG_SIZE) + 'px' + '_' + now.strftime('%H%m_%d%b') + file_extension
    cv.imwrite(filepath, out)


imshow(img, window_name='Tulip')

mask = np.empty_like(img)
# imshow(mask)
# 500x500luk resmi al

# her piksel icin icin renkleri normalize et. normalize degerleri o konuma gelecek karenin tum piksellerine katsayi
# olarak ver.

# 0-1 arasi normalizasyon formulu
#           y = (x – min) / (max – min)

for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        for cha in range(img.shape[2]):
            mask[row, col, cha] = img[row, col, cha].astype(np.float16) * 10 /   \
                                  ((img[row, col].max() - img[row, col].min()).astype(np.float16) + 1)

# sifir ile bolumden kurtulmak icin paydada +1 var
# her piksele kendi icinde normalizasyon uyguladim

# inspect_3d_array(mask)

output = np.zeros((IMG_SIZE**2, IMG_SIZE**2, 3), dtype=np.uint8)
output.shape

for ch in range(3):
    for row in range(IMG_SIZE):
        rowB = row * IMG_SIZE
        for col in range(IMG_SIZE):
            colB = col * IMG_SIZE
            output[rowB:rowB+IMG_SIZE, colB:colB+IMG_SIZE, ch] = img[:, :, ch] + img[row, col, ch]


imshow(output)
imwrite(output, name='talip_output')

