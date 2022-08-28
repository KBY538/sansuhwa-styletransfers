import numpy as np
import PIL.Image as Image
import cv2

def pil2cv2(img):
    return np.array(img)

def cv22pil(imgarray):
    return Image.fromarray(imgarray)

def figure_out(imgs, size=256):

    temp = Image.new('RGB',(len(imgs)*size, size), (255,255,255))

    for i, img in enumerate(imgs):
        if type(img) == np.ndarray:
            img = cv22pil(img)
        
        img = img.resize((size, size))
    
        temp.paste(img,(i*size,0))

    return temp

def make_alpha(img):
    
    if type(img) != np.ndarray:
        img = pil2cv2(img)
        
    b_channel, g_channel, r_channel = cv2.split(img)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return img_BGRA

def make_nonalpha(img):
    
    if type(img) != np.ndarray:
        img = pil2cv2(img)

    if img.shape[-1] != 4:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def multiply_effect(under, over, times=1):

    if under.shape[-1] != 4:
        under = cv2.cvtColor(under, cv2.COLOR_BGR2BGRA)

    if over.shape[-1] != 4:
        over = cv2.cvtColor(over, cv2.COLOR_BGR2BGRA)

    under_f = under.astype(np.float32)
    over_f = over.astype(np.float32)

    under_normalized = under_f/255
    over_normalized = over_f/255

    mul = np.multiply(under_normalized, over_normalized)

    output = mul*255
    output = output.astype(np.uint8)

    output = np.clip(output, 0, 255)

    output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
    
    return output

def make_mask(labelmap, size=256):

    if type(labelmap) != np.ndarray:
        labelmap = pil2cv2(labelmap)
    
    if labelmap.shape[-1] == 4:
        labelmap = make_nonalpha(labelmap)
    
    labelmap = cv2.resize(labelmap, (size, size))

    forethings = {
                "앞산": (144, 238, 144),
                "바위": (153, 136, 119),
                "풀": (128, 128, 240),
                "가까운 나무": (19, 69, 139),
            }
            
    contour = np.zeros((size, size, 3), dtype='uint8')

    for pixel_value in forethings.values():
        
        mask = cv2.inRange(labelmap, pixel_value, pixel_value)
        range_image = cv2.bitwise_and(labelmap, labelmap, mask=mask)
        contour = cv2.add(contour, range_image)

    return contour

def make_contour_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    gradient = gradient.transpose((2, 1, 0))[0]
    gradient = gradient.T

    temp = np.ones(gradient.shape, dtype=gradient.dtype) * 255

    gradient_inverse = temp-gradient

    return gradient, gradient_inverse

def contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return result

def merge_sansu(labelmap, from_spade, from_nst, alpha=0.5, size=256):

    threshold = {
                'contour': (135,135,135),
                'shape': (80, 80, 80)
                }

    if type(labelmap) != np.ndarray:
        labelmap = pil2cv2(labelmap)
        
    if type(from_spade) != np.ndarray:
        from_spade = pil2cv2(from_spade)

    if type(from_nst) != np.ndarray:
        from_nst = pil2cv2(from_nst)

    from_nst = contrast(from_nst)

    # 135 이하 선 뽑기
    contour_region = cv2.inRange(from_nst, (0, 0, 0), threshold['contour'])
    contour_region_image = np.ones(contour_region.shape, dtype=contour_region.dtype) * 255
    contour_region_image = np.stack(((contour_region_image),)*3, axis=-1)
    cv2.copyTo(from_nst, contour_region, contour_region_image)
    
    contour_region_image = cv2.resize(contour_region_image, (size, size))
    
    # 80 이하 선 뽑기
    shape_region = cv2.inRange(from_nst, (0, 0, 0), threshold['shape'])
    shape_region_image = np.ones(contour_region.shape, dtype=contour_region.dtype) * 255
    shape_region_image = np.stack(((shape_region_image),)*3, axis=-1)
    cv2.copyTo(from_nst, shape_region, shape_region_image)

    shape_region_image = cv2.resize(shape_region_image, (size, size))

    # 라벨맵의 주요 윤곽 영역
    contour_mask, shape_mask = make_contour_mask(make_mask(labelmap))
    contour_mask = cv2.resize(contour_mask, (256, 256))
    shape_mask = cv2.resize(shape_mask, (256, 256))

    # 135 이하 선 영역대로 자르기
    contour_image = np.ones(contour_region_image.shape, dtype=contour_region_image.dtype) * 255
    cv2.copyTo(contour_region_image, contour_mask, contour_image)
    contour_image = contrast(contour_image)
    
    contour_image = cv2.GaussianBlur(contour_image, (0, 0), 1)

    shape_image = np.ones(shape_region_image.shape, dtype=shape_region_image.dtype) * 255
    cv2.copyTo(shape_region_image, shape_mask, shape_image)
    shape_image = cv2.GaussianBlur(shape_image, (0, 0), 1)

    shape_output = multiply_effect(from_spade, shape_image)

    shape_output = cv2.addWeighted(from_spade, alpha, shape_output, 1-alpha, 0)

    final_output = contour_output = multiply_effect(shape_output, contour_image)
    
    return from_nst, contour_image, shape_image, final_output