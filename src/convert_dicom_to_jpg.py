import dicomsdl
import numpy as np
from PIL import Image
from utils.apply_voilut import DicomsdlMetadata, ApplyWindowing


def convert_dcm_to_img(dcm_path:str) -> np.array:
    dcm = dicomsdl.open(dcm_path)
    meta = DicomsdlMetadata(dcm)
    info = dcm.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    ori_dtype = info['dtype']
    img = np.empty(shape, dtype=ori_dtype)
    index= 0
    dcm.copyFrameData(index, img)
    if meta.window_widths:
        convert_img = ApplyWindowing.apply_windowing_np_v2(img,
                                window_width=meta.window_widths[0],
                                window_center=meta.window_centers[0],
                                voi_func=meta.voilut_func,
                                y_min=0,
                                y_max=255,
                                )
    else:
        print('No windowing param!')
        convert_img = ApplyWindowing.min_max_scale(img)
        convert_img = convert_img * 255

    if meta.invert:
        convert_img = 255 - convert_img

    return convert_img

def save_np_to_jpg(arr, output_path):
    im = Image.fromarray(arr)
    im = im.convert("L")
    im.save(output_path)

if __name__ == '__main__':
    dcm_path = 'data/test.dcm'
    output_path = 'data/test.jpg'
    np_image = convert_dcm_to_img(dcm_path)
    save_np_to_jpg(np_image, output_path)

